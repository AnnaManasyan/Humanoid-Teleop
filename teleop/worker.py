import json
import lzma
import os
import pickle
import queue
import shutil
import sys
import threading
import time
import zipfile
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq
from vr import VuerTeleop
from writers import AsyncImageWriter, AsyncWriter

from utils.logger import logger

# from turbojpeg import TJPF_BGR, TurboJPEG


FREQ = 30
DELAY = 1 / FREQ

from constants import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class TeleoperatorProcess:
    def __init__(self, teleop_shm_array, img_shm_name, kill_event, ir_data_queue):
        self.teleop_shm_array = teleop_shm_array
        self.kill_event = kill_event
        self.ir_data_queue = ir_data_queue

        # Connect to the shared memory for images created by the worker
        self.img_shm = shared_memory.SharedMemory(name=img_shm_name)
        height, width = 720, 1280  # Match the dimensions expected by teleoperator
        self.img_array = np.ndarray(
            (height, width, 3), dtype=np.uint8, buffer=self.img_shm.buf
        )

        # Pass the shared memory name so that VuerTeleop attaches to the same memory
        self.teleoperator = VuerTeleop("inspire_hand.yml", img_shm_name)

    def _ir_loop(self):
        """Thread that processes IR image data and copies it to shared memory."""
        while not self.kill_event.is_set():
            try:
                raw_ir_data = self.ir_data_queue.get(timeout=0.01)
                raw_ir_np = np.frombuffer(raw_ir_data, dtype=np.uint8)
                decoded_frame = cv2.imdecode(raw_ir_np, cv2.IMREAD_COLOR)
                if decoded_frame is not None:
                    resized_frame = cv2.resize(
                        decoded_frame, (1280, 720), interpolation=cv2.INTER_LINEAR
                    )
                    np.copyto(self.img_array, resized_frame)
                else:
                    logger.warning("Teleoperator: Failed to decode IR frame.")
            except queue.Empty:
                pass
            # A small sleep can help reduce CPU usage
            time.sleep(0.005)

    def _step_loop(self):
        """Thread that repeatedly calls teleoperator.step() and updates shared state."""
        while not self.kill_event.is_set():
            try:
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                    self.teleoperator.step()
                )
                self.teleop_shm_array[0:9] = head_rmat.flatten()
                self.teleop_shm_array[9:25] = left_pose.flatten()
                self.teleop_shm_array[25:41] = right_pose.flatten()
                self.teleop_shm_array[41:48] = np.array(left_qpos).flatten()
                self.teleop_shm_array[48:55] = np.array(right_qpos).flatten()
            except Exception as e:
                logger.error(f"Teleoperator step error: {e}")
            time.sleep(0.01)

    def run(self):
        # logger.info(f"Teleoperator process started with PID {os.getpid()}")
        # Start IR processing and teleoperator stepping in separate threads
        ir_thread = threading.Thread(target=self._ir_loop, daemon=True)
        step_thread = threading.Thread(target=self._step_loop, daemon=True)
        ir_thread.start()
        step_thread.start()

        try:
            while not self.kill_event.is_set():
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Teleoperator process encountered an error: {e}")
        finally:
            logger.info("Teleoperator process shutting down")
            self.kill_event.set()
            ir_thread.join()
            step_thread.join()
            self.teleoperator.shutdown()
            self.img_shm.close()
            logger.info("Teleoperator process exited")


class RobotDataWorker:
    def __init__(self, shared_data, robot_shm_array, teleop_shm_array, robot="h1"):
        self.robot = robot
        self.modality = Modality(self.robot)
        self.shared_data = shared_data
        self.kill_event = shared_data["kill_event"]
        self.session_start_event = shared_data["session_start_event"]
        self.end_event = shared_data["end_event"]  # TODO: redundent
        self.camera_mode = shared_data["camera_mode"]
        # self.h1_lock = Lock()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        if robot == "h1":
            self.socket.connect("tcp://192.168.123.162:5556")
        else:
            self.socket.connect("tcp://192.168.123.164:5556")

        self.robot_shm_array = robot_shm_array
        self.teleop_shm_array = teleop_shm_array
        self.depth_kill_event = Event()
        self.teleop_kill_event = Event()

        height, width = 720, 1280  # Match the dimensions expected by teleoperator
        self.img_shm = shared_memory.SharedMemory(
            create=True, size=height * width * 3 * np.uint8().itemsize
        )
        self.img_array = np.ndarray(
            (height, width, 3), dtype=np.uint8, buffer=self.img_shm.buf
        )

        self.depth_queue = Queue()
        self.async_image_writer = AsyncImageWriter()

        self.depth_proc = Process(
            target=self.depth_writer_process,
            args=(self.depth_queue, self.depth_kill_event),
        )

        self.ir_data_queue = Queue()
        self.teleop_proc = Process(
            target=self._run_teleoperator_process,
            args=(
                self.teleop_shm_array,
                self.img_shm.name,
                self.teleop_kill_event,
                self.ir_data_queue,
            ),
        )
        self.teleop_proc.start()
        logger.debug(
            f"Worker started teleoperator process with PID {self.teleop_proc.pid}"
        )

        # resetable vars
        self.frame_idx = 0
        self.last_robot_data = None
        self.robot_data_writer = None

    def _run_teleoperator_process(
        self, teleop_shm_array, img_shm_name, kill_event, ir_data_queue
    ):
        teleop_process = TeleoperatorProcess(
            teleop_shm_array, img_shm_name, kill_event, ir_data_queue
        )
        teleop_process.run()

    def depth_writer_process(self, depth_queue, kill_event):
        while not kill_event.is_set():
            try:
                filename, depth_array = depth_queue.get(timeout=0.5)
                # buffer = io.BytesIO()
                # np.save(buffer, depth_array)
                # depth_bytes = buffer.getvalue()
                compressed_data = lzma.compress(depth_array.tobytes(), preset=0)
                with open(filename, "wb") as f:
                    f.write(compressed_data)
            except queue.Empty:
                continue

    def dump_state(self, filename=None):
        """Dump current system state for debugging"""
        if filename is None:
            filename = f"debug_dump_{time.strftime('%Y%m%d_%H%M%S')}.pkl"

        state = {
            "h1_data": self.robot_shm_array.copy(),
            "teleop_data": self.teleop_shm_array.copy(),
            "frame_idx": self.frame_idx if hasattr(self, "frame_idx") else None,
            "timestamp": time.time(),
        }

        with open(filename, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"State dumped to {filename}")

    def _recv_zmq_frame(self) -> Tuple[Any, Any, Any]:
        logger.debug("worker: start sending request")
        self.socket.send(b"get_frame")

        rgb_bytes, ir_bytes, depth_bytes = (
            self.socket.recv_multipart()
        )  # resource leak due to blocking?
        logger.debug("worker: recv request")

        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        ir_array = np.frombuffer(ir_bytes, np.uint8)
        depth_array = np.frombuffer(depth_bytes, np.uint16).reshape((480, 640))
        logger.debug("worker: processed depth array")

        return rgb_array, ir_array, depth_array

    # def extract_usable(self, row):
    #     if all(val in (0.0, 30000.0) for val in row):
    #         return None, None
    #
    #     if row[0] not in (0.0, 30000.0):
    #         usable = [row[i] for i in [0, 2, 9, 11]]
    #         sensor_type = "A"
    #     elif row[3] not in (0.0, 30000.0):
    #         usable = [row[i] for i in [3, 6, 8]]
    #         sensor_type = "B"
    #     else:
    #         return None, None
    #
    #     return sensor_type, usable

    def format_pressure_data(self, data):
        # sensors = []
        # for idx, row in enumerate(data, start=1):
        #     sensor_type, readings = self.extract_usable(row)
        #     if readings is not None:
        #         sensors.append(
        #             {
        #                 "sensor_id": idx,
        #                 "sensor_type": sensor_type,
        #                 "usable_readings": readings,
        #             }
        #         )
        return data

    def _get_modality_slice(self, robot_data, modality_name):
        start_idx = self.modality.get_start_index(modality_name)
        end_idx = self.modality.get_end_index(modality_name)
        return robot_data[start_idx:end_idx]

    def get_robot_data(self, time_curr):
        logger.debug(f"worker: starting to get robot data")
        # with self.h1_lock:
        robot_data = self.robot_shm_array.copy()

        legstate = self._get_modality_slice(robot_data, "leg")
        armstate = self._get_modality_slice(robot_data, "arm")
        handstate = self._get_modality_slice(robot_data, "hand")

        imustate = {
            "quaternion": self._get_modality_slice(
                robot_data, "imu_quaternion"
            ).tolist(),
            "accelerometer": self._get_modality_slice(
                robot_data, "imu_accelerometer"
            ).tolist(),
            "gyroscope": self._get_modality_slice(
                robot_data, "imu_gyroscope"
            ).tolist(),
            "rpy": self._get_modality_slice(robot_data, "imu_rpy").tolist(),
        }

        # odomstate = {
        #     "position": self._get_modality_slice(
        #         robot_data, "odom_position"
        #     ).tolist(),
        #     "velocity": self._get_modality_slice(
        #         robot_data, "odom_velocity"
        #     ).tolist(),
        #     "rpy": self._get_modality_slice(robot_data, "odom_rpy").tolist(),
        #     "quat": self._get_modality_slice(robot_data, "odom_quaternion").tolist(),
        # }
        if self.modality.has_modality("odom_position"):
            odomstate = {
                "position": self._get_modality_slice(robot_data, "odom_position").tolist(),
                "velocity": self._get_modality_slice(robot_data, "odom_velocity").tolist(),
                "rpy": self._get_modality_slice(robot_data, "odom_rpy").tolist(),
                "quat": self._get_modality_slice(robot_data, "odom_quaternion").tolist(),
            }
        else:
            odomstate = {"position": None, "velocity": None, "rpy": None, "quat": None}


        pressure_state = None
        if self.modality.has_modality("hand_press"):
            pressure_state = self._get_modality_slice(robot_data, "hand_press")

        robot_data = {
            "time": time_curr,
            "robot_type": self.robot,
            "states": {
                "arm_state": armstate.tolist(),
                "leg_state": legstate.tolist(),
                "hand_state": handstate.tolist(),
                "hand_pressure_state": (
                    self.format_pressure_data(
                        pressure_state.reshape(
                            18, 12
                        ).tolist()
                    )
                    if pressure_state is not None
                    else None
                ),
                "imu": imustate,
                "odometry": odomstate,
            },
            "actions": None,
            "image": f"color/frame_{self.frame_idx:06d}.jpg",
            "depth": f"depth/frame_{self.frame_idx:06d}.npy.lzma",
            "lidar": None,
        }
        # logger.debug(f"worker: finish getting robot data")
        return robot_data

    def _receiver_loop(self):
        """Continuously receive ZMQ frames and update the sample-hold buffer.

        Runs for the entire lifetime of the worker (across sessions) so the
        teleoperator display stays live even between episodes.
        """
        while not self.end_event.is_set():
            try:
                rgb_array, ir_array, depth_array = self._recv_zmq_frame()
                mode = self.camera_mode.value
                if mode == CAMERA_MODE_RGB:
                    self._send_image_to_teleoperator(self._make_stereo_rgb(rgb_array))
                elif mode == CAMERA_MODE_OVERLAY:
                    self._send_image_to_teleoperator(
                        self._make_overlay(rgb_array, ir_array)
                    )
                else:
                    self._send_image_to_teleoperator(ir_array)
                with self._holder_lock:
                    self._holder_rgb = rgb_array
                    self._holder_depth = depth_array
            except zmq.ZMQError:
                break  # socket closed during shutdown
            except Exception as e:
                logger.error(f"Worker receiver error: {e}")

    def _saver_loop(self):
        """Drain the save queue and write paired image + JSON for each entry.

        Each queue item contains (frame_idx, rgb, depth, json_str) — both the
        image and the JSON line are always written together so their counts
        can never diverge.
        """
        dirname = self.shared_data["dirname"]
        while not self._saver_kill.is_set() or not self._save_queue.empty():
            try:
                frame_idx, rgb_array, depth_array, robot_data_json = (
                    self._save_queue.get(timeout=0.5)
                )
            except queue.Empty:
                continue

            color_filename = os.path.join(
                dirname, f"color/frame_{frame_idx:06d}.jpg"
            )
            depth_filename = os.path.join(
                dirname, f"depth/frame_{frame_idx:06d}.npy.lzma"
            )
            self.async_image_writer.write_image(color_filename, rgb_array)
            self.depth_queue.put((depth_filename, depth_array))
            self.robot_data_writer.write(robot_data_json)

    def start(self):
        self.depth_proc.start()

        # Persistent sample-hold buffer — receiver fills it, session loop reads it
        self._holder_lock = threading.Lock()
        self._holder_rgb = None
        self._holder_depth = None

        # Single receiver thread for the lifetime of the worker
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, daemon=True
        )
        self._receiver_thread.start()

        try:
            while not self.end_event.is_set():
                logger.info(
                    "Worker: waiting for new session start (session_start_event)."
                )
                while not self.session_start_event.is_set() and not self.end_event.is_set():
                    time.sleep(0.05)  # receiver thread handles ZMQ + teleoperator
                if self.end_event.is_set():
                    break
                logger.info("Worker: starting new session.")
                self.run_session()
                self.async_image_writer.close()
                self.async_image_writer = AsyncImageWriter()
        finally:
            self.socket.close()
            self.context.term()

            self.teleop_kill_event.set()
            self.ir_data_queue.cancel_join_thread()
            self.teleop_proc.join()
            logger.info("Worker: teleoperator process joined.")
            self.img_shm.close()
            self.img_shm.unlink()

            self.depth_kill_event.set()
            self.depth_proc.join()
            logger.info("Worker: exited")

    def _make_stereo_rgb(self, rgb_array):
        """Build a side-by-side stereo frame from the single RGB camera.

        The RealSense has one color camera but two IR cameras, so the IR stream
        is already a genuine stereo pair. RGB is mono, so we duplicate the color
        frame into both halves; otherwise the headset's left/right eye split
        ([:, :w] and [:, w:]) would show each eye only half the scene. This is a
        display-only frame and does not affect the single-width color saved to
        disk (self._holder_rgb stays untouched).
        """
        color = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        if color is None:
            return rgb_array  # fall back; headset just shows the mono frame
        combined = np.hstack((color, color))
        ret, encoded = cv2.imencode(".jpg", combined)
        return encoded if ret else rgb_array

    def _make_overlay(self, rgb_array, ir_array):
        """Blend the RGB color frame on top of the IR stereo frame.

        IR keeps the stereo depth cues (one IR view per eye) while the mono RGB
        color is alpha-blended over both halves. The color camera and IR cameras
        have slightly different viewpoints, so the color is not pixel-perfectly
        registered to the IR depth, but it adds a usable color reference. Falls
        back to plain IR if either frame fails to decode. Display only — the
        single-width color saved to disk (self._holder_rgb) is untouched.
        """
        ir = cv2.imdecode(ir_array, cv2.IMREAD_COLOR)
        color = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        if ir is None or color is None:
            return ir_array
        # Duplicate mono color side-by-side and match the IR frame size so each
        # eye-half overlaps its own IR view.
        color_combined = np.hstack((color, color))
        if color_combined.shape[:2] != ir.shape[:2]:
            color_combined = cv2.resize(color_combined, (ir.shape[1], ir.shape[0]))
        blended = cv2.addWeighted(
            color_combined, CAMERA_OVERLAY_ALPHA, ir, 1.0 - CAMERA_OVERLAY_ALPHA, 0.0
        )
        ret, encoded = cv2.imencode(".jpg", blended)
        return encoded if ret else ir_array

    def _send_image_to_teleoperator(self, ir_array):
        try:
            self.ir_data_queue.put_nowait(ir_array)
        except Exception:
            pass

    def _session_init(self):
        if "dirname" not in self.shared_data:
            logger.error("Worker: failed to get dirname")
            exit(-1)
        self.robot_data_writer = AsyncWriter(
            os.path.join(self.shared_data["dirname"], "robot_data.jsonl")
        )

    def run_session(self):
        self._session_init()

        self._save_queue = queue.Queue()
        self._saver_kill = threading.Event()

        # Wait until the receiver has populated the holder with at least one frame
        while self._holder_rgb is None:
            time.sleep(0.01)

        time.sleep(0.2)  # wait for master to populate shared robot data

        saver = threading.Thread(target=self._saver_loop, daemon=True)
        saver.start()

        try:
            initial_time = time.time()
            while not self.kill_event.is_set():
                # Fixed-rate 30 Hz tick
                next_time = initial_time + self.frame_idx * DELAY
                now = time.time()
                if next_time > now:
                    time.sleep(next_time - now)

                # Sample latest frame from holder (reuses previous if no new frame)
                with self._holder_lock:
                    rgb = self._holder_rgb
                    depth = self._holder_depth

                # Build JSON from current robot shared-memory state
                robot_data = self.get_robot_data(time.time())
                robot_data_json = json.dumps(robot_data)

                # Enqueue paired (image + JSON) — saver writes both atomically
                self._save_queue.put(
                    (self.frame_idx, rgb, depth, robot_data_json)
                )

                self.last_robot_data = robot_data
                self.frame_idx += 1

        finally:
            logger.info("Worker begin exiting.")
            # Signal saver to finish and wait for queue to drain
            self._saver_kill.set()
            saver.join()
            logger.info("Worker: saver thread joined.")
            self.robot_data_writer.close()
            logger.info("Worker: writer closed.")
            self.shared_data["worker_flush_event"].set()
            self.reset()
            logger.info("Worker: closing async image writer.")
            if hasattr(self, "async_image_writer"):
                self.async_image_writer.close()
                logger.info("Worker: async image writer closed.")

            # Zip images and create video in background so next session can start immediately
            dirname = self.shared_data.get("dirname")
            if dirname:
                threading.Thread(
                    target=self._post_process_episode, args=(dirname,), daemon=True
                ).start()

            logger.info("Worker process has exited.")

    def _post_process_episode(self, dirname):
        """Zip color/depth dirs and create video from color frames. Runs in background thread."""
        color_dir = os.path.join(dirname, "color")
        depth_dir = os.path.join(dirname, "depth")

        # Create video from color frames
        if os.path.isdir(color_dir):
            frames = sorted(f for f in os.listdir(color_dir) if f.endswith(".jpg"))
            if frames:
                first = cv2.imread(os.path.join(color_dir, frames[0]))
                if first is not None:
                    h, w = first.shape[:2]
                    video_path = os.path.join(dirname, "video.mp4")
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), FREQ, (w, h))
                    writer.write(first)
                    for fname in frames[1:]:
                        frame = cv2.imread(os.path.join(color_dir, fname))
                        if frame is not None:
                            writer.write(frame)
                    writer.release()
                    logger.info(f"Post-process: created {video_path}")

        # Zip color dir
        if os.path.isdir(color_dir):
            zip_path = os.path.join(dirname, "color.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in sorted(os.listdir(color_dir)):
                    zf.write(os.path.join(color_dir, fname), arcname=os.path.join("color", fname))
            shutil.rmtree(color_dir)
            logger.info(f"Post-process: zipped and removed {color_dir}")

        # Zip depth dir
        if os.path.isdir(depth_dir):
            zip_path = os.path.join(dirname, "depth.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in sorted(os.listdir(depth_dir)):
                    zf.write(os.path.join(depth_dir, fname), arcname=os.path.join("depth", fname))
            shutil.rmtree(depth_dir)
            logger.info(f"Post-process: zipped and removed {depth_dir}")

    def reset(self):
        self.frame_idx = 0
        self.initial_capture_time = None
