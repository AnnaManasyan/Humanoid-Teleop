import logging
import os
import re
import shutil
import sys
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from pathlib import Path

import numpy as np

from constants import *
from master import RobotTaskmaster
from progress import ProgressTracker
from utils.logger import logger
from worker import RobotDataWorker
from StreamDeck.DeviceManager import DeviceManager

FREQ = 30
DELAY = 1 / FREQ

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# # How to use the streaming pedal
# def key_change_callback(deck, key, state):
#     action = "pressed" if state else "released"
#     print(f"Pedal {key} {action}")

# def main():
#     manager = DeviceManager()
#     decks = manager.enumerate()
#     if not decks:
#         print("No Stream Deck devices found.")
#         return
#     for deck in decks:
#         deck.open()
#         deck.set_key_callback(key_change_callback)
#         print(f"Connected: {deck.deck_type()} (serial: {deck.get_serial_number()})")
#     # Keep running
#     try:
#         import threading
#         threading.Event().wait()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         for deck in decks:
#             deck.close()

# if __name__ =="__main__":
#     main()

# Outputs something like:
# (tv) g1@al-tango:~/Desktop/G1/new_try/Humanoid-Teleop/teleop$ python manager.py 
# Connected: Stream Deck Pedal (serial: A00YA43220Z03U)
# Pedal 0 pressed
# Pedal 0 released
# Pedal 2 pressed
# Pedal 2 released
# Pedal 1 pressed
# Pedal 1 released

class TeleopManager:
    def __init__(self, task_name="default_task", robot="h1", debug=False):
        self.task_name = task_name
        logger.info(f"#### (Task: {self.task_name}):")
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.manager = Manager()
        self.shared_dict = self.manager.dict()

        self.shared_dict["kill_event"] = self.manager.Event()
        self.shared_dict["session_start_event"] = self.manager.Event()
        self.shared_dict["failure_event"] = self.manager.Event()
        self.shared_dict["end_event"] = self.manager.Event()  # TODO: redundent
        self.shared_dict["following_paused_event"] = self.manager.Event()
        self.progress_tracker = ProgressTracker()


        if robot == "h1":
            totalsize = (
                H1_sizes.LEG_STATE_SIZE
                + H1_sizes.ARM_STATE_SIZE
                + H1_sizes.HAND_STATE_SIZE
                + H1_sizes.IMU_QUATERNION_SIZE
                + H1_sizes.IMU_ACCELEROMETER_SIZE
                + H1_sizes.IMU_GYROSCOPE_SIZE
                + H1_sizes.IMU_RPY_SIZE
                # + H1_sizes.ODOM_POSITION_SIZE
                # + H1_sizes.ODOM_VELOCITY_SIZE
                # + H1_sizes.ODOM_RPY_SIZE
                # + H1_sizes.ODOM_QUATERNION_SIZE
            )
        elif robot == "g1":
            totalsize = (
                G1_sizes.LEG_STATE_SIZE
                + G1_sizes.ARM_STATE_SIZE
                + G1_sizes.HAND_STATE_SIZE
                + G1_sizes.IMU_QUATERNION_SIZE
                + G1_sizes.IMU_ACCELEROMETER_SIZE
                + G1_sizes.IMU_GYROSCOPE_SIZE
                + G1_sizes.IMU_RPY_SIZE
                + G1_sizes.ODOM_POSITION_SIZE
                + G1_sizes.ODOM_VELOCITY_SIZE
                + G1_sizes.ODOM_RPY_SIZE
                + G1_sizes.ODOM_QUATERNION_SIZE
                + G1_sizes.HAND_PRESS_SIZE
            )
        elif robot == "g1_inspire":
            totalsize = (
                G1_inspire_sizes.LEG_STATE_SIZE
                + G1_inspire_sizes.ARM_STATE_SIZE
                + G1_inspire_sizes.HAND_STATE_SIZE
                + G1_inspire_sizes.IMU_QUATERNION_SIZE
                + G1_inspire_sizes.IMU_ACCELEROMETER_SIZE
                + G1_inspire_sizes.IMU_GYROSCOPE_SIZE
                + G1_inspire_sizes.IMU_RPY_SIZE
                # + G1_inspire_sizes.ODOM_POSITION_SIZE
                # + G1_inspire_sizes.ODOM_VELOCITY_SIZE
                # + G1_inspire_sizes.ODOM_RPY_SIZE
                # + G1_inspire_sizes.ODOM_QUATERNION_SIZE
                # + G1_inspire_sizes.HAND_PRESS_SIZE
            )

        self.robot_data_shm = shared_memory.SharedMemory(
            create=True, size=totalsize * np.dtype(np.float64).itemsize
        )
        self.robot_shm_array = np.ndarray(
            (totalsize,), dtype=np.float64, buffer=self.robot_data_shm.buf
        )

        self.teleop_shm = shared_memory.SharedMemory(
            create=True, size=55 * np.dtype(np.float64).itemsize
        )
        self.teleop_shm_array = np.ndarray(
            (55,), dtype=np.float64, buffer=self.teleop_shm.buf
        )

        def run_taskmaster():
            taskmaster = RobotTaskmaster(
                self.task_name,
                self.shared_dict,
                self.robot_shm_array,
                self.teleop_shm_array,
                robot,
            )
            taskmaster.start()

        def run_dataworker():
            taskworker = RobotDataWorker(
                self.shared_dict, self.robot_shm_array, self.teleop_shm_array, robot
            )
            taskworker.start()

        self.taskmaster_proc = Process(target=run_taskmaster)
        self.dataworker_proc = Process(target=run_dataworker)

    def start_processes(self):
        logger.info("Starting taskmaster and dataworker processes.")
        self.taskmaster_proc.start()
        self.dataworker_proc.start()

    def update_directory(self):
        self.shared_dict["dirname"] = self.progress_tracker.get_next()
        os.makedirs(self.shared_dict["dirname"], exist_ok=True)
        os.makedirs(os.path.join(self.shared_dict["dirname"], "color"), exist_ok=True)
        os.makedirs(os.path.join(self.shared_dict["dirname"], "depth"), exist_ok=True)
        # dirname = self.shared_dict["dirname"]

        # logger.info(f"Data directory set to: {dirname}")

    def start_session(self):
        self.update_directory()
        self.shared_dict["failure_event"].clear()
        self.shared_dict["kill_event"].clear()
        self.shared_dict["session_start_event"].set()
        logger.info("Session started.")

    def stop_session(self):
        self.shared_dict["kill_event"].set()
        self.shared_dict["session_start_event"].clear()
        logger.info("Session stopped.")

    def cleanup(self):
        logger.info("Cleaning up processes and shared resources...")
        self.shared_dict["end_event"].set()
        self.shared_dict["kill_event"].set()
        self.shared_dict["session_start_event"].set()
        self.taskmaster_proc.join(timeout=10)
        self.dataworker_proc.join(timeout=10)

        if self.taskmaster_proc.is_alive():
            logger.warning("Forcing termination of taskmaster process.")
            self.taskmaster_proc.terminate()
            self.taskmaster_proc.join(timeout=2)

        if self.dataworker_proc.is_alive():
            logger.warning("Forcing termination of dataworker process.")
            self.dataworker_proc.terminate()
            self.dataworker_proc.join(timeout=2)

        self.manager.shutdown()

        self.robot_data_shm.close()
        self.robot_data_shm.unlink()
        self.teleop_shm.close()
        self.teleop_shm.unlink()
        logger.info("Cleanup complete.")

    _AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "audios")
    _AUDIO_FILES = {
        "Recording started": "recording_started.mp3",
        "Already recording": "already_recording.mp3",
        "Session saved": "session_saved.mp3",
        "Session failed": "session_failed.mp3",
        "Following paused": "following_paused.mp3",
        "Following resumed": "following_resumed.mp3",
        "Shutting down": "shutting_down.mp3",
    }

    def _speak(self, text):
        import subprocess
        audio_file = self._AUDIO_FILES.get(text)
        if audio_file:
            path = os.path.join(self._AUDIO_DIR, audio_file)
            subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    def _execute_command(self, cmd, last_cmd):
        if cmd == "s":
            if last_cmd[0] != "s":
                self.start_session()
                last_cmd[0] = "s"
                self._speak("Recording started")
                logger.info(f"Session started: {self.shared_dict['dirname']}")
            else:
                self._speak("Already recording")
        elif cmd == "q":
            self.stop_session()
            last_cmd[0] = "q"
            self._speak("Session saved")
        elif cmd == "d":
            self.shared_dict["failure_event"].set()
            self.stop_session()
            last_cmd[0] = "d"
            self._speak("Session failed")
        elif cmd == "p":
            if self.shared_dict["following_paused_event"].is_set():
                self.shared_dict["following_paused_event"].clear()
                self._speak("Following resumed")
            else:
                self.shared_dict["following_paused_event"].set()
                self._speak("Following paused")
        elif cmd == "exit":
            self._speak("Shutting down")
            self.cleanup()
            os._exit(0)

    def run_command_loop(self):
        import threading
        last_cmd = [None]

        def keyboard_loop():
            logger.info("Keyboard: 's'=start, 'q'=stop, 'd'=fail, 'p'=toggle follow, 'exit'=quit")
            while True:
                try:
                    user_input = input("> ").lower().strip()
                    if user_input in ("s", "q", "d", "p", "exit"):
                        self._execute_command(user_input, last_cmd)
                    else:
                        logger.info("Invalid command. Use 's', 'q', 'd', 'p', or 'exit'.")
                except EOFError:
                    break

        def pedal_loop():
            try:
                from StreamDeck.DeviceManager import DeviceManager
                decks = DeviceManager().enumerate()
                if not decks:
                    logger.warning("No StreamDeck pedal found, pedal control disabled")
                    return
                deck = decks[0]
                deck.open()
                logger.info(f"Pedal connected: {deck.deck_type()} (serial: {deck.get_serial_number()})")

                def key_change_callback(deck, key, state):
                    if not state:  # only on press, not release
                        return
                    if key == 0:
                        self._execute_command("s", last_cmd)
                    elif key == 2:
                        self._execute_command("q", last_cmd)

                deck.set_key_callback(key_change_callback)
                threading.Event().wait()
            except Exception as e:
                logger.warning(f"Pedal error: {e}")

        kb_thread = threading.Thread(target=keyboard_loop, daemon=True)
        pedal_thread = threading.Thread(target=pedal_loop, daemon=True)
        kb_thread.start()
        pedal_thread.start()

        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            try:
                self.cleanup()
            except Exception:
                pass
            os._exit(0)

