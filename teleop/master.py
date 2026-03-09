import os
import sys
import time
from multiprocessing import Array, Event, Lock, Manager, Process, Queue, shared_memory

import numpy as np

from lidar import LidarProcess
from merger import DataMerger
from robot_control.robot_arm import G1_29_ArmController, H1_2_ArmController
from robot_control.robot_arm_ik import G1_29_ArmIK, H1_2_ArmIK
from robot_control.robot_hand_inspire import Inspire_Controller
from robot_control.g1_robot_hand_inspire import InspireController as G1_Inspire_Controller
from robot_control.robot_hand_unitree import Dex3_1_Controller
from utils.logger import logger
from writers import IKDataWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from constants import *


class RobotTaskmaster:
    def __init__(
        self, task_name, shared_data, robot_shm_array, teleop_shm_array, robot="h1"
    ):
        self.task_name = task_name
        self.robot = robot
        self.boundaries = (0,0)

        self.shared_data = shared_data
        self.kill_event = shared_data["kill_event"]
        self.session_start_event = shared_data["session_start_event"]
        self.failure_event = shared_data["failure_event"]  # TODO: redundent
        self.end_event = shared_data["end_event"]  # TODO: redundent
        self.following_paused_event = shared_data["following_paused_event"]

        self.robot_shm_array = robot_shm_array
        self.teleop_shm_array = teleop_shm_array

        self.teleop_lock = Lock()
        try:
            if robot == "g1":
                logger.info("Using g1 controllers")
                self.arm_ctrl = G1_29_ArmController()
                self.arm_ik = G1_29_ArmIK(Visualization=False)
                self.dual_hand_data_lock = Lock()
                dual_hand_state_array = Array(
                    "d", 14, lock=False
                )  # [output] current left, right hand state(14) data.
                dual_hand_action_array = Array(
                    "d", 14, lock=False
                )  # [output] current left, right hand action(14) data.
                self.hand_shm = shared_memory.SharedMemory(
                    create=True, size=14 * np.dtype(np.float64).itemsize
                )
                self.hand_shm_array = np.ndarray(
                    (14,), dtype=np.float64, buffer=self.hand_shm.buf
                )

                self.hand_ctrl = Dex3_1_Controller(
                    self.hand_shm_array,
                    self.dual_hand_data_lock,
                    dual_hand_state_array,
                    dual_hand_action_array,
                )
            elif robot == "h1":
                logger.info("Using h1 controllers")
                self.arm_ctrl = H1_2_ArmController()
                self.arm_ik = H1_2_ArmIK(Visualization=False)
                self.dual_hand_data_lock = Lock()
                dual_hand_state_array = Array(
                    "d", 12, lock=False
                )  # [output] current left, right hand state(14) data.
                dual_hand_action_array = Array(
                    "d", 12, lock=False
                )  # [output] current left, right hand action(14) data.
                self.hand_shm_left = shared_memory.SharedMemory(
                    create=True, size=6 * np.dtype(np.float64).itemsize
                )
                self.lefthand_shm_array = np.ndarray(
                    (6,), dtype=np.float64, buffer=self.hand_shm_left.buf
                )
                self.hand_shm_right = shared_memory.SharedMemory(
                    create=True, size=6 * np.dtype(np.float64).itemsize
                )
                self.righthand_shm_array = np.ndarray(
                    (6,), dtype=np.float64, buffer=self.hand_shm_right.buf
                )
                self.hand_ctrl = G1_Inspire_Controller(
                    self.lefthand_shm_array,
                    self.righthand_shm_array,
                    self.dual_hand_data_lock,
                    dual_hand_state_array,
                    dual_hand_action_array,
                )
                # self.arm_ctrl = H1ArmController()
                # self.arm_ik = Arm_IK()
                # self.hand_ctrl = H1HandController()

            elif robot == "g1_inspire":
                logger.info("Using g1 controllers")
                self.arm_ctrl = G1_29_ArmController()
                self.arm_ik = G1_29_ArmIK(Visualization=False)
                self.dual_hand_data_lock = Lock()
                # dual_hand_state_array = Array(
                #     "d", 14, lock=False
                # )  # [output] current left, right hand state(14) data.
                # dual_hand_action_array = Array(
                #     "d", 14, lock=False
                # )  # [output] current left, right hand action(14) data.
                # self.hand_shm = shared_memory.SharedMemory(
                #     create=True, size=14 * np.dtype(np.float64).itemsize
                # )
                # self.hand_shm_array = np.ndarray(
                #     (14,), dtype=np.float64, buffer=self.hand_shm.buf
                # )


                # dual_hand_state_array = Array(
                #     "d", 12, lock=False
                # )  # [output] current left, right hand state(14) data.
                # dual_hand_action_array = Array(
                #     "d", 12, lock=False
                # )  # [output] current left, right hand action(14) data.
                # self.hand_shm_left = shared_memory.SharedMemory(
                #     create=True, size=6 * np.dtype(np.float64).itemsize
                # )
                # self.lefthand_shm_array = np.ndarray(
                #     (6,), dtype=np.float64, buffer=self.hand_shm_left.buf
                # )
                # self.hand_shm_right = shared_memory.SharedMemory(
                #     create=True, size=6 * np.dtype(np.float64).itemsize
                # )
                # self.righthand_shm_array = np.ndarray(
                #     (6,), dtype=np.float64, buffer=self.hand_shm_right.buf
                # )

                # self.hand_ctrl = G1_Inspire_Controller(
                #     self.lefthand_shm_array,
                #     self.righthand_shm_array,
                #     self.dual_hand_data_lock,
                #     dual_hand_state_array,
                #     dual_hand_action_array,
                # ) 
                self.hand_ctrl = G1_Inspire_Controller()
                
            else:
                logger.error("unknown robot")
                exit(-1)
        except Exception as e:
            logger.error(f"Master: failed initalizing controllers/ik_solvers: {e}")
            logger.error(f"Master: exiting")
            exit(-1)

        self.first = True
        self.lidar_proc = None
        self.merge_proc = None
        self.ik_writer = None
        self.running = False
        # self.h1_lock = Lock()

    def safelySetMotor(self, sol_q, last_sol_q, tau_ff):
        q_poseList = sol_q
        q_tau_ff = tau_ff
        dynamic_thresholds = np.array(
            [np.pi / 3] * 5  # left shoulder and elbow
            + [np.pi] * 2  # left wrists
            + [np.pi / 3] * 5
            + [np.pi] * 2
        )
        if last_sol_q is None:
            self.arm_ctrl.ctrl_dual_arm(q_poseList, q_tau_ff, True)
            return True

        if last_sol_q is not None and np.any(
            np.abs(last_sol_q - sol_q) > dynamic_thresholds
        ):
            logger.error("Master: ik movement too large!")
            return False

        logger.debug("Master: preparing to set motor")
        self.arm_ctrl.ctrl_dual_arm(q_poseList, q_tau_ff)
        logger.debug("Master: motor set")
        return True

    # run_idx=0
    def setHandMotors(self, right_qpos, left_qpos):
        # print(len(right_qpos))
        if right_qpos is not None and left_qpos is not None:
            # Finger proximal joints: URDF [0, 1.7], hardware [0, 1.7] — direct
            # right_hand_angles = [1.7-right_qpos[i] for i in range(6)]
            # Use _0 (proximal) joints, range [0, 1.571]; normalize to [0,1]: (1.571-qpos)/1.571
            right_hand_angles = [right_qpos[i] for i in [3,3,3,5,0,2]]
            right_hand_angles[0]=(1.571-right_hand_angles[0])/1.571
            right_hand_angles[1]=(1.571-right_hand_angles[1])/1.571
            right_hand_angles[2]=(1.571-right_hand_angles[2])/1.571
            right_hand_angles[3]=(1.571-right_hand_angles[3])/1.571
            right_hand_angles[4]=-(right_hand_angles[4]+0.5)/0.5
            right_hand_angles[5]=-right_hand_angles[5]
            # Thumb pitch: URDF [0, 0.5] → hardware [0, 1.2]
            # right_hand_angles.append(1.2-right_qpos[9]/0.3 * 1.2)
            # Thumb yaw: URDF [-0.1, 1.3] → hardware [0, 0.5]
            # right_hand_angles.append(0.5-(right_qpos[8] + 0.1) / 1.4 * 0.5)
            # print(right_qpos[8], right_qpos[9])
            # self.boundaries = (min(self.boundaries[0], right_qpos[4]), max(self.boundaries[1], right_qpos[4]))
            # print(self.boundaries)

            # print(left_qpos)
            # Left DEX3 finger joints have opposite sign vs right: range [-1.745, 0] (0=open, -=closed)
            # Use +1.7 instead of 1.7- to map [-1.745,0] → [0,1.7]
            # Left _0 joints range [-1.571, 0]; normalize to [0,1]: (1.571+qpos)/1.571
            left_hand_angles = [left_qpos[i] for i in [3,3,3,5,0,2]]
            left_hand_angles[0]=(1.571+left_hand_angles[0])/1.571
            left_hand_angles[1]=(1.571+left_hand_angles[1])/1.571
            left_hand_angles[2]=(1.571+left_hand_angles[2])/1.571
            left_hand_angles[3]=(1.571+left_hand_angles[3])/1.571
            left_hand_angles[4]=-(left_hand_angles[4]+0.5)/0.5
            # Left thumb_2 range is [0, 1.745] (opposite of right's [-1.745,0]), no negation needed
            left_hand_angles[5]=left_hand_angles[5]
            # left_hand_angles.append(1.2-left_qpos[9]/0.3 * 1.2)
            # left_hand_angles.append(0.5-(left_qpos[8] + 0.1) / 1.4 * 0.5)
            self.hand_ctrl.ctrl(left_hand_angles, right_hand_angles)

            # self.run_idx+=1
            # if self.run_idx%20==0:
            #     print(f"thumb_pitch qpos[9]: {right_qpos[9]:.3f}thumb_pitch qpos[8]: {right_qpos[8]:.3f}")


            # if self.hand_ctrl is not None:
            #     self.hand_ctrl.ctrl_dual_hand(right_hand_angles, left_hand_angles)
            # else:   
            #     print("TODO")

            # self.left_hand_array[:] = left_qpos
            # self.right_hand_array[:] = right_qpos
            # self.hand_ctrl.ctrl_dual_hand(right_qpos, left_qpos)
        return left_qpos, right_qpos

    def start(self):
        # logger.debug(f"Master: Process ID (PID) {os.getpid()}")
        try:
            while not self.end_event.is_set():
                logger.info("Master: waiting to start")
                self.session_start_event.wait()
                if self.end_event.is_set():
                    break
                logger.info(
                    "Master: start event recvd. clearing start event. starting session"
                )
                self.run_session()
                logger.debug("Master: merging data...")
                if not self.failure_event.is_set():
                    self.merge_data()  # TODO: maybe a separate thread?
                    logger.info("Master: merge finished. Preparing for a new run...")
                else:
                    # self.delete_last_data()
                    logger.info(
                        "Master: not merging. Preparing for a new run to override..."
                    )
                self.reset()
                logger.info("Master: reset finished")
        finally:
            self.arm_ctrl.gradually_set_weight_to_0()
            self.stop()

            if self.robot == "g1":
                self.hand_shm.close()
                self.hand_shm.unlink()
            logger.info("Master: exited")

    def get_robot_data(self):
        motorstate = self.arm_ctrl.get_current_motor_q()
        logger.debug(f"motorstate f{motorstate}")
        
        """handstate = self.hand_ctrl.get_current_dual_hand_q()
        """
        if self.robot == "g1":
            hand_press_state = self.hand_ctrl.get_current_dual_hand_pressure()
            robot_sizes = G1_sizes
        elif self.robot == "g1_inspire":
            """hand_press_state = self.hand_ctrl.get_current_dual_hand_pressure()"""
            robot_sizes = G1_inspire_sizes
        elif self.robot == "h1":
            robot_sizes = H1_sizes

        imustate = self.arm_ctrl.get_imu_data()

        odomstate = self.arm_ctrl.get_odom_data()

        # var_imu = dir(imustate)
        current_lr_arm_q = self.arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = self.arm_ctrl.get_current_dual_arm_dq()

        motor_state_size = robot_sizes.ARM_STATE_SIZE + robot_sizes.LEG_STATE_SIZE
        # with self.h1_lock:
        motor_start = 0
        hand_start = motor_start + motor_state_size
        quat_start = hand_start + robot_sizes.HAND_STATE_SIZE
        accel_start = quat_start + robot_sizes.IMU_QUATERNION_SIZE
        gyro_start = accel_start + robot_sizes.IMU_ACCELEROMETER_SIZE
        rpy_start = gyro_start + robot_sizes.IMU_GYROSCOPE_SIZE
        pos_start = rpy_start + robot_sizes.IMU_RPY_SIZE
        velocity_start = pos_start + robot_sizes.ODOM_POSITION_SIZE
        odom_rpy_start = velocity_start + robot_sizes.ODOM_VELOCITY_SIZE
        odom_quat_start = odom_rpy_start + robot_sizes.ODOM_RPY_SIZE
        odom_quat_end = odom_quat_start + robot_sizes.ODOM_QUATERNION_SIZE

        self.robot_shm_array[motor_start:hand_start] = motorstate[0:motor_state_size]
        # self.robot_shm_array[hand_start:quat_start] = handstate
        self.robot_shm_array[quat_start:accel_start] = imustate.quaternion
        self.robot_shm_array[accel_start:gyro_start] = imustate.accelerometer
        self.robot_shm_array[gyro_start:rpy_start] = imustate.gyroscope
        self.robot_shm_array[rpy_start:pos_start] = imustate.rpy

        # self.robot_shm_array[pos_start:velocity_start] = odomstate["position"]
        # self.robot_shm_array[velocity_start:odom_rpy_start] = odomstate["velocity"]
        # self.robot_shm_array[odom_rpy_start:odom_quat_start] = odomstate["orientation_rpy"]
        # self.robot_shm_array[odom_quat_start:odom_quat_end] = odomstate["orientation_quaternion"]

        if self.robot == "g1":
            # press_start = rpy_start + robot_sizes.IMU_RPY_SIZE
            # self.robot_shm_array[rpy_start:press_start] = imustate.rpy
            self.robot_shm_array[pos_start:velocity_start] = odomstate["position"]
            self.robot_shm_array[velocity_start:odom_rpy_start] = odomstate["velocity"]
            self.robot_shm_array[odom_rpy_start:odom_quat_start] = odomstate[
                "orientation_rpy"
            ]
            self.robot_shm_array[odom_quat_start:odom_quat_end] = odomstate[
                "orientation_quaternion"
            ]

            self.robot_shm_array[
                odom_quat_end : odom_quat_end + robot_sizes.HAND_PRESS_SIZE
            ] = hand_press_state.flatten()
        # elif self.robot == "g1_inspire":
        #     # press_start = rpy_start + robot_sizes.IMU_RPY_SIZE
        #     # self.robot_shm_array[rpy_start:press_start] = imustate.rpy
        #     self.robot_shm_array[pos_start:velocity_start] = odomstate["position"]
        #     self.robot_shm_array[velocity_start:odom_rpy_start] = odomstate["velocity"]
        #     self.robot_shm_array[odom_rpy_start:odom_quat_start] = odomstate[
        #         "orientation_rpy"
        #     ]
        #     self.robot_shm_array[odom_quat_start:odom_quat_end] = odomstate[
        #         "orientation_quaternion"
        #     ]

            # self.robot_shm_array[
            #     odom_quat_end : odom_quat_end + robot_sizes.HAND_PRESS_SIZE
            # ] = hand_press_state.flatten()

        # elif self.robot == "h1":
        #     self.robot_shm_array[rpy_start:] = imustate.rpy

        return current_lr_arm_q, current_lr_arm_dq

    def get_teleoperator_data(self):
        with self.teleop_lock:
            teleop_data = self.teleop_shm_array.copy()
        # logger.debug(f"Master: receving data : {teleop_data}")
        if np.all(teleop_data == 0):
            logger.debug(f"Master: not receving data yet: {teleop_data}")
            return False, None, None, None, None, None
        head_rmat = teleop_data[0:9].reshape(3, 3)
        left_pose = teleop_data[9:25].reshape(4, 4)
        right_pose = teleop_data[25:41].reshape(4, 4)
        left_qpos = teleop_data[41:48]
        right_qpos = teleop_data[48:55]
        return True, head_rmat, left_pose, right_pose, left_qpos, right_qpos

    def _session_init(self):
        if "dirname" not in self.shared_data:
            logger.error("Master: failed to get dirname")
            exit(-1)
        self.lidar_proc = LidarProcess(self.shared_data["dirname"])
        self.lidar_proc.run()
        logger.debug("Master: lidar process started")
        self.running = True
        self.ik_writer = IKDataWriter(self.shared_data["dirname"])
        logger.debug("Master: getting teleop shm name")

    def run_session(self):
        self._session_init()
        last_sol_q = None
        logger.debug("Master: waiting for kill event")
        self.arm_ctrl.set_weight_to_1()
        while not self.kill_event.is_set():
            logger.debug("Master: looping")
            current_lr_arm_q, current_lr_arm_dq = self.get_robot_data()
            motor_time = (
                time.time()
            )  # TODO: might be late here/ consider puting it before getmotorstate

            get_tv_success, head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                self.get_teleoperator_data()
            )
            # logger.debug("Master: got teleop ddata")

            # self.arm_ctrl.gradually_increase_weight_to_1()
            if not get_tv_success:
                continue

            if self.following_paused_event.is_set():
                continue

            sol_q, tau_ff = self.arm_ik.solve_ik(
                left_pose, right_pose, current_lr_arm_q, current_lr_arm_dq
            )

            ik_time = time.time()

            # logger.debug(f"Master: moving motor {sol_q}")
            if self.safelySetMotor(sol_q, last_sol_q, tau_ff):
                last_sol_q = sol_q
            else:
                continue

            if self.robot == "h1":
                self.setHandMotors(right_qpos, left_qpos)
            elif self.robot == "g1":
                with self.dual_hand_data_lock:
                    self.hand_shm_array[0:7] = left_qpos
                    self.hand_shm_array[7:14] = right_qpos
            elif self.robot == "g1_inspire":
                self.setHandMotors(right_qpos, left_qpos)

            # logger.debug("Master: writing data")
            # logger.debug(f"Master: head_rmat: {head_rmat}")
            self.ik_writer.write_data(
                right_qpos,
                left_qpos,
                motor_time,
                ik_time,
                sol_q,
                tau_ff,
                head_rmat,
                left_pose,
                right_pose,
            )

    def stop(self):
        self.running = False
        if self.lidar_proc is not None:
            self.lidar_proc.cleanup()

        if self.merge_proc is not None and self.merge_proc.is_alive():
            logger.debug("Master: Waiting for merge process to complete...")
            self.merge_proc.join(timeout=5)
            if self.merge_proc.is_alive():
                logger.warning(
                    "Master: Merge process did not complete in time, terminating"
                )
                self.merge_proc.terminate()

        logger.debug("Master: shutting down h1 contorllers...")
        self.arm_ctrl.shutdown()
        # self.hand_ctrl.shutdown()
        logger.debug("Master: h1 controlleers shutdown")
        logger.info("Master: Stopping all threads ended!")

    def reset(self):
        logger.info("Master: Resetting RobotTaskmaster...")
        if self.running:
            self.stop()
        logger.info("Master: Clearing stop event...")
        # self.kill_event.clear()  # TODO: create a new one?

        # self.hand_ctrl.reset()
        self.arm_ctrl.reset()
        self.first = True
        self.running = False

        self.robot_shm_array[:] = 0

        self.ik_writer = IKDataWriter(self.shared_data["dirname"])

        logger.info("RobotTaskmaster has been reset and is ready to start again.")

    def merge_data(self):
        if self.ik_writer is not None:
            self.ik_writer.close()

        if self.merge_proc is not None and self.merge_proc.is_alive():
            logger.debug(
                "Master: Previous merge process still running, not starting a new one"
            )
            return

        def merge_process():
            merger = DataMerger(self.shared_data["dirname"])
            merger.merge_json()

        self.merge_proc = Process(target=merge_process)
        self.merge_proc.daemon = True
        self.merge_proc.start()
        logger.debug("Master: Started merge process in background")

    def delete_last_data(self):
        # TODO: auto delete
        with open(self.shared_data["dirname"] + "/failed", "w"):
            pass
