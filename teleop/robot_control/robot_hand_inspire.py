# this file is legacy, need to fix.
import threading
import time
from enum import IntEnum
from multiprocessing import Array, Process

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # dds
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

inspire_tip_indices = [4, 9, 14, 19, 24]
Inspire_Num_Motors = 6
kTopicInspireCommand = "rt/inspire/cmd"
kTopicInspireState = "rt/inspire/state"

# FTP Hand topics (separate left/right)
kTopicInspireFTPCommandLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPCommandRight = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPStateLeft = "rt/inspire_hand/state/l"
kTopicInspireFTPStateRight = "rt/inspire_hand/state/r"

class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False):
        print("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.HandCmb_publisher = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
        self.HandCmb_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(kTopicInspireState, MotorStates_)
        self.HandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.right_hand_state_array): # any(self.left_hand_state_array) and
                break
            time.sleep(0.01)
            print("[Inspire_Controller] Waiting to subscribe dds...")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            hand_msg  = self.HandState_subscriber.Read()
            if hand_msg is not None:
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = right_q_target[idx]

        self.HandCmb_publisher.Write(self.hand_msg)
        # print("hand ctrl publish ok.")

    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        self.running = True

        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # initialize inspire hand's cmd msg
        self.hand_msg  = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))]

        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                left_hand_mat  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_mat[inspire_tip_indices]
                    ref_right_value = right_hand_mat[inspire_tip_indices]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # In website https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand, you can find
                    #     In the official document, the angles are in the range [0, 1] ==> 0.0: fully closed  1.0: fully open
                    # The q_target now is in radians, ranges:
                    #     - idx 0~3: 0~1.7 (1.7 = closed)
                    #     - idx 4:   0~0.5
                    #     - idx 5:  -0.1~1.3
                    # We normalize them using (max - value) / range
                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            print("Dex3_1_Controller has been closed.")

# Update hand state, according to the official documentation, https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11


class Inspire_Controller_FTP:
    """
    Controller for Inspire FTP dexterous hands on G1 robot.

    Key differences from Inspire_Controller (DFX):
    - Uses separate left/right DDS topics: rt/inspire_hand/ctrl/l, rt/inspire_hand/ctrl/r
    - Uses inspire_sdkpy instead of unitree_sdk2py for hand messages
    - Commands scaled to 0-1000 integer range (instead of 0-1 float)
    - Same 6 DOF per hand (12 total)
    """

    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
    ):
        print("Initialize Inspire_Controller_FTP...")
        self.fps = fps
        self.Unit_Test = Unit_Test

        # Import FTP-specific SDK
        import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default
        from inspire_sdkpy import inspire_dds

        self.inspire_dds = inspire_dds
        self.inspire_hand_default = inspire_hand_default

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)
            ChannelFactoryInitialize(0)

        # Initialize separate left/right DDS publishers
        self.left_hand_publisher = ChannelPublisher(
            kTopicInspireFTPCommandLeft,
            inspire_dds.InspireHandMsg_
        )
        self.left_hand_publisher.Init()

        self.right_hand_publisher = ChannelPublisher(
            kTopicInspireFTPCommandRight,
            inspire_dds.InspireHandMsg_
        )
        self.right_hand_publisher.Init()

        # Initialize separate left/right DDS subscribers
        self.left_hand_subscriber = ChannelSubscriber(
            kTopicInspireFTPStateLeft,
            inspire_dds.InspireHandMsg_
        )
        self.left_hand_subscriber.Init()

        self.right_hand_subscriber = ChannelSubscriber(
            kTopicInspireFTPStateRight,
            inspire_dds.InspireHandMsg_
        )
        self.right_hand_subscriber.Init()

        # Shared Arrays for hand states (6 per hand)
        self.left_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)

        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(
            target=self._subscribe_hand_state
        )
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Wait for initial hand state
        while True:
            if any(self.right_hand_state_array) or any(self.left_hand_state_array):
                break
            time.sleep(0.01)
            print("[Inspire_Controller_FTP] Waiting to subscribe dds...")

        # Start control process
        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Initialize Inspire_Controller_FTP OK!\n")

    def _subscribe_hand_state(self):
        """Subscribe to left and right hand state topics."""
        while True:
            # Read left hand state
            left_msg = self.left_hand_subscriber.Read()
            if left_msg is not None:
                # FTP state angle_act is scaled by 1/1000
                for idx in range(Inspire_Num_Motors):
                    self.left_hand_state_array[idx] = left_msg.angle_act[idx] / 1000.0

            # Read right hand state
            right_msg = self.right_hand_subscriber.Read()
            if right_msg is not None:
                for idx in range(Inspire_Num_Motors):
                    self.right_hand_state_array[idx] = right_msg.angle_act[idx] / 1000.0

            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q.
        FTP uses 0-1000 integer range for commands.
        """
        # Create left hand message
        left_msg = self.inspire_hand_default.get_inspire_hand_ctrl()
        left_msg.mode = 0b0001  # angle control mode
        left_msg.angle_set = [int(q * 1000) for q in left_q_target]

        # Create right hand message
        right_msg = self.inspire_hand_default.get_inspire_hand_ctrl()
        right_msg.mode = 0b0001  # angle control mode
        right_msg.angle_set = [int(q * 1000) for q in right_q_target]

        # Publish to separate topics
        self.left_hand_publisher.Write(left_msg)
        self.right_hand_publisher.Write(right_msg)

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        self.running = True

        left_q_target = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        try:
            while self.running:
                start_time = time.time()
                # Get dual hand state from shared arrays
                left_hand_mat = np.array(left_hand_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate(
                    (
                        np.array(left_hand_state_array[:]),
                        np.array(right_hand_state_array[:]),
                    )
                )

                # Check if hand data has been initialized
                if not np.all(right_hand_mat == 0.0) and not np.all(
                    left_hand_mat[4] == np.array([-1.13, 0.3, 0.15])
                ):
                    ref_left_value = left_hand_mat[inspire_tip_indices]
                    ref_right_value = right_hand_mat[inspire_tip_indices]

                    left_q_target = self.hand_retargeting.left_retargeting.retarget(
                        ref_left_value
                    )[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(
                        ref_right_value
                    )[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # Normalize angles from radians to 0-1 range
                    # Same normalization as DFX controller
                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx] = normalize(
                                left_q_target[idx], 0.0, 1.7
                            )
                            right_q_target[idx] = normalize(
                                right_q_target[idx], 0.0, 1.7
                            )
                        elif idx == 4:
                            left_q_target[idx] = normalize(
                                left_q_target[idx], 0.0, 0.5
                            )
                            right_q_target[idx] = normalize(
                                right_q_target[idx], 0.0, 0.5
                            )
                        elif idx == 5:
                            left_q_target[idx] = normalize(
                                left_q_target[idx], -0.1, 1.3
                            )
                            right_q_target[idx] = normalize(
                                right_q_target[idx], -0.1, 1.3
                            )

                # Get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            print("Inspire_Controller_FTP has been closed.")

    def get_current_dual_hand_q(self):
        """Get current hand joint positions (normalized 0-1)."""
        return np.concatenate(
            (
                np.array(self.left_hand_state_array[:]),
                np.array(self.right_hand_state_array[:]),
            )
        )

    def shutdown(self):
        """Shutdown the controller."""
        self.running = False
        print("Inspire_Controller_FTP shutdown.")

    def reset(self):
        """Reset the controller state."""
        pass
