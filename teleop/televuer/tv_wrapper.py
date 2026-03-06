"""
TeleVuerWrapper - Post-processing wrapper for TeleVuer data.

Handles coordinate transformations from OpenXR Convention to Robot Convention
for Meta Quest 3, Apple Vision Pro, and PICO devices.

Based on Unitree's xr_teleoperate implementation.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .televuer import TeleVuer

# OpenXR Convention: y up, z back, x right.
# Robot Convention: z up, y left, x front.


def safe_mat_update(prev_mat, mat):
    """Return previous matrix if the new matrix is singular (determinant ~ 0)."""
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return prev_mat, False
    return mat, True


def fast_mat_inv(mat):
    """Fast inverse for SE(3) matrices."""
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def safe_rot_update(prev_rot_array, rot_array):
    """Return previous rotation if any new rotation is singular."""
    dets = np.linalg.det(rot_array)
    if not np.all(np.isfinite(dets)) or np.any(np.isclose(dets, 0.0, atol=1e-6)):
        return prev_rot_array, False
    return rot_array, True


# Transformation matrices for coordinate conversion
T_TO_UNITREE_HUMANOID_LEFT_ARM = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

T_TO_UNITREE_HUMANOID_RIGHT_ARM = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
])

T_TO_UNITREE_HAND = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
])

T_ROBOT_OPENXR = np.array([
    [0, 0, -1, 0],
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

T_OPENXR_ROBOT = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
])

R_ROBOT_OPENXR = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

R_OPENXR_ROBOT = np.array([
    [0, -1, 0],
    [0, 0, 1],
    [-1, 0, 0],
])

# Default poses
CONST_HEAD_POSE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 1.5],
    [0, 0, 1, -0.2],
    [0, 0, 0, 1],
])

CONST_RIGHT_ARM_POSE = np.array([
    [1, 0, 0, 0.15],
    [0, 1, 0, 1.13],
    [0, 0, 1, -0.3],
    [0, 0, 0, 1],
])

CONST_LEFT_ARM_POSE = np.array([
    [1, 0, 0, -0.15],
    [0, 1, 0, 1.13],
    [0, 0, 1, -0.3],
    [0, 0, 0, 1],
])

CONST_HAND_ROT = np.tile(np.eye(3)[None, :, :], (25, 1, 1))


@dataclass
class TeleData:
    """Data class containing processed teleoperation data from XR device."""

    head_pose: np.ndarray  # (4,4) SE(3) pose of head matrix
    left_wrist_pose: np.ndarray  # (4,4) SE(3) pose of left wrist of arm
    right_wrist_pose: np.ndarray  # (4,4) SE(3) pose of right wrist of arm

    # Hand tracking data (25 joints per hand)
    # https://docs.vuer.ai/en/latest/examples/19_hand_tracking.html
    # https://immersive-web.github.io/webxr-hand-input/
    left_hand_pos: np.ndarray = None  # (25,3) 3D positions of left hand joints
    right_hand_pos: np.ndarray = None  # (25,3) 3D positions of right hand joints
    left_hand_rot: np.ndarray = None  # (25,3,3) rotation matrices of left hand joints
    right_hand_rot: np.ndarray = None  # (25,3,3) rotation matrices of right hand joints

    # Hand gesture state
    left_hand_pinch: bool = False  # True if index and thumb are pinching
    left_hand_pinchValue: float = 10.0  # (~15.0 -> 0.0) pinch distance
    left_hand_squeeze: bool = False  # True if hand is making a fist
    left_hand_squeezeValue: float = 0.0  # (0.0 -> 1.0) degree of hand squeeze

    right_hand_pinch: bool = False
    right_hand_pinchValue: float = 10.0
    right_hand_squeeze: bool = False
    right_hand_squeezeValue: float = 0.0

    # Controller tracking data (Meta Quest 3 / PICO controllers)
    # https://docs.vuer.ai/en/latest/examples/20_motion_controllers.html
    left_ctrl_trigger: bool = False  # True if trigger is actively pressed
    left_ctrl_triggerValue: float = 10.0  # (10.0 -> 0.0) trigger pull depth
    left_ctrl_squeeze: bool = False  # True if grip button is pressed
    left_ctrl_squeezeValue: float = 0.0  # (0.0 -> 1.0) grip pull depth
    left_ctrl_aButton: bool = False  # True if A(X) button is pressed
    left_ctrl_bButton: bool = False  # True if B(Y) button is pressed
    left_ctrl_thumbstick: bool = False  # True if thumbstick button is pressed
    left_ctrl_thumbstickValue: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )  # 2D vector (x, y)

    right_ctrl_trigger: bool = False
    right_ctrl_triggerValue: float = 10.0
    right_ctrl_squeeze: bool = False
    right_ctrl_squeezeValue: float = 0.0
    right_ctrl_aButton: bool = False
    right_ctrl_bButton: bool = False
    right_ctrl_thumbstick: bool = False
    right_ctrl_thumbstickValue: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )


class TeleVuerWrapper:
    """
    Wrapper for TeleVuer that handles coordinate transformations.

    Transforms XR device data from OpenXR Convention to Robot Convention
    for seamless integration with Unitree humanoid robots.
    """

    def __init__(
        self,
        use_hand_tracking: bool,
        binocular: bool = True,
        img_shape: tuple = (480, 1280),
        display_fps: float = 30.0,
        display_mode: Literal["immersive", "pass-through", "ego"] = "immersive",
        zmq: bool = False,
        webrtc: bool = False,
        webrtc_url: str = None,
        cert_file: str = None,
        key_file: str = None,
        return_hand_rot_data: bool = False,
    ):
        """
        Initialize TeleVuerWrapper.

        :param use_hand_tracking: Use hand tracking (True) or controller tracking (False)
        :param binocular: Stereoscopic (True) or monocular (False) display
        :param img_shape: Shape of head camera image (height, width)
        :param display_fps: Target frames per second for display
        :param display_mode: VR viewing mode ("immersive", "pass-through", "ego")
        :param zmq: Use ZMQ for image transmission
        :param webrtc: Use WebRTC for real-time communication
        :param webrtc_url: URL for WebRTC offer (required if webrtc=True)
        :param cert_file: Path to SSL certificate file
        :param key_file: Path to SSL key file
        :param return_hand_rot_data: Include hand rotation data in output
        """
        self.use_hand_tracking = use_hand_tracking
        self.return_hand_rot_data = return_hand_rot_data
        self.tvuer = TeleVuer(
            use_hand_tracking=use_hand_tracking,
            binocular=binocular,
            img_shape=img_shape,
            display_fps=display_fps,
            display_mode=display_mode,
            zmq=zmq,
            webrtc=webrtc,
            webrtc_url=webrtc_url,
            cert_file=cert_file,
            key_file=key_file,
        )

    def get_tele_data(self) -> TeleData:
        """
        Get processed motion state data from the TeleVuer instance.

        All returned data are transformed from OpenXR Convention to Robot Convention.

        Returns:
            TeleData: Processed teleoperation data with coordinate transformations applied
        """
        # Get head pose from XR device
        Bxr_world_head, head_pose_is_valid = safe_mat_update(
            CONST_HEAD_POSE, self.tvuer.head_pose
        )

        if self.use_hand_tracking:
            # Hand tracking mode (Quest 3 hand tracking / Vision Pro)
            left_IPxr_Bxr_world_arm, left_arm_is_valid = safe_mat_update(
                CONST_LEFT_ARM_POSE, self.tvuer.left_arm_pose
            )
            right_IPxr_Bxr_world_arm, right_arm_is_valid = safe_mat_update(
                CONST_RIGHT_ARM_POSE, self.tvuer.right_arm_pose
            )

            # Change basis: OpenXR -> Robot Convention
            Brobot_world_head = T_ROBOT_OPENXR @ Bxr_world_head @ T_OPENXR_ROBOT
            left_IPxr_Brobot_world_arm = (
                T_ROBOT_OPENXR @ left_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT
            )
            right_IPxr_Brobot_world_arm = (
                T_ROBOT_OPENXR @ right_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT
            )

            # Change initial pose: OpenXR Arm -> Unitree Humanoid Arm URDF
            left_IPunitree_Brobot_world_arm = left_IPxr_Brobot_world_arm @ (
                T_TO_UNITREE_HUMANOID_LEFT_ARM if left_arm_is_valid else np.eye(4)
            )
            right_IPunitree_Brobot_world_arm = right_IPxr_Brobot_world_arm @ (
                T_TO_UNITREE_HUMANOID_RIGHT_ARM if right_arm_is_valid else np.eye(4)
            )

            # Transfer from WORLD to HEAD coordinate (translation adjustment)
            left_IPunitree_Brobot_head_arm = left_IPunitree_Brobot_world_arm.copy()
            right_IPunitree_Brobot_head_arm = right_IPunitree_Brobot_world_arm.copy()
            left_IPunitree_Brobot_head_arm[0:3, 3] = (
                left_IPunitree_Brobot_head_arm[0:3, 3] - Brobot_world_head[0:3, 3]
            )
            right_IPunitree_Brobot_head_arm[0:3, 3] = (
                right_IPunitree_Brobot_world_arm[0:3, 3] - Brobot_world_head[0:3, 3]
            )

            # Coordinate origin offset: HEAD to WAIST
            left_IPunitree_Brobot_wrist_arm = left_IPunitree_Brobot_head_arm.copy()
            right_IPunitree_Brobot_wrist_arm = right_IPunitree_Brobot_head_arm.copy()
            left_IPunitree_Brobot_wrist_arm[0, 3] += 0.15  # x
            right_IPunitree_Brobot_wrist_arm[0, 3] += 0.15
            left_IPunitree_Brobot_wrist_arm[2, 3] += 0.45  # z
            right_IPunitree_Brobot_wrist_arm[2, 3] += 0.45

            # Process hand positions
            if left_arm_is_valid and right_arm_is_valid:
                # Homogeneous coordinates
                left_IPxr_Bxr_world_hand_pos = np.concatenate(
                    [
                        self.tvuer.left_hand_positions.T,
                        np.ones((1, self.tvuer.left_hand_positions.shape[0])),
                    ]
                )
                right_IPxr_Bxr_world_hand_pos = np.concatenate(
                    [
                        self.tvuer.right_hand_positions.T,
                        np.ones((1, self.tvuer.right_hand_positions.shape[0])),
                    ]
                )

                # Change basis: OpenXR -> Robot
                left_IPxr_Brobot_world_hand_pos = (
                    T_ROBOT_OPENXR @ left_IPxr_Bxr_world_hand_pos
                )
                right_IPxr_Brobot_world_hand_pos = (
                    T_ROBOT_OPENXR @ right_IPxr_Bxr_world_hand_pos
                )

                # Transfer from WORLD to ARM frame
                left_IPxr_Brobot_arm_hand_pos = (
                    fast_mat_inv(left_IPxr_Brobot_world_arm)
                    @ left_IPxr_Brobot_world_hand_pos
                )
                right_IPxr_Brobot_arm_hand_pos = (
                    fast_mat_inv(right_IPxr_Brobot_world_arm)
                    @ right_IPxr_Brobot_world_hand_pos
                )

                # Change initial pose: XR Hand -> Unitree Hand URDF
                left_IPunitree_Brobot_arm_hand_pos = (
                    T_TO_UNITREE_HAND @ left_IPxr_Brobot_arm_hand_pos
                )[0:3, :].T
                right_IPunitree_Brobot_arm_hand_pos = (
                    T_TO_UNITREE_HAND @ right_IPxr_Brobot_arm_hand_pos
                )[0:3, :].T
            else:
                left_IPunitree_Brobot_arm_hand_pos = np.zeros((25, 3))
                right_IPunitree_Brobot_arm_hand_pos = np.zeros((25, 3))

            # Process hand rotations if requested
            if self.return_hand_rot_data:
                left_Bxr_world_hand_rot, left_hand_rot_is_valid = safe_rot_update(
                    CONST_HAND_ROT, self.tvuer.left_hand_orientations
                )
                right_Bxr_world_hand_rot, right_hand_rot_is_valid = safe_rot_update(
                    CONST_HAND_ROT, self.tvuer.right_hand_orientations
                )

                if left_hand_rot_is_valid and right_hand_rot_is_valid:
                    left_Bxr_arm_hand_rot = np.einsum(
                        "ij,njk->nik",
                        left_IPxr_Bxr_world_arm[:3, :3].T,
                        left_Bxr_world_hand_rot,
                    )
                    right_Bxr_arm_hand_rot = np.einsum(
                        "ij,njk->nik",
                        right_IPxr_Bxr_world_arm[:3, :3].T,
                        right_Bxr_world_hand_rot,
                    )
                    left_Brobot_arm_hand_rot = np.einsum(
                        "ij,njk,kl->nil",
                        R_ROBOT_OPENXR,
                        left_Bxr_arm_hand_rot,
                        R_OPENXR_ROBOT,
                    )
                    right_Brobot_arm_hand_rot = np.einsum(
                        "ij,njk,kl->nil",
                        R_ROBOT_OPENXR,
                        right_Bxr_arm_hand_rot,
                        R_OPENXR_ROBOT,
                    )
                else:
                    left_Brobot_arm_hand_rot = left_Bxr_world_hand_rot
                    right_Brobot_arm_hand_rot = right_Bxr_world_hand_rot
            else:
                left_Brobot_arm_hand_rot = None
                right_Brobot_arm_hand_rot = None

            return TeleData(
                head_pose=Brobot_world_head,
                left_wrist_pose=left_IPunitree_Brobot_wrist_arm,
                right_wrist_pose=right_IPunitree_Brobot_wrist_arm,
                left_hand_pos=left_IPunitree_Brobot_arm_hand_pos,
                right_hand_pos=right_IPunitree_Brobot_arm_hand_pos,
                left_hand_rot=left_Brobot_arm_hand_rot,
                right_hand_rot=right_Brobot_arm_hand_rot,
                left_hand_pinch=self.tvuer.left_hand_pinch,
                left_hand_pinchValue=self.tvuer.left_hand_pinchValue * 100.0,
                left_hand_squeeze=self.tvuer.left_hand_squeeze,
                left_hand_squeezeValue=self.tvuer.left_hand_squeezeValue,
                right_hand_pinch=self.tvuer.right_hand_pinch,
                right_hand_pinchValue=self.tvuer.right_hand_pinchValue * 100.0,
                right_hand_squeeze=self.tvuer.right_hand_squeeze,
                right_hand_squeezeValue=self.tvuer.right_hand_squeezeValue,
            )

        else:
            # Controller tracking mode (Quest 3 Touch Controllers)
            left_IPunitree_Bxr_world_arm, left_arm_is_valid = safe_mat_update(
                CONST_LEFT_ARM_POSE, self.tvuer.left_arm_pose
            )
            right_IPunitree_Bxr_world_arm, right_arm_is_valid = safe_mat_update(
                CONST_RIGHT_ARM_POSE, self.tvuer.right_arm_pose
            )

            # Change basis
            Brobot_world_head = T_ROBOT_OPENXR @ Bxr_world_head @ T_OPENXR_ROBOT
            left_IPunitree_Brobot_world_arm = (
                T_ROBOT_OPENXR @ left_IPunitree_Bxr_world_arm @ T_OPENXR_ROBOT
            )
            right_IPunitree_Brobot_world_arm = (
                T_ROBOT_OPENXR @ right_IPunitree_Bxr_world_arm @ T_OPENXR_ROBOT
            )

            # Transfer from WORLD to HEAD coordinate
            left_IPunitree_Brobot_head_arm = left_IPunitree_Brobot_world_arm.copy()
            right_IPunitree_Brobot_head_arm = right_IPunitree_Brobot_world_arm.copy()
            left_IPunitree_Brobot_head_arm[0:3, 3] = (
                left_IPunitree_Brobot_head_arm[0:3, 3] - Brobot_world_head[0:3, 3]
            )
            right_IPunitree_Brobot_head_arm[0:3, 3] = (
                right_IPunitree_Brobot_head_arm[0:3, 3] - Brobot_world_head[0:3, 3]
            )

            # Coordinate origin offset: HEAD to WAIST
            left_IPunitree_Brobot_wrist_arm = left_IPunitree_Brobot_head_arm.copy()
            right_IPunitree_Brobot_wrist_arm = right_IPunitree_Brobot_head_arm.copy()
            left_IPunitree_Brobot_wrist_arm[0, 3] += 0.15  # x
            right_IPunitree_Brobot_wrist_arm[0, 3] += 0.15
            left_IPunitree_Brobot_wrist_arm[2, 3] += 0.45  # z
            right_IPunitree_Brobot_wrist_arm[2, 3] += 0.45

            return TeleData(
                head_pose=Brobot_world_head,
                left_wrist_pose=left_IPunitree_Brobot_wrist_arm,
                right_wrist_pose=right_IPunitree_Brobot_wrist_arm,
                left_ctrl_trigger=self.tvuer.left_ctrl_trigger,
                left_ctrl_triggerValue=10.0 - self.tvuer.left_ctrl_triggerValue * 10,
                left_ctrl_squeeze=self.tvuer.left_ctrl_squeeze,
                left_ctrl_squeezeValue=self.tvuer.left_ctrl_squeezeValue,
                left_ctrl_aButton=self.tvuer.left_ctrl_aButton,
                left_ctrl_bButton=self.tvuer.left_ctrl_bButton,
                left_ctrl_thumbstick=self.tvuer.left_ctrl_thumbstick,
                left_ctrl_thumbstickValue=self.tvuer.left_ctrl_thumbstickValue,
                right_ctrl_trigger=self.tvuer.right_ctrl_trigger,
                right_ctrl_triggerValue=10.0 - self.tvuer.right_ctrl_triggerValue * 10,
                right_ctrl_squeeze=self.tvuer.right_ctrl_squeeze,
                right_ctrl_squeezeValue=self.tvuer.right_ctrl_squeezeValue,
                right_ctrl_aButton=self.tvuer.right_ctrl_aButton,
                right_ctrl_bButton=self.tvuer.right_ctrl_bButton,
                right_ctrl_thumbstick=self.tvuer.right_ctrl_thumbstick,
                right_ctrl_thumbstickValue=self.tvuer.right_ctrl_thumbstickValue,
            )

    def render_to_xr(self, img):
        """Render an image to the XR headset display."""
        self.tvuer.render_to_xr(img)

    def close(self):
        """Close the TeleVuer connection."""
        self.tvuer.close()
