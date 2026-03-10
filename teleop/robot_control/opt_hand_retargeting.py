"""Optimization-based hand retargeting for Inspire hands.

Extracted from debug_hand_retargeting/debug_hand_retargeting.py.
Uses SLSQP optimization over 6 actuated DOF with combined
fingertip-position + pinch objective (Xin et al. 2025).
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.spatial.transform import Rotation

# ── Paths ──
_MODULE_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _MODULE_DIR.parent.parent
_INSPIRE_DIR = (
    _PROJECT_ROOT
    / "debug_hand_retargeting"
    / "dex-retargeting"
    / "assets"
    / "robots"
    / "hands"
    / "inspire_hand"
)

# ── OpenXR 25-joint hand layout ──
#   0: wrist, 1-4: thumb, 5-9: index, 10-14: middle, 15-19: ring, 20-24: pinky
H_WRIST = 0
H_TIPS = np.array([4, 9, 14, 19, 24])
H_IDX_META = 5
H_MID_META = 10
H_PNK_META = 20

# ── Hyperparameters ──
EPS1 = 0.10
EPS2 = 0.01
SIG_W = 10.0
W_SHAPE = 1.0
W_PINCH = 10.0
W_CURL = 0.5
W_VEL = 0.05
EMA_ALPHA = 0.6

# ── Inspire hand actuated joints (optimization variable order) ──
_ACTUATED = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]

_MIMICS = [
    ("thumb_intermediate_joint", 1, 1.334, 0.0),
    ("thumb_distal_joint", 1, 0.667, 0.0),
    ("index_intermediate_joint", 2, 1.06399, -0.04545),
    ("middle_intermediate_joint", 3, 1.06399, -0.04545),
    ("ring_intermediate_joint", 4, 1.06399, -0.04545),
    ("pinky_intermediate_joint", 5, 1.06399, -0.04545),
]

_TIP_BODIES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
_META_BODIES = ["index_proximal", "middle_proximal", "pinky_proximal"]

# WebXR (Y-up) -> MuJoCo (Z-up) rotation
R_Y2Z = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


# ── URDF -> MJCF conversion (kinematic chain only) ──


def _rpy_to_quat_str(rpy):
    r = Rotation.from_euler("xyz", rpy)
    q = r.as_quat()  # [x, y, z, w]
    return f"{q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f}"


def _parse_origin(elem):
    if elem is None:
        return "", ""
    xyz = elem.get("xyz", "0 0 0")
    rpy_str = elem.get("rpy", "0 0 0")
    rpy = [float(x) for x in rpy_str.split()]
    pos_str = f'pos="{xyz}"'
    quat_str = ""
    if any(abs(v) > 1e-8 for v in rpy):
        quat_str = f'quat="{_rpy_to_quat_str(rpy)}"'
    return pos_str, quat_str


def _urdf_to_mjcf_fragments(urdf_path, prefix):
    """Parse URDF, return MJCF body-hierarchy XML (no meshes needed for FK)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link in root.findall("link"):
        links[link.get("name")] = link

    children = {}
    all_children = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children.setdefault(parent, []).append((joint, child))
        all_children.add(child)

    root_link = [n for n in links if n not in all_children][0]

    def _build_body(link_name, indent):
        lines = []
        for joint, child_name in children.get(link_name, []):
            jtype = joint.get("type")
            jname = joint.get("name")
            origin = joint.find("origin")
            p, q = _parse_origin(origin)

            pj = f"{prefix}_{jname}"
            pc = f"{prefix}_{child_name}"

            body_attrs = f'name="{pc}"'
            if p:
                body_attrs += f" {p}"
            if q:
                body_attrs += f" {q}"
            lines.append(f'{" " * indent}<body {body_attrs}>')
            lines.append(f'{" " * (indent + 2)}<inertial mass="0.01" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>')

            if jtype == "revolute":
                axis_el = joint.find("axis")
                axis = axis_el.get("xyz") if axis_el is not None else "0 0 1"
                lim = joint.find("limit")
                lo = lim.get("lower", "0")
                hi = lim.get("upper", "0")
                lines.append(
                    f'{" " * (indent + 2)}<joint name="{pj}" type="hinge" axis="{axis}" range="{lo} {hi}"/>'
                )

            lines.extend(_build_body(child_name, indent + 2))
            lines.append(f'{" " * indent}</body>')
        return lines

    body_lines = []
    for joint, child_name in children.get(root_link, []):
        origin = joint.find("origin")
        p, q = _parse_origin(origin)
        attrs = f'name="{prefix}_{child_name}"'
        if p:
            attrs += f" {p}"
        if q:
            attrs += f" {q}"
        body_lines.append(f"      <body {attrs}>")
        body_lines.append(f'        <inertial mass="0.01" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>')
        body_lines.extend(_build_body(child_name, 8))
        body_lines.append("      </body>")

    return "\n".join(body_lines)


def _build_fk_model():
    """Build a minimal MJCF with both Inspire hands for FK only."""
    right_body = _urdf_to_mjcf_fragments(
        str(_INSPIRE_DIR / "inspire_hand_right.urdf"), "R"
    )
    left_body = _urdf_to_mjcf_fragments(
        str(_INSPIRE_DIR / "inspire_hand_left.urdf"), "L"
    )

    return f"""\
<mujoco model="inspire_fk">
  <option gravity="0 0 0" integrator="Euler" timestep="0.01"/>
  <compiler angle="radian"/>
  <worldbody>
    <body name="R_root" pos="0.15 0 0">
{right_body}
    </body>
    <body name="L_root" pos="-0.15 0 0">
{left_body}
    </body>
  </worldbody>
</mujoco>"""


# ── Geometry helpers ──


def _hand_frame(wrist, mid_ref, idx_ref, pnk_ref):
    """Build a 3x3 wrist frame from palm geometry.

    Y = wrist -> middle-finger base
    Z = palm normal (Y x index->pinky)
    X = Y x Z
    """
    y = mid_ref - wrist
    yn = np.linalg.norm(y)
    if yn < 1e-8:
        return np.eye(3)
    y /= yn
    across = pnk_ref - idx_ref
    z = np.cross(y, across)
    zn = np.linalg.norm(z)
    if zn < 1e-8:
        return np.eye(3)
    z /= zn
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _sigmoid(x, c=0.0, w=1.0):
    return 1.0 / (1.0 + np.exp(np.clip(w * (x - c), -50, 50)))


def _rescale_dist(d):
    d = np.asarray(d)
    return np.where(d < EPS2, 0.0, np.where(d > EPS1, d, EPS1 / (EPS1 - EPS2) * (d - EPS2)))


# ── OptRetargeter ──


class OptRetargeter:
    """SLSQP-based retargeter for one Inspire hand (L or R)."""

    def __init__(self, model, prefix, w_shape=W_SHAPE, w_pinch=W_PINCH, w_curl=W_CURL):
        self.model = model
        self.fk = mujoco.MjData(model)
        self.prefix = prefix
        self.w_shape = w_shape
        self.w_pinch = w_pinch
        self.w_curl = w_curl
        self.ndof = len(_ACTUATED)

        # Actuated joint addresses & limits
        self.jnt_addr = np.empty(self.ndof, dtype=np.intp)
        self.lo = np.empty(self.ndof)
        self.hi = np.empty(self.ndof)
        for i, jn in enumerate(_ACTUATED):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{jn}")
            self.jnt_addr[i] = model.jnt_qposadr[jid]
            self.lo[i] = model.jnt_range[jid, 0]
            self.hi[i] = model.jnt_range[jid, 1]
        # Override thumb_pitch (idx 1) effective range: URDF says 0.6 but
        # real hardware maps 0-1000 to a larger physical range.
        self.hi[1] = 0.4
        self.bounds = list(zip(self.lo, self.hi))

        # Mimic joints
        n_mim = len(_MIMICS)
        self.mim_addr = np.empty(n_mim, dtype=np.intp)
        self.mim_src = np.empty(n_mim, dtype=np.intp)
        self.mim_mult = np.empty(n_mim)
        self.mim_off = np.empty(n_mim)
        for j, (mn, si, mu, off) in enumerate(_MIMICS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{mn}")
            self.mim_addr[j] = model.jnt_qposadr[jid]
            self.mim_src[j] = si
            self.mim_mult[j] = mu
            self.mim_off[j] = off

        # Body IDs
        self.wrist_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_hand_base_link"
        )
        self.tip_bids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{n}")
                for n in _TIP_BODIES
            ],
            dtype=np.intp,
        )
        self.meta_bids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{n}")
                for n in _META_BODIES
            ],
            dtype=np.intp,
        )

        self._inv_reach = None

        # Robot rest-pose reach & wrist frame (at q=0)
        self._do_fk(np.zeros(self.ndof))
        r_w = self.fk.xpos[self.wrist_bid].copy()
        r_tips = self.fk.xpos[self.tip_bids]
        self.robot_reach = np.linalg.norm(r_tips - r_w, axis=1)
        self._inv_reach = 1.0 / np.maximum(self.robot_reach, 1e-8)

        self.R_robot = _hand_frame(
            r_w,
            self.fk.xpos[self.meta_bids[1]],
            self.fk.xpos[self.meta_bids[0]],
            self.fk.xpos[self.meta_bids[2]],
        )

        self._human_max_reach = np.zeros(5)
        self._scale_printed = False

        self.q_prev = np.zeros(self.ndof)
        self.q_smooth = np.zeros(self.ndof)
        self.warm = False

    def _do_fk(self, q):
        self.fk.qpos[self.jnt_addr] = q
        self.fk.qpos[self.mim_addr] = q[self.mim_src] * self.mim_mult + self.mim_off
        mujoco.mj_fwdPosition(self.model, self.fk)

    def _cost(self, q, ref):
        self._do_fk(q)
        r_tips = self.fk.xpos[self.tip_bids]
        r_w = self.fk.xpos[self.wrist_bid]
        h_vecs, sw_shape, sw_pinch, pinch_tgt, curl_h = ref

        rv = r_tips - r_w
        shape_diff = rv - h_vecs
        cost = self.w_shape * np.sum(sw_shape * np.sum(shape_diff * shape_diff, axis=1))

        pinch_diff = (r_tips[1:] - r_tips[0]) - pinch_tgt
        cost += self.w_pinch * np.sum(sw_pinch * np.sum(pinch_diff * pinch_diff, axis=1))

        curl_r = 1.0 - np.linalg.norm(rv, axis=1) * self._inv_reach
        curl_diff = curl_h - curl_r
        cost += self.w_curl * (curl_diff @ curl_diff)

        dq = q - self.q_prev
        cost += W_VEL * (dq @ dq)
        return cost

    def retarget(self, landmarks):
        """Retarget from 25-joint landmarks (in MuJoCo Z-up frame).

        Returns (6,) array of actuated joint angles.
        """
        h_w = landmarks[H_WRIST]
        h_tips = landmarks[H_TIPS]

        R_h = _hand_frame(
            h_w, landmarks[H_MID_META], landmarks[H_IDX_META], landmarks[H_PNK_META]
        )
        R_align = self.R_robot @ R_h.T

        h_raw = (R_align @ (h_tips - h_w).T).T

        # Adaptive per-finger scaling
        h_lens = np.linalg.norm(h_raw, axis=1)
        self._human_max_reach = np.maximum(self._human_max_reach, h_lens)
        finger_scale = self.robot_reach / np.maximum(self._human_max_reach, 1e-4)

        if not self._scale_printed and np.all(self._human_max_reach > 0.01):
            print(
                f"  [{self.prefix}] auto-scale: "
                + " ".join(f"{s:.2f}" for s in finger_scale)
            )
            self._scale_printed = True

        h_vecs = h_raw * finger_scale[:, None]

        d_tf = np.linalg.norm(h_vecs[1:] - h_vecs[0], axis=1)

        sw_shape = np.empty(5)
        sw_shape[0] = _sigmoid(d_tf.min(), EPS1, -SIG_W)
        sw_shape[1:] = _sigmoid(d_tf, EPS1, -SIG_W)
        sw_pinch = _sigmoid(d_tf, EPS1, SIG_W)

        pinch_vecs = h_vecs[1:] - h_vecs[0]
        pinch_norms = np.linalg.norm(pinch_vecs, axis=1, keepdims=True)
        pinch_dirs = np.where(pinch_norms > 1e-8, pinch_vecs / pinch_norms, 0.0)
        pinch_tgt = pinch_dirs * _rescale_dist(d_tf)[:, None]

        curl_h = 1.0 - h_lens / np.maximum(self._human_max_reach, 1e-4)

        ref = (h_vecs, sw_shape, sw_pinch, pinch_tgt, curl_h)

        res = scipy_minimize(
            self._cost,
            self.q_prev,
            args=(ref,),
            method="SLSQP",
            bounds=self.bounds,
            options={"maxiter": 30, "ftol": 1e-6},
        )
        q_opt = np.clip(res.x, self.lo, self.hi)

        if not self.warm:
            self.q_smooth = q_opt.copy()
            self.warm = True
        else:
            self.q_smooth = EMA_ALPHA * q_opt + (1 - EMA_ALPHA) * self.q_smooth

        self.q_prev = self.q_smooth.copy()
        return self.q_smooth

    def normalize(self, q):
        """Normalize joint angles to [0, 1].  0 = open (q=lo), 1 = closed (q=hi)."""
        return (q - self.lo) / np.maximum(self.hi - self.lo, 1e-8)


# ── Main wrapper for teleop pipeline ──


class InspireOptRetargeting:
    """Drop-in replacement for dex-retargeting that uses optimization-based retargeting.

    Usage:
        retargeting = InspireOptRetargeting()
        left_hw, right_hw = retargeting.retarget(left_25x3, right_25x3)
        # left_hw, right_hw: (6,) in hardware format [0,1], 1=open, 0=closed
        # Order: [pinky, ring, middle, index, thumb_bend, thumb_rotation]
    """

    # Max register value per motor in hardware order:
    #   [pinky, ring, middle, index, thumb_bend, thumb_rotation]
    # All motors accept 0-1000.  "open" = 1000,  "closed" = 0.
    REG_MAX = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)

    def __init__(self):
        print("[OptRetargeting] Building FK model from Inspire hand URDFs...")
        xml = _build_fk_model()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.retarget_L = OptRetargeter(self.model, "L")
        self.retarget_R = OptRetargeter(self.model, "R")
        print("[OptRetargeting] Ready.")

    def _to_hardware(self, norm):
        """Convert optimizer-order normalized [0,1] to hardware register values.

        Optimizer order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
        Hardware order:  [pinky, ring, middle, index, thumb_bend, thumb_rotation]

        Returns register values (0 to REG_MAX per joint).
        """
        activation = np.clip(norm[::-1], 0.0, 1.0)  # reorder to hardware
        return self.REG_MAX * (1.0 - activation)     # open=REG_MAX, closed=0

    def retarget(self, left_lm, right_lm, coord_rot=None):
        """Retarget from 25-joint hand landmarks.

        Args:
            left_lm: (25, 3) left hand landmark positions
            right_lm: (25, 3) right hand landmark positions
            coord_rot: 3x3 rotation to convert landmarks to MuJoCo Z-up frame.
                       Default: R_Y2Z (for WebXR/Manus Y-up input).

        Returns:
            (left_regs, right_regs): each (6,) register values (0 to REG_MAX),
            or None if the corresponding hand has no valid data.
            Order: [pinky, ring, middle, index, thumb_bend, thumb_rotation]
        """
        if coord_rot is None:
            coord_rot = R_Y2Z

        left_regs = None
        right_regs = None

        if not np.allclose(left_lm, 0.0):
            left_mj = (coord_rot @ left_lm.T).T
            q_L = self.retarget_L.retarget(left_mj)
            left_regs = self._to_hardware(self.retarget_L.normalize(q_L))

        if not np.allclose(right_lm, 0.0):
            right_mj = (coord_rot @ right_lm.T).T
            q_R = self.retarget_R.retarget(right_mj)
            right_regs = self._to_hardware(self.retarget_R.normalize(q_R))

        return left_regs, right_regs
