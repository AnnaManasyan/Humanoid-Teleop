#!/usr/bin/env python3


# Run with: DYLD_LIBRARY_PATH=/Users/jbeisswenger/.local/share/uv/python/cpython-3.10.19-macos-aarch64-none/lib mjpython debug_hand_retargeting/debug_hand_retargeting.py

"""
debug_hand_retargeting.py

Quest 3 hand tracking → optimization-based retargeting → Inspire hand in MuJoCo.

Uses a combined fingertip-position + pinch objective (Xin et al. 2025) solved via
SLSQP over the Inspire hand's 6 actuated DOF.  Sigmoid switching smoothly hands
off between shape-matching (far) and pinch-matching (close) modes.  Distance
rescaling clamps noisy sub-cm tracking into hard contact.

Usage:
    DYLD_LIBRARY_PATH=... mjpython debug_hand_retargeting/debug_hand_retargeting.py
    DYLD_LIBRARY_PATH=... mjpython debug_hand_retargeting/debug_hand_retargeting.py --debug
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from multiprocessing import Array, Process
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from scipy.optimize import minimize as scipy_minimize
from scipy.spatial.transform import Rotation

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent.resolve()
DEX_RETARGET_DIR = SCRIPT_DIR / "dex-retargeting"
ROBOT_HANDS_DIR = DEX_RETARGET_DIR / "assets" / "robots" / "hands"
INSPIRE_DIR = ROBOT_HANDS_DIR / "inspire_hand"

# ═══════════════════════════════════════════════════════════════════
# OpenXR 25-joint hand layout & landmark visualization
# ═══════════════════════════════════════════════════════════════════
#
#   0       : wrist
#   1-4     : thumb    (metacarpal, proximal, distal, tip)
#   5-9     : index    (metacarpal, proximal, intermediate, distal, tip)
#   10-14   : middle
#   15-19   : ring
#   20-24   : pinky
#
# MediaPipe uses 21 joints with tips at [4, 8, 12, 16, 20].
# OpenXR uses 25 joints with tips at [4, 9, 14, 19, 24].

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),                    # thumb
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),            # index
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),   # middle
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),   # ring
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),   # pinky
]

TIP_INDICES = {4, 9, 14, 19, 24}

FINGER_RGBA = {
    "wrist": "1.0 1.0 1.0 0.4",
    "thumb": "0.9 0.2 0.2 0.4",
    "index": "0.2 0.9 0.2 0.4",
    "middle": "0.2 0.4 0.9 0.4",
    "ring": "0.9 0.9 0.2 0.4",
    "pinky": "0.9 0.2 0.9 0.4",
}

BONE_RGBA = {
    "thumb": "0.9 0.2 0.2 0.3",
    "index": "0.2 0.9 0.2 0.3",
    "middle": "0.2 0.4 0.9 0.3",
    "ring": "0.9 0.9 0.2 0.3",
    "pinky": "0.9 0.2 0.9 0.3",
}

# WebXR (Y-up) → MuJoCo (Z-up) rotation
R_Y2Z = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def _finger(idx):
    if idx == 0:
        return "wrist"
    if idx <= 4:
        return "thumb"
    if idx <= 9:
        return "index"
    if idx <= 14:
        return "middle"
    if idx <= 19:
        return "ring"
    return "pinky"


# ═══════════════════════════════════════════════════════════════════
# URDF → MJCF conversion helpers
# ═══════════════════════════════════════════════════════════════════


def _rpy_to_quat_str(rpy):
    """Convert [roll, pitch, yaw] (radians) to MuJoCo quaternion string 'w x y z'."""
    r = Rotation.from_euler("xyz", rpy)
    q = r.as_quat()  # scipy convention: [x, y, z, w]
    return f"{q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f}"


def _parse_origin(elem):
    """Return (pos_str, quat_str) from a URDF <origin> element."""
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


def urdf_to_mjcf_fragments(urdf_path, prefix):
    """Parse a URDF and return MJCF body-hierarchy XML and mesh assets.

    Returns:
        body_xml:    string of nested <body> elements
        mesh_assets: list of (asset_name, abs_file_path) — deduplicated
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = str(Path(urdf_path).parent)

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

    _mesh_map = {}
    mesh_assets = []
    hand_rgba = "0.8 0.75 0.7 1"

    def _add_mesh(abs_path, preferred_name):
        if abs_path in _mesh_map:
            return _mesh_map[abs_path]
        _mesh_map[abs_path] = preferred_name
        mesh_assets.append((preferred_name, abs_path))
        return preferred_name

    def _link_geoms(link_name, indent):
        link = links[link_name]
        lines = []
        visual = link.find("visual")
        if visual is not None:
            geom_el = visual.find("geometry")
            if geom_el is not None:
                mesh_el = geom_el.find("mesh")
                if mesh_el is not None:
                    glb_filename = mesh_el.get("filename")
                    if glb_filename:
                        obj_filename = glb_filename.replace("visual/", "visual_obj/").replace(".glb", ".obj")
                        abs_path = os.path.join(urdf_dir, obj_filename)
                        if os.path.isfile(abs_path):
                            name = _add_mesh(abs_path, f"{prefix}_{link_name}")
                            vis_origin = visual.find("origin")
                            p, q = _parse_origin(vis_origin)
                            attrs = f'type="mesh" mesh="{name}" rgba="{hand_rgba}" contype="0" conaffinity="0"'
                            if p:
                                attrs += f" {p}"
                            if q:
                                attrs += f" {q}"
                            lines.append(f'{" " * indent}<geom {attrs}/>')
                            return lines
        return lines

    def _build_body(link_name, indent):
        lines = []
        lines.extend(_link_geoms(link_name, indent))
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
        body_lines.extend(_build_body(child_name, 8))
        body_lines.append("      </body>")

    return "\n".join(body_lines), mesh_assets


# ═══════════════════════════════════════════════════════════════════
# Full MJCF builder — robot hands + landmark mocap bodies
# ═══════════════════════════════════════════════════════════════════


def build_mjcf():
    """Build MJCF with both inspire hands and landmark visualization bodies."""
    right_body, right_meshes = urdf_to_mjcf_fragments(
        str(INSPIRE_DIR / "inspire_hand_right.urdf"), "R"
    )
    left_body, left_meshes = urdf_to_mjcf_fragments(
        str(INSPIRE_DIR / "inspire_hand_left.urdf"), "L"
    )

    mesh_lines = []
    for name, fpath in right_meshes + left_meshes:
        mesh_lines.append(f'    <mesh name="{name}" file="{fpath}"/>')

    # Landmark mocap bodies (small semi-transparent spheres)
    landmark_lines = []
    for side in ("L", "R"):
        for i in range(25):
            name = f"{side}{i}"
            rgba = FINGER_RGBA[_finger(i)]
            r = "0.006" if i in TIP_INDICES else "0.004"
            landmark_lines += [
                f'    <body name="{name}" mocap="true" pos="0 0 -1">',
                f'      <site name="{name}s" size="0.001"/>',
                f'      <geom type="sphere" size="{r}" rgba="{rgba}" contype="0" conaffinity="0"/>',
                f"    </body>",
            ]

    # Tendon skeleton
    tendon_lines = ['  <tendon>']
    for side in ("L", "R"):
        for a, b in BONES:
            rgba = BONE_RGBA[_finger(b)]
            tendon_lines += [
                f'    <spatial limited="false" width="0.001" rgba="{rgba}">',
                f'      <site site="{side}{a}s"/>',
                f'      <site site="{side}{b}s"/>',
                f"    </spatial>",
            ]
    tendon_lines.append("  </tendon>")

    xml = f"""\
<mujoco model="inspire_hand_retargeting">
  <option gravity="0 0 0" integrator="Euler" timestep="0.01"/>
  <compiler angle="radian"/>
  <statistic center="0.3 0 1.2" extent="0.6"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>

  <asset>
{chr(10).join(mesh_lines)}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="3 3 0.01" rgba="0.12 0.12 0.12 1"/>

    <body name="R_root" pos="0.15 0 0.15">
{right_body}
    </body>

    <body name="L_root" pos="-0.15 0 0.15">
{left_body}
    </body>

{chr(10).join(landmark_lines)}
  </worldbody>

{chr(10).join(tendon_lines)}
</mujoco>"""

    return xml


# ═══════════════════════════════════════════════════════════════════
# Optimization-based retargeting  (Xin et al. 2025)
#
# 6 keypoints from OpenXR:  wrist(0), thumb(4), index(9),
#                            middle(14), ring(19), pinky(24)
# 6 actuated DOF on Inspire: thumb_yaw, thumb_pitch,
#                             index, middle, ring, pinky proximal
# ═══════════════════════════════════════════════════════════════════

# Human landmark indices
H_WRIST = 0
H_TIPS  = np.array([4, 9, 14, 19, 24])       # thumb, index, middle, ring, pinky

# Paper hyperparameters
EPS1     = 0.10    # sigmoid centre / distance-rescaling outer threshold  (m)
EPS2     = 0.01    # noise floor / distance-rescaling inner threshold     (m)
SIG_W    = 10.0    # sigmoid steepness
W_SHAPE  = 1.0     # λ₃  wrist→tip vectors
W_PINCH  = 10.0    # λ₅  thumb→finger vectors
W_CURL   = 0.5     # λ_curl  scalar finger-curl matching
W_VEL    = 0.05    # joint-velocity regularisation (smoothness)
EMA_ALPHA = 0.6    # exponential-moving-average smoothing factor

# Inspire hand actuated joints (order = optimisation variable vector)
#
# Joint limits (from URDF):
#   idx  joint                        lo    hi     notes
#   0    thumb_proximal_yaw_joint     0.0   1.308  thumb side-to-side
#   1    thumb_proximal_pitch_joint   0.0   0.6    thumb curl
#   2    index_proximal_joint         0.0   1.47   index curl
#   3    middle_proximal_joint        0.0   1.47   middle curl
#   4    ring_proximal_joint          0.0   1.47   ring curl
#   5    pinky_proximal_joint         0.0   1.47   pinky curl
#
# The real robot expects actions in [0, 1].  To convert:
#   action = (q - lo) / (hi - lo)
# Since all lo == 0, this simplifies to:  action = q / hi
#
_ACTUATED = [
    'thumb_proximal_yaw_joint',
    'thumb_proximal_pitch_joint',
    'index_proximal_joint',
    'middle_proximal_joint',
    'ring_proximal_joint',
    'pinky_proximal_joint',
]

# Mimic joints: (name, source-index in _ACTUATED, multiplier, offset)
_MIMICS = [
    ('thumb_intermediate_joint', 1, 1.334,   0.0),
    ('thumb_distal_joint',       1, 0.667,   0.0),
    ('index_intermediate_joint', 2, 1.06399, -0.04545),
    ('middle_intermediate_joint',3, 1.06399, -0.04545),
    ('ring_intermediate_joint',  4, 1.06399, -0.04545),
    ('pinky_intermediate_joint', 5, 1.06399, -0.04545),
]

# Robot fingertip body names (same order as H_TIPS)
_TIP_BODIES = [
    'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip',
]

# Robot metacarpal-equivalent bodies (for wrist frame — stable, don't move with curl)
_META_BODIES = ['index_proximal', 'middle_proximal', 'pinky_proximal']

# Human metacarpal landmark indices (same geometric role)
H_IDX_META = 5
H_MID_META = 10
H_PNK_META = 20


def _hand_frame(wrist, mid_ref, idx_ref, pnk_ref):
    """Build a 3×3 wrist frame from palm geometry.

    Uses the same geometric convention for both human and robot:
      Y = wrist → middle-finger base  (finger direction)
      Z = palm normal                  (Y × index→pinky)
      X = Y × Z
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
    """Vectorized sigmoid — works on scalars and arrays."""
    return 1.0 / (1.0 + np.exp(np.clip(w * (x - c), -50, 50)))


def _rescale_dist(d):
    """Vectorized piecewise-linear distance rescaling."""
    d = np.asarray(d)
    out = np.where(d < EPS2, 0.0, np.where(d > EPS1, d,
                   EPS1 / (EPS1 - EPS2) * (d - EPS2)))
    return out


class OptRetargeter:
    """SLSQP-based retargeter for one Inspire hand (L or R).

    Optimises 6 joint angles to minimise:
      λ₃·L_shape  +  λ₅·L_pinch
    with sigmoid switching so pinch dominates when fingers are close and
    shape dominates when they are far.
    """

    def __init__(self, model, prefix, w_shape=1.0, w_pinch=10.0, w_curl=0.5):
        self.model = model
        self.fk = mujoco.MjData(model)          # scratch-pad for FK
        self.prefix = prefix
        self.w_shape = w_shape
        self.w_pinch = w_pinch
        self.w_curl = w_curl
        self.ndof = len(_ACTUATED)

        # ── actuated joint addresses & limits ──
        self.jnt_addr = np.empty(self.ndof, dtype=np.intp)
        self.lo = np.empty(self.ndof)
        self.hi = np.empty(self.ndof)
        for i, jn in enumerate(_ACTUATED):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{prefix}_{jn}")
            self.jnt_addr[i] = model.jnt_qposadr[jid]
            self.lo[i] = model.jnt_range[jid, 0]
            self.hi[i] = model.jnt_range[jid, 1]
        self.bounds = list(zip(self.lo, self.hi))

        # ── mimic joint addresses (numpy arrays for vectorized apply) ──
        n_mim = len(_MIMICS)
        self.mim_addr = np.empty(n_mim, dtype=np.intp)
        self.mim_src  = np.empty(n_mim, dtype=np.intp)
        self.mim_mult = np.empty(n_mim)
        self.mim_off  = np.empty(n_mim)
        for j, (mn, si, mu, off) in enumerate(_MIMICS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{prefix}_{mn}")
            self.mim_addr[j] = model.jnt_qposadr[jid]
            self.mim_src[j]  = si
            self.mim_mult[j] = mu
            self.mim_off[j]  = off

        # ── body IDs for FK readout ──
        self.wrist_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_hand_base_link")
        self.tip_bids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              f"{prefix}_{n}")
            for n in _TIP_BODIES
        ], dtype=np.intp)

        # ── metacarpal body IDs (for wrist frame) ──
        self.meta_bids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              f"{prefix}_{n}")
            for n in _META_BODIES
        ], dtype=np.intp)

        self._inv_reach = None  # set after robot_reach is computed

        # Robot rest-pose finger reach & wrist frame (at q=0)
        self._do_fk(np.zeros(self.ndof))
        r_w = self.fk.xpos[self.wrist_bid].copy()
        r_tips = self.fk.xpos[self.tip_bids]
        self.robot_reach = np.linalg.norm(r_tips - r_w, axis=1)  # (5,)
        self._inv_reach = 1.0 / np.maximum(self.robot_reach, 1e-8)

        # Wrist frame from metacarpal geometry (same convention as human)
        self.R_robot = _hand_frame(
            r_w,
            self.fk.xpos[self.meta_bids[1]],  # middle proximal
            self.fk.xpos[self.meta_bids[0]],  # index proximal
            self.fk.xpos[self.meta_bids[2]],  # pinky proximal
        )

        # Adaptive per-finger scaling: tracks max observed wrist→tip distance
        self._human_max_reach = np.zeros(5)
        self._scale_printed = False

        # ── optimiser state ──
        self.q_prev   = np.zeros(self.ndof)
        self.q_smooth = np.zeros(self.ndof)
        self.warm     = False

    # ────────────────────── forward kinematics ──────────────────────

    def _do_fk(self, q):
        """Set actuated + mimic joints, run forward kinematics."""
        self.fk.qpos[self.jnt_addr] = q
        self.fk.qpos[self.mim_addr] = q[self.mim_src] * self.mim_mult + self.mim_off
        mujoco.mj_fwdPosition(self.model, self.fk)

    # ────────────────────── cost function ──────────────────────

    def _cost(self, q, ref):
        """Total retargeting cost.  `ref` is a pre-packed tuple."""
        self._do_fk(q)
        r_tips = self.fk.xpos[self.tip_bids]                    # (5, 3) view
        r_w    = self.fk.xpos[self.wrist_bid]                   # (3,) view

        h_vecs, sw_shape, sw_pinch, pinch_tgt, curl_h = ref

        # L_shape: wrist→tip vectors
        rv = r_tips - r_w
        shape_diff = rv - h_vecs                                 # (5, 3)
        cost = self.w_shape * np.sum(sw_shape * np.sum(shape_diff * shape_diff, axis=1))

        # L_pinch: thumb→finger vectors  (4 pairs, index..pinky)
        pinch_diff = (r_tips[1:] - r_tips[0]) - pinch_tgt       # (4, 3)
        cost += self.w_pinch * np.sum(sw_pinch * np.sum(pinch_diff * pinch_diff, axis=1))

        # L_curl: scalar finger curl (direction-independent)
        curl_r = 1.0 - np.linalg.norm(rv, axis=1) * self._inv_reach
        curl_diff = curl_h - curl_r
        cost += self.w_curl * (curl_diff @ curl_diff)

        # L_vel: smoothness — penalise jumps from previous solution
        dq = q - self.q_prev
        cost += W_VEL * (dq @ dq)

        return cost

    # ────────────────────── main entry point ──────────────────────

    def retarget(self, landmarks, debug=False):
        """Retarget from 25-joint OpenXR landmarks (MuJoCo Z-up frame).

        Returns (6,) array of actuated joint angles.
        """
        # ── extract keypoints ──
        h_w    = landmarks[H_WRIST]
        h_tips = landmarks[H_TIPS]                           # (5, 3)

        # ── align human wrist frame → robot wrist frame ──
        # Both frames use the same geometric convention (metacarpal-based),
        # so the rotation correctly cancels wrist orientation differences.
        R_h     = _hand_frame(h_w, landmarks[H_MID_META],
                              landmarks[H_IDX_META], landmarks[H_PNK_META])
        R_align = self.R_robot @ R_h.T

        # Wrist-relative vectors rotated into robot frame
        h_raw = (R_align @ (h_tips - h_w).T).T               # (5, 3)

        # ── adaptive per-finger scaling ──
        h_lens = np.linalg.norm(h_raw, axis=1)               # (5,)
        self._human_max_reach = np.maximum(self._human_max_reach, h_lens)
        finger_scale = self.robot_reach / np.maximum(
            self._human_max_reach, 1e-4)                      # (5,)

        if not self._scale_printed and np.all(self._human_max_reach > 0.01):
            print(f"  [{self.prefix}] auto-scale: "
                  + " ".join(f"{s:.2f}" for s in finger_scale))
            self._scale_printed = True

        h_vecs = h_raw * finger_scale[:, None]                # (5, 3)

        # ── thumb→finger distances (in the scaled space) ──
        d_tf = np.linalg.norm(h_vecs[1:] - h_vecs[0], axis=1)  # (4,)

        # ── sigmoid switching weights ──
        sw_shape = np.empty(5)
        sw_shape[0]  = _sigmoid(d_tf.min(), EPS1, -SIG_W)       # thumb
        sw_shape[1:] = _sigmoid(d_tf, EPS1, -SIG_W)
        sw_pinch = _sigmoid(d_tf, EPS1, SIG_W)                  # (4,)

        # ── distance-rescaled pinch targets ──
        pinch_vecs = h_vecs[1:] - h_vecs[0]                     # (4, 3)
        pinch_norms = np.linalg.norm(pinch_vecs, axis=1, keepdims=True)
        pinch_dirs = np.where(pinch_norms > 1e-8,
                              pinch_vecs / pinch_norms, 0.0)
        pinch_tgt = pinch_dirs * _rescale_dist(d_tf)[:, None]

        # ── scalar curl targets (direction-independent) ──
        curl_h = 1.0 - h_lens / np.maximum(self._human_max_reach, 1e-4)  # (5,)

        ref = (h_vecs, sw_shape, sw_pinch, pinch_tgt, curl_h)

        # ── SLSQP optimisation (warm-started) ──
        res = scipy_minimize(
            self._cost, self.q_prev, args=(ref,),
            method='SLSQP',
            bounds=self.bounds,
            options={'maxiter': 30, 'ftol': 1e-6},
        )
        q_opt = np.clip(res.x, self.lo, self.hi)

        # ── EMA smoothing ──
        if not self.warm:
            self.q_smooth = q_opt.copy()
            self.warm = True
        else:
            self.q_smooth = EMA_ALPHA * q_opt + (1 - EMA_ALPHA) * self.q_smooth

        self.q_prev = self.q_smooth.copy()

        if debug:
            labels = ['th_yaw', 'th_pit', 'idx', 'mid', 'rng', 'pnk']
            vals   = ' '.join(f'{labels[i]}={self.q_smooth[i]:.3f}'
                              for i in range(self.ndof))
            nvals  = self.normalize(self.q_smooth)
            nstr   = ' '.join(f'{labels[i]}={nvals[i]:.3f}'
                              for i in range(self.ndof))
            print(f"  opt iters={res.nit:2d}  cost={res.fun:.5f}  {vals}")
            print(f"  action [0-1]: {nstr}")

        return self.q_smooth

    def normalize(self, q):
        """Normalize joint angles to [0, 1] for the real robot.

        action_i = (q_i - lo_i) / (hi_i - lo_i)
        """
        return (q - self.lo) / np.maximum(self.hi - self.lo, 1e-8)

    def apply(self, main_data, q):
        """Write joint angles (actuated + mimic) into a live MjData."""
        main_data.qpos[self.jnt_addr] = q
        main_data.qpos[self.mim_addr] = q[self.mim_src] * self.mim_mult + self.mim_off


# ═══════════════════════════════════════════════════════════════════
# Vuer hand-tracking server (Quest 3)
# ═══════════════════════════════════════════════════════════════════


def _vuer_process(cert_file, key_file, left_lm, right_lm, head_mat):
    """Vuer child process — receives HAND_MOVE events from Quest 3."""
    import asyncio
    from vuer import Vuer
    from vuer.schemas import DefaultScene, Hands, Sphere

    app = Vuer(
        host="0.0.0.0",
        cert=cert_file,
        key=key_file,
        queries=dict(grid=False),
        queue_len=3,
    )

    @app.add_handler("CAMERA_MOVE")
    async def on_cam(event, session, fps=60):
        try:
            head_mat[:] = event.value["camera"]["matrix"]
        except Exception:
            pass

    _counter = [0]
    _got_data = [False]

    @app.add_handler("HAND_MOVE")
    async def on_hand(event, session, fps=60):
        try:
            v = event.value
            ll = v.get("leftLandmarks")
            rl = v.get("rightLandmarks")

            _counter[0] += 1

            if not _got_data[0] and isinstance(ll, list) and len(ll) == 25:
                _got_data[0] = True
                print(f"[HAND_MOVE #{_counter[0]}] Tracking active!")

            if isinstance(ll, list) and len(ll) == 25 and isinstance(ll[0], (list, tuple)):
                left_lm[:] = np.array(ll, dtype=np.float64).flatten()

            if isinstance(rl, list) and len(rl) == 25 and isinstance(rl[0], (list, tuple)):
                right_lm[:] = np.array(rl, dtype=np.float64).flatten()

            if _counter[0] % 300 == 0:
                has = isinstance(ll, list) and len(ll) == 25
                print(f"  ... {_counter[0]} events, tracking={'YES' if has else 'NO'}")

            if isinstance(ll, list) and len(ll) == 25 and isinstance(ll[0], (list, tuple)):
                flat = np.array(ll, dtype=np.float64).flatten()
                if len(flat) == 75:
                    left_lm[:] = flat

            if isinstance(rl, list) and len(rl) == 25 and isinstance(rl[0], (list, tuple)):
                flat = np.array(rl, dtype=np.float64).flatten()
                if len(flat) == 75:
                    right_lm[:] = flat

        except Exception as e:
            print(f"[HAND_MOVE] error: {e}")

    @app.spawn(start=False)
    async def scene(session, fps=60):
        session.set @ DefaultScene(frameloop="always")
        session.upsert @ Sphere(
            key="marker",
            args=[0.05],
            position=[0, 1.2, -0.5],
            materialType="standard",
            material=dict(color="#444444"),
        )
        session.upsert @ Hands(
            fps=fps,
            stream=True,
            key="hands",
            showLeft=True,
            showRight=True,
        )
        print("[Vuer] Scene ready — tap 'Enter VR' on the Quest")
        while True:
            await asyncio.sleep(1.0)

    app.run()


class HandTrackingServer:
    """Lightweight Vuer server that receives hand-tracking data via shared memory."""

    def __init__(self, cert_file="./cert.pem", key_file="./key.pem"):
        self.left_lm_shared = Array("d", 75, lock=True)
        self.right_lm_shared = Array("d", 75, lock=True)
        self.head_mat_shared = Array("d", 16, lock=True)

        self._proc = Process(
            target=_vuer_process,
            args=(cert_file, key_file, self.left_lm_shared, self.right_lm_shared, self.head_mat_shared),
            daemon=True,
        )
        self._proc.start()

    def get_landmarks(self):
        """Return (left, right), each (25, 3), in WebXR Y-up frame."""
        left = np.array(self.left_lm_shared[:]).reshape(25, 3)
        right = np.array(self.right_lm_shared[:]).reshape(25, 3)
        return left, right


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description="Quest 3 → optimization-based retargeting → Inspire hands in MuJoCo")
    ap.add_argument("--cert", default="./cert.pem", help="SSL certificate")
    ap.add_argument("--key", default="./key.pem", help="SSL private key")
    ap.add_argument("--w-shape", type=float, default=W_SHAPE,
                    help="Weight for fingertip-position term (λ₃)")
    ap.add_argument("--w-pinch", type=float, default=W_PINCH,
                    help="Weight for pinch term (λ₅)")
    ap.add_argument("--w-curl", type=float, default=W_CURL,
                    help="Weight for scalar curl term (λ_curl)")
    ap.add_argument("--debug", action="store_true",
                    help="Print optimiser stats every second")
    args = ap.parse_args()

    for path, label in [(args.cert, "cert"), (args.key, "key")]:
        if not os.path.isfile(path):
            print(f"ERROR: SSL {label} file not found: {path}")
            print("Generate with: openssl req -x509 -nodes -days 365 "
                  "-newkey rsa:2048 -keyout key.pem -out cert.pem")
            sys.exit(1)

    # ── Build MuJoCo model ──
    print("Building MJCF from inspire hand URDFs ...")
    xml = build_mjcf()

    debug_xml_path = SCRIPT_DIR / "debug_mjcf.xml"
    with open(debug_xml_path, "w") as f:
        f.write(xml)
    print(f"  Saved debug XML → {debug_xml_path}")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # ── Landmark mocap body IDs ──
    mid_L = np.array([model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"L{i}")]
        for i in range(25)], dtype=np.intp)
    mid_R = np.array([model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"R{i}")]
        for i in range(25)], dtype=np.intp)

    # ── Optimisation retargeters (per-finger scaling computed automatically) ──
    retarget_R = OptRetargeter(model, "R",
                               w_shape=args.w_shape, w_pinch=args.w_pinch,
                               w_curl=args.w_curl)
    retarget_L = OptRetargeter(model, "L",
                               w_shape=args.w_shape, w_pinch=args.w_pinch,
                               w_curl=args.w_curl)

    print(f"  w_shape={args.w_shape}  w_pinch={args.w_pinch}  w_curl={args.w_curl}")
    print(f"  EPS1={EPS1}  EPS2={EPS2}  EMA_ALPHA={EMA_ALPHA}")

    # ── Start Vuer server ──
    print("Starting Vuer hand-tracking server ...")
    server = HandTrackingServer(cert_file=args.cert, key_file=args.key)
    print("  1. Open the Vuer URL on the Quest 3 browser")
    print("  2. Tap 'Enter VR' to start hand tracking")
    print("  3. Put controllers down — Quest uses camera-based hand tracking")
    print("Press Ctrl-C or close the viewer to quit.\n")

    got_left = got_right = False
    frame = 0

    # ── Benchmarking state ──
    bench_times_L = []
    bench_times_R = []
    bench_interval = 120  # print stats every N frames
    loop_t_prev = time.perf_counter()
    loop_times = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                loop_t_now = time.perf_counter()
                loop_times.append(loop_t_now - loop_t_prev)
                loop_t_prev = loop_t_now

                left_lm, right_lm = server.get_landmarks()
                do_debug = args.debug and frame % 60 == 0

                # ── Left hand ──
                if not np.allclose(left_lm, 0.0):
                    if not got_left:
                        print("[OK] LEFT hand tracking active")
                        got_left = True

                    pts_mj = (R_Y2Z @ left_lm.T).T
                    data.mocap_pos[mid_L] = pts_mj

                    if do_debug:
                        print("LEFT:")
                    t0 = time.perf_counter()
                    q = retarget_L.retarget(pts_mj, debug=do_debug)
                    bench_times_L.append(time.perf_counter() - t0)
                    retarget_L.apply(data, q)

                # ── Right hand ──
                if not np.allclose(right_lm, 0.0):
                    if not got_right:
                        print("[OK] RIGHT hand tracking active")
                        got_right = True

                    pts_mj = (R_Y2Z @ right_lm.T).T
                    data.mocap_pos[mid_R] = pts_mj

                    if do_debug:
                        print("RIGHT:")
                    t0 = time.perf_counter()
                    q = retarget_R.retarget(pts_mj, debug=do_debug)
                    bench_times_R.append(time.perf_counter() - t0)
                    retarget_R.apply(data, q)

                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.0 / 60.0)
                frame += 1

                # ── Print benchmark stats periodically ──
                if frame % bench_interval == 0:
                    parts = [f"[bench frame={frame}]"]
                    for label, times in [("L", bench_times_L), ("R", bench_times_R)]:
                        if times:
                            arr = np.array(times) * 1000  # ms
                            parts.append(
                                f"  {label}: n={len(arr)} "
                                f"avg={arr.mean():.2f}ms "
                                f"med={np.median(arr):.2f}ms "
                                f"min={arr.min():.2f}ms "
                                f"max={arr.max():.2f}ms "
                                f"({1000/arr.mean():.0f} Hz)"
                            )
                    if loop_times:
                        lt = np.array(loop_times) * 1000
                        parts.append(
                            f"  loop: avg={lt.mean():.2f}ms "
                            f"({1000/lt.mean():.0f} Hz)"
                        )
                    print("\n".join(parts))
                    bench_times_L.clear()
                    bench_times_R.clear()
                    loop_times.clear()

        except KeyboardInterrupt:
            # Print final benchmark summary
            parts = ["\nShutting down."]
            for label, times in [("L", bench_times_L), ("R", bench_times_R)]:
                if times:
                    arr = np.array(times) * 1000
                    parts.append(
                        f"  final {label}: n={len(arr)} "
                        f"avg={arr.mean():.2f}ms "
                        f"med={np.median(arr):.2f}ms "
                        f"min={arr.min():.2f}ms "
                        f"max={arr.max():.2f}ms"
                    )
            print("\n".join(parts))


if __name__ == "__main__":
    main()
