"""試行2の共通部品: 自作グリッパ付き Franka・上限つき吸着・姿勢計算・結果の記録。

把持ツール（自作）
  - 平行グリッパ: 指1本あたり 0〜0.105 m 開く（最大開き幅 0.21 m）。本の幅（0.13 m）も厚み（0.02 m）も挟める。指のパッドは幅 6 cm
  - 中央の吸盤: プランジャで 0〜0.08 m 伸縮する。伸ばすと指先より 0.04 m 先に出る
  - 吸着は補助。吸盤が本に接している間だけ、上限つきの力で本を吸盤側へ引く（表面ごとに上限が違う）
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

import genesis as gs

# 自作グリッパの寸法（gripper リンク座標。z が手先方向）
FINGER_BASE_Z = 0.04
FINGER_LEN = 0.14
FINGERTIP_Z = FINGER_BASE_Z + FINGER_LEN  # 0.18
CUP_TIP_RETRACTED_Z = 0.14
PLUNGER_STROKE = 0.08
CUP_TIP_EXTENDED_Z = CUP_TIP_RETRACTED_Z + PLUNGER_STROKE  # 0.22
FINGER_OPEN = 0.105

# dof の並び: 腕 0-6 / 左指 7 / 右指 8 / プランジャ 9
ARM = np.arange(7)
FINGERS = np.array([7, 8])
PLUNGER = 9
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 8000, 8000, 3000], dtype=float)  # 指は強め（把持力が上限 60 N まで出るように）
KV = np.array([450, 450, 350, 350, 200, 200, 200, 100, 100, 150], dtype=float)
FORCE = np.array([87, 87, 87, 87, 12, 12, 12, 60, 60, 40], dtype=float)  # 指の把持力は最大 60 N
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.75, -0.785, FINGER_OPEN, FINGER_OPEN, PLUNGER_STROKE])

# 本の表面ごとの性質（摩擦係数・吸着力の上限 N）。本の重さは既定密度で約 4.9 N
SURFACES = {
    "smooth": {"friction": 0.4, "suction_max_n": 12.0, "color": (0.75, 0.25, 0.25)},
    "rough": {"friction": 0.9, "suction_max_n": 3.0, "color": (0.45, 0.55, 0.30)},
}

GRIPPER_XML = f"""
<body name="gripper" pos="0 0 0">
  <inertial mass="0.6" pos="0 0 0.03" diaginertia="0.002 0.002 0.001"/>
  <geom name="palm" type="box" size="0.03 0.13 0.02" pos="0 0 0.02" rgba="0.30 0.30 0.32 1"/>
  <body name="finger_l" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.05" pos="0 0.006 0.07" diaginertia="1e-4 1e-4 1e-5"/>
    <joint name="finger_l_joint" type="slide" axis="0 1 0" range="0 {FINGER_OPEN}" damping="5" armature="0.1"/>
    <geom type="box" size="0.03 0.006 {FINGER_LEN / 2}" pos="0 0.006 {FINGER_LEN / 2}"
          friction="1.5 0.01 0.001" rgba="0.85 0.85 0.85 1"/>
  </body>
  <body name="finger_r" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.05" pos="0 -0.006 0.07" diaginertia="1e-4 1e-4 1e-5"/>
    <joint name="finger_r_joint" type="slide" axis="0 -1 0" range="0 {FINGER_OPEN}" damping="5" armature="0.1"/>
    <geom type="box" size="0.03 0.006 {FINGER_LEN / 2}" pos="0 -0.006 {FINGER_LEN / 2}"
          friction="1.5 0.01 0.001" rgba="0.85 0.85 0.85 1"/>
  </body>
  <body name="plunger" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.03" pos="0 0 0.05" diaginertia="2e-5 2e-5 1e-6"/>
    <joint name="plunger_joint" type="slide" axis="0 0 1" range="0 {PLUNGER_STROKE}" damping="5" armature="0.1"/>
    <geom type="cylinder" size="0.005 0.045" pos="0 0 0.045" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
    <geom name="cup" type="cylinder" size="0.016 0.005" pos="0 0 {CUP_TIP_RETRACTED_Z - FINGER_BASE_Z - 0.005}"
          friction="1.0 0.01 0.001" rgba="0.10 0.40 0.90 1"/>
  </body>
</body>
"""


def build_robot_xml(out_dir: Path) -> Path:
    """Genesis 同梱の panda_nohand.xml（Apache-2.0）に自作グリッパを取り付けた MJCF を実行時に生成する。"""
    src_dir = Path(gs.__file__).parent / "assets" / "xml" / "franka_emika_panda"
    xml = (src_dir / "panda_nohand.xml").read_text()
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{src_dir / "assets"}"', xml, count=1)
    anchor = '<site name="attachment_site"/>'
    assert anchor in xml
    xml = xml.replace(anchor, anchor + GRIPPER_XML, 1)
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    path = out_dir / "panda_book_gripper.xml"
    path.write_text(xml)
    return path


# ---------- 姿勢計算（quat は w, x, y, z） ----------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_z(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=float)
    z = np.zeros_like(yaw)
    return np.stack([np.cos(yaw / 2), z, z, np.sin(yaw / 2)], axis=-1)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return np.array([w, x, y, z])


Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # gripper の z を真下に向ける


def down_quat_with_finger_axis(yaw_of_axis: np.ndarray) -> np.ndarray:
    """真下向きで、指の開閉軸を水平方向 yaw_of_axis に合わせる姿勢。軸は ±180° 同値なので手首が回りすぎない側を選ぶ。"""
    psi = np.asarray(yaw_of_axis, dtype=float) + math.pi / 2
    psi = (psi + math.pi / 2) % math.pi - math.pi / 2
    return quat_mul(quat_z(psi), np.broadcast_to(Q_DOWN, psi.shape + (4,)))


def yaw_of(q: np.ndarray) -> np.ndarray:
    m = quat_to_mat(q)
    return np.arctan2(m[..., 1, 0], m[..., 0, 0])


# ---------- ロボットの操作 ----------

class Robot:
    def __init__(self, scene, entity, n_envs: int):
        self.scene, self.e, self.n = scene, entity, n_envs
        self.gripper = entity.get_link("gripper")
        self.plunger = entity.get_link("plunger")
        self.finger_links = {entity.get_link("finger_l").idx, entity.get_link("finger_r").idx}
        self.target = np.tile(HOME, (n_envs, 1))
        self.suction = None  # Suction を後から差す

    def setup(self) -> None:
        dofs = np.arange(10)
        self.e.set_dofs_kp(KP, dofs)
        self.e.set_dofs_kv(KV, dofs)
        self.e.set_dofs_force_range(-FORCE, FORCE, dofs)
        self.e.set_qpos(self.target)
        self.e.control_dofs_position(self.target)

    def step(self, k: int = 1, hook=None) -> None:
        for _ in range(k):
            self.e.control_dofs_position(self.target)
            if self.suction is not None:
                self.suction.apply()
            if hook is not None:
                hook()
            self.scene.step()

    def ik(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        q, err = self.e.inverse_kinematics(link=self.gripper, pos=pos, quat=quat, dofs_idx_local=ARM,
                                           init_qpos=self.e.get_qpos(), return_error=True)
        e = err.cpu().numpy().reshape(self.n, -1)
        self.ik_err_max = np.maximum(getattr(self, "ik_err_max", np.zeros(self.n)), np.linalg.norm(e[:, :3], axis=1))
        return q.cpu().numpy()[:, :7]

    def move_arm(self, pos: np.ndarray, quat: np.ndarray, steps: int, hook=None) -> None:
        goal = self.ik(pos, quat)
        start = self.target[:, :7].copy()
        for k in range(1, steps + 1):
            self.target[:, :7] = start + (goal - start) * (k / steps)
            self.step(1, hook)

    def move_linear(self, pos: np.ndarray, quat: np.ndarray, steps: int, n_way: int = 12, hook=None) -> None:
        """手先を直線上の中継点に沿って動かす（遠い目標で IK が別の解の枝へ飛び、腕が振り回されるのを防ぐ）。"""
        p0 = self.gripper.get_pos().cpu().numpy()
        q0 = self.gripper.get_quat().cpu().numpy()
        q1 = np.where((np.sum(q0 * quat, axis=1) < 0)[:, None], -quat, quat)
        per = max(1, steps // n_way)
        for w in range(1, n_way + 1):
            a = w / n_way
            qw = q0 + (q1 - q0) * a
            qw /= np.linalg.norm(qw, axis=1, keepdims=True)
            goal = self.ik(p0 + (pos - p0) * a, qw)
            start = self.target[:, :7].copy()
            for k in range(1, per + 1):
                self.target[:, :7] = start + (goal - start) * (k / per)
                self.step(1, hook)

    def correct(self, pos: np.ndarray, quat: np.ndarray, iters: int = 2, steps: int = 30) -> np.ndarray:
        """重力によるたわみ（定常誤差）を、実測の誤差だけ目標をずらして打ち消す。戻り値は補正後の誤差(m)。"""
        aim = pos.copy()
        for _ in range(iters):
            self.step(40)  # 動いている途中で測ると補正が過大になるので、先に落ち着かせる
            err = pos - self.gripper.get_pos().cpu().numpy()
            norm = np.linalg.norm(err, axis=1, keepdims=True)
            err = np.where(norm > 0.02, err * (0.02 / np.maximum(norm, 1e-9)), err)  # 1 回の補正は 2 cm まで
            aim = aim + err
            goal = self.ik(aim, quat)
            start = self.target[:, :7].copy()
            for k in range(1, steps + 1):
                self.target[:, :7] = start + (goal - start) * (k / steps)
                self.step(1)
        return np.linalg.norm(pos - self.gripper.get_pos().cpu().numpy(), axis=1)

    def move_joints(self, idx, goal: np.ndarray, steps: int, hook=None) -> None:
        start = self.target[:, idx].copy()
        for k in range(1, steps + 1):
            self.target[:, idx] = start + (goal - start) * (k / steps)
            self.step(1, hook)

    def cup_tip(self) -> np.ndarray:
        p = self.plunger.get_pos().cpu().numpy()
        m = quat_to_mat(self.plunger.get_quat().cpu().numpy())
        return p + m[:, :, 2] * (CUP_TIP_RETRACTED_Z - FINGER_BASE_Z)

    def approach_axis(self) -> np.ndarray:
        return quat_to_mat(self.gripper.get_quat().cpu().numpy())[:, :, 2]

    def finger_opening(self) -> np.ndarray:
        q = self.e.get_qpos().cpu().numpy()
        return q[:, 7] + q[:, 8]

    def touching(self, book) -> tuple[np.ndarray, np.ndarray]:
        """(指が本に触れているか, 吸盤が本に触れているか) を環境ごとに返す。"""
        ci = self.e.get_contacts(with_entity=book)
        la, lb = ci["link_a"].cpu().numpy(), ci["link_b"].cpu().numpy()
        valid = ci["valid_mask"].cpu().numpy()
        fingers = list(self.finger_links)
        f = ((np.isin(la, fingers) | np.isin(lb, fingers)) & valid).any(axis=1)
        c = (((la == self.plunger.idx) | (lb == self.plunger.idx)) & valid).any(axis=1)
        return f, c


class Suction:
    """上限つきの吸着。吸盤が本に接した時点で密着し、その位置から 5 mm 以上ずれたら外れる。"""

    def __init__(self, robot: Robot, book, max_force_n: float):
        self.r, self.book, self.fmax = robot, book, max_force_n
        n = robot.n
        self.on = np.zeros(n, bool)
        self.sealed = np.zeros(n, bool)
        self.anchor_local = np.zeros((n, 3))
        self.ever_sealed = np.zeros(n, bool)

    def apply(self) -> None:
        if not self.on.any():
            return
        tip = self.r.cup_tip()
        bpos = self.book.get_pos().cpu().numpy()
        bmat = quat_to_mat(self.book.get_quat().cpu().numpy())
        _, cup_touch = self.r.touching(self.book)
        new = self.on & ~self.sealed & cup_touch
        if new.any():
            self.sealed |= new
            self.ever_sealed |= new
            self.anchor_local[new] = np.einsum("nji,nj->ni", bmat[new], tip[new] - bpos[new])
        anchor_world = bpos + np.einsum("nij,nj->ni", bmat, self.anchor_local)
        drift = np.linalg.norm(anchor_world - tip, axis=1)
        self.sealed &= self.on & (drift < 0.005)
        idx = np.nonzero(self.sealed)[0]
        if len(idx):
            f = -self.r.approach_axis()[idx] * self.fmax
            self.book.base_link.apply_external_force(f, envs_idx=idx, pos=anchor_world[idx])
            self.r.plunger.apply_external_force(-f, envs_idx=idx, pos=tip[idx])


# ---------- 実行の記録 ----------

def new_run(trial: str, args) -> tuple[str, Path, dict]:
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = Path("/workspace/outputs") / trial / run_id  # named volume（OneDrive の外）
    out.mkdir(parents=True, exist_ok=True)
    meta = {"trial": trial, "run_id": run_id, "args": vars(args), "genesis": gs.__version__,
            "git_commit": _git_commit(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return run_id, out, meta


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_result(out: Path, meta: dict, result: dict) -> None:
    meta = {**meta, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **result}
    (out / "result.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_np_default))
    summary = {k: v for k, v in meta.items() if k != "per_env"}
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_np_default))


def _np_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return round(float(o), 4)
    raise TypeError(type(o))


def init_genesis(seed: int) -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    gs.init(backend=gs.gpu, seed=seed, logging_level="warning")


def make_scene(n_envs: int, dt: float = 0.01):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
        show_viewer=False,
    )
