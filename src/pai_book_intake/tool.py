"""自作グリッパ付き Franka（ツール v2）と、上限つき吸着。spikes/trial2_common.py から本体へ移したもの。

把持ツール v2（自作）
  - 平行グリッパ: 指1本あたり 0〜0.105 m 開く（最大開き幅 0.21 m）。指のパッドは幅 6 cm
  - 右の指先に内側へ 2.5 cm 張り出した薄い爪（本の下に滑り込ませる用。試行3 の scoop）と、上下 ±3 cm の関節
  - 中央の吸盤: プランジャで 0〜0.08 m 伸縮する。伸ばすと指先より 0.04 m 先に出る
  - 吸着は補助。吸盤が本に接している間だけ、上限つきの力で本を吸盤側へ引く（表面ごとに上限が違う）
  - 手の形（爪の長さ・厚み・先端・左右など）は hand_design.HandDesign で変えられる（試行5）。
    build_robot_xml(design=...) で作る。省略すると従来どおり v2（rounded=False なら v1）。dof の並びは変わらない

試行との違い
  - 1 つのシーンにロボットが 2 台ある。シーンを 1 歩進めるのは World.step だけにし、
    その中で全ロボットへ目標姿勢を送り続け、全ての吸着を効かせる（止まっている側も毎回指令する）
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import genesis as gs

from .geometry import down_quat_branch, quat_to_mat, wrap_pi, wrist_psi
from .hand_design import V1, V2, HandDesign, insert_into_robot_xml, xml_file_name
from .hand_design import gripper_xml as hand_gripper_xml

# 自作グリッパの寸法（gripper リンク座標。z が手先方向）
FINGER_BASE_Z = 0.04
FINGER_LEN = 0.14
FINGERTIP_Z = FINGER_BASE_Z + FINGER_LEN  # 0.18
CUP_TIP_RETRACTED_Z = 0.14
PLUNGER_STROKE = 0.08
CUP_TIP_EXTENDED_Z = CUP_TIP_RETRACTED_Z + PLUNGER_STROKE  # 0.22
FINGER_OPEN = 0.105
LIP_LEN = 0.025  # v2 の右の爪の長さ（内側へ張り出す）
LIP_T = 0.003  # 爪の厚み

# dof の並び: 腕 0-6 / 左指 7 / 右指 8 / 右指の上下 9 / プランジャ 10
ARM = np.arange(7)
FINGERS = np.array([7, 8])
LIFT = 9
PLUNGER = 10
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 8000, 8000, 8000, 3000], dtype=float)
KV = np.array([450, 450, 350, 350, 200, 200, 200, 100, 100, 100, 150], dtype=float)
FORCE = np.array([87, 87, 87, 87, 12, 12, 12, 60, 60, 60, 40], dtype=float)  # 指の把持力は最大 60 N
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.75, -0.785, FINGER_OPEN, FINGER_OPEN, 0.0, PLUNGER_STROKE])
GRIP_LIMIT_ROUGH_N = 25.0  # ざらざらの本は把持力を抑える（試行3）


def gripper_xml(rounded: bool = True, design: HandDesign | None = None) -> str:
    """自作グリッパの MJCF。design を省略すると rounded=True は v2（右の指先に爪）、False は v1（爪なし）。
    形の定義と MJCF の生成は hand_design.py（Genesis なしでテストできる）。"""
    return hand_gripper_xml(design if design is not None else (V2 if rounded else V1))


def build_robot_xml(out_dir: Path, rounded: bool = True, design: HandDesign | None = None) -> Path:
    """Genesis 同梱の panda_nohand.xml（Apache-2.0）に自作グリッパを取り付けた MJCF を実行時に生成する。
    生成物は実行の出力先（named volume）に置く（同梱アセットの絶対パスを含むため、リポジトリには置かない）。
    design を省略すると従来どおり（rounded=True は v2、False は v1）。"""
    d = design if design is not None else (V2 if rounded else V1)
    src_dir = Path(gs.__file__).parent / "assets" / "xml" / "franka_emika_panda"
    xml = (src_dir / "panda_nohand.xml").read_text()
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{src_dir / "assets"}"', xml, count=1)
    if '<site name="attachment_site"/>' not in xml:
        raise RuntimeError("panda_nohand.xml に attachment_site が無い（Genesis の版が想定と違う）")
    xml = insert_into_robot_xml(xml, d)
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    path = Path(out_dir) / xml_file_name(d)
    path.write_text(xml)
    return path


def home_for(design: HandDesign) -> np.ndarray:
    """手の形に合わせた待機姿勢（指は開き切る）。"""
    h = HOME.copy()
    h[FINGERS] = design.finger_open_m
    return h


class World:
    """シーンを進める唯一の入口。全ロボットへ毎歩、目標姿勢を送り、全ての吸着を効かせる。"""

    def __init__(self, scene, n_envs: int, dt: float):
        self.scene, self.n, self.dt = scene, n_envs, dt
        self.robots: list[Robot] = []
        self.monitors: list = []  # 毎歩の後に呼ぶ関数（技能の中の歩みも含めて、全部の歩みを見張るため）
        self.steps = 0

    @property
    def sim_time(self) -> float:
        return self.steps * self.dt

    def step(self, k: int = 1, hook=None) -> None:
        for _ in range(k):
            for r in self.robots:
                # 止めている環境では、止めた時点の目標を送り続ける（技能が全環境ぶんの目標を書き換えても動かない）
                r.e.control_dofs_position(np.where(r.frozen[:, None], r.frozen_target, r.target))
                r.suction.on &= ~r.frozen
                if r.suction.on.any():
                    r.suction.apply()
            if hook is not None:
                hook()
            self.scene.step()
            self.steps += 1
            for m in self.monitors:
                m()


class Robot:
    def __init__(self, world: World, entity, name: str, home: np.ndarray | None = None,
                 base_xy: tuple[float, float] = (0.0, 0.0)):
        self.world, self.e, self.name, self.n = world, entity, name, world.n
        self.base_xy = np.array(base_xy, dtype=float)
        self.gripper = entity.get_link("gripper")
        self.plunger = entity.get_link("plunger")
        self.finger_links = {entity.get_link("finger_l").idx, entity.get_link("finger_r_tip").idx}
        self.home = np.array(HOME if home is None else home, dtype=float)
        self.target = np.tile(self.home, (self.n, 1))
        self.suction = Suction(self)
        self.ik_err_max = np.zeros(self.n)
        self.frozen = np.zeros(self.n, bool)  # True の環境ではこのロボットを動かさない（運搬許可の無い本に触れない等）
        self.frozen_target = self.target.copy()
        world.robots.append(self)

    def freeze(self, mask: np.ndarray) -> None:
        """mask の環境で、いまの目標のまま止める。解くまで技能を呼んでも動かない。"""
        mask = np.asarray(mask, bool)
        self.frozen_target[mask] = self.target[mask]
        self.frozen = self.frozen | mask

    def unfreeze(self) -> None:
        """止めていた環境の目標を、止めた時点の値に戻してから解く（止めている間の書き換えを持ち越さない）。"""
        self.target[self.frozen] = self.frozen_target[self.frozen]
        self.frozen = np.zeros(self.n, bool)

    def setup(self, grip_limit_n: float | None = None) -> None:
        dofs = np.arange(len(HOME))
        self.e.set_dofs_kp(KP, dofs)
        self.e.set_dofs_kv(KV, dofs)
        self.set_grip_limit(grip_limit_n)
        self.e.set_qpos(self.target)
        self.e.control_dofs_position(self.target)

    def set_grip_limit(self, grip_limit_n: float | None) -> None:
        force = FORCE.copy()
        if grip_limit_n is not None:
            force[FINGERS] = grip_limit_n
        self.e.set_dofs_force_range(-force, force, np.arange(len(HOME)))

    def step(self, k: int = 1, hook=None) -> None:
        self.world.step(k, hook)

    def ik(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        q, err = self.e.inverse_kinematics(link=self.gripper, pos=pos, quat=quat, dofs_idx_local=ARM,
                                           init_qpos=self.e.get_qpos(), return_error=True)
        e = err.cpu().numpy().reshape(self.n, -1)
        self.ik_err_max = np.maximum(self.ik_err_max, np.linalg.norm(e[:, :3], axis=1))
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

    def go_home(self, steps: int = 120) -> None:
        self.move_joints(ARM, np.tile(self.home[:7], (self.n, 1)), steps)

    def down_quat_toward(self, axis_yaw: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        """目標の位置で指の開閉軸を axis_yaw に合わせる下向きの姿勢。±180° の 2 つの枝のうち、
        腕全体が台座まわりに回る分（肩の回転）を差し引くと手首の回転が少なくて済むほうを選ぶ。"""
        g, gq = self.gripper_pose()
        ang = lambda p: np.arctan2(p[:, 1] - self.base_xy[1], p[:, 0] - self.base_xy[0])  # noqa: E731
        hint = wrist_psi(gq) + wrap_pi(ang(np.asarray(target_xy)) - ang(g))
        return down_quat_branch(axis_yaw, hint)

    def gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.gripper.get_pos().cpu().numpy(), self.gripper.get_quat().cpu().numpy()

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
    """上限つきの吸着。吸盤が本に接した時点で密着し、その位置から 5 mm 以上ずれたら外れる。
    対象の本は attach で差し替える（1 回に 1 冊）。"""

    def __init__(self, robot: Robot):
        self.r = robot
        self.book = None
        self.fmax = 0.0
        n = robot.n
        self.on = np.zeros(n, bool)
        self.sealed = np.zeros(n, bool)
        self.anchor_local = np.zeros((n, 3))
        self.ever_sealed = np.zeros(n, bool)

    def attach(self, book, max_force_n: float) -> None:
        self.book, self.fmax = book, float(max_force_n)
        self.on[:] = False
        self.sealed[:] = False
        self.ever_sealed[:] = False

    def apply(self) -> None:
        if self.book is None:
            raise RuntimeError("吸着の対象の本が決まっていない（attach を先に呼ぶ）")
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
