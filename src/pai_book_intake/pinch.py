"""背の側で挟む手（PinchHand）を付けたロボットと、その動き。spikes/trial8_pinch.py（試行8）から本体へ移したもの。

動き（技能）は「歩みを譲るジェネレータ」で書く（yield k = k 歩待つ）。run_generators で複数のロボットを同時に動かせる。
  g_pick        本の背と小口をゴムで挟んで浮かせ、爪を背の側から下に入れ、板と爪で挟んで持ち上げる（試行8 の 1〜8）
  g_place_flat  平らな面（査定台・廃棄箱の底）へ下ろし、爪を本の底から離してから抜き、台に下ろして放す（試行8 の 9）
  g_shelve      本棚の前で、手首を回して本を立て（背表紙を手前・天を上）、小口から差し込み、床に下ろして放し、
                手を横（手のひらの側）へ抜く（B4-3）
  g_drive       台車（ロボット B）を動かす

ロボット B は台車に載る（mobile=True）。台車は世界の x・y の直動と鉛直軸まわりの回転の 3 関節で表す（車輪は表さない）。

数値の根拠は試行8（spikes/trial8_pinch.py の定数のコメント・Obsidian の日報 2026-10-18・19）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .books import GRAVITY, BookSpec, grip_plan
from .geometry import Q_DOWN, quat_mul, quat_to_mat, quat_z
from .hand_design import CLAW_RECESS, PALM_MARGIN, PLATE_BASE_Z, PinchHand
from .tool import HOME, KP, KV, Robot, World, build_robot_xml

# ---- 試行8 の値（理由は spikes/trial8_pinch.py の同名の定数） ----
MARGIN_SPINE = 0.012
MARGIN_FORE = 0.012
TIP_CLEAR = 0.002
LIFT_GAP = 0.005
LIFT_GAP_STEPS = 60
MAX_LIFT = 0.010
SPINE_ANCHOR_N = 60.0
SPINE_PRESS = 0.001
SQUEEZE = 0.010
CLAW_PAST_CENTER = 0.010
CLAW_SHORT = 0.015
CLAW_STEPS = 150
CLAW_PRELOAD = 0.002
CLAW_FLOOR_CLEAR = 0.0005
CLAW_UNDER_CLEAR = 0.0015
SIDE_HOLD_RATIO = 0.2
PLATE_MIN_N = 2.0  # 板で押さえる力の下限。8 N では 250 g 以下の本で板が爪に勝ち、本がゴムの摩擦で押し下げられて
# 下からの支えが負になった（試行8 本試験 20261020T0134〜、book-03/05/07 が全部 0/16）
WITHDRAW_PLATE_N = 2.0
PLATE_PRESS = 0.003
PLACE_CLEAR = 0.0005
WITHDRAW_GAP = 0.0005
WITHDRAW_ITERS = 8
SET_DOWN_PRESS = -0.0005
WITHDRAW_STEPS = 60
LIFT_STEPS, LOWER_STEPS = 100, 100

GAINS = {  # name: (kp, kv, force)
    "finger_l_joint": (8000, 100, 60), "finger_r_joint": (8000, 100, 60), "finger_r_lift": (8000, 100, 60),
    "claw_joint": (3000, 100, 40), "plunger_joint": (3000, 150, 40),
    "plate_y_joint": (3000, 100, 40), "plate_z_joint": (3000, 100, 40),
}
BASE_JOINTS = ("base_x", "base_y", "base_yaw")
BASE_GAINS = (40000.0, 8000.0, 4000.0)  # 台車（位置で動かす。車輪の滑りは表さない）
ARM_FORCE = [87, 87, 87, 87, 12, 12, 12]

# 台車の見た目（当たらない）。腕の台座はこの上ではなく床の高さのまま（試行8 と同じ腕の高さで届く範囲を変えないため）
CHASSIS_XML = ('<body name="base" pos="0 0 0">'
               '<inertial mass="40" pos="0 0 0.02" diaginertia="2 2 2"/>'
               '<joint name="base_x" type="slide" axis="1 0 0" damping="200" armature="5"/>'
               '<joint name="base_y" type="slide" axis="0 1 0" damping="200" armature="5"/>'
               '<joint name="base_yaw" type="hinge" axis="0 0 1" damping="50" armature="2"/>'
               '<geom type="box" size="0.30 0.24 0.012" pos="-0.05 0 0.012" contype="0" conaffinity="0" rgba="0.2 0.25 0.3 1"/>')


def build_pinch_robot_xml(out_dir: Path, design: PinchHand, mobile: bool = False, palm_through_env: bool = False) -> Path:
    """PinchHand を付けた Franka の MJCF。mobile=True なら台車の 3 関節（base_x・base_y・base_yaw）を根に足す。

    palm_through_env=True: 手のひら（指の開閉のレール。奥行 ±145 mm）を見た目だけにして、何とも当たらないようにする。
      Genesis 1.4.3 は MJCF の物体と別の物体の間では contype/conaffinity を見ない（collider.py の is_local_collision_mask）ので、
      「本とだけ当たる」にはできない。contype=conaffinity=0 の形は見た目だけとして読み込まれる。
      本を立てて小口から差し込むと、レールが本の横で水平になり、本棚の後ろの壁と台に当たって差し込めない（20261020T032325）。
      この手の形のままでは避けられないので、通しで動かすための仮の扱い。当たった量は palm_shelf_intrusion で毎回測って結果に残す
      （物理的に正しくない動きとして数える）。直し方はユーザーと相談する（Obsidian 02_確認事項 B8）"""
    path = build_robot_xml(out_dir, design=design)
    if not mobile and not palm_through_env:
        return path
    xml = path.read_text()
    if palm_through_env:
        palm = '<geom name="palm" type="box"'
        if palm not in xml:
            raise RuntimeError("手のひらの geom が見つからない")
        xml = xml.replace(palm, '<geom name="palm" contype="0" conaffinity="0" type="box"', 1)
    if not mobile:
        out = path.with_name(path.stem + "_palmthrough.xml")
        out.write_text(xml)
        return out
    root = '<body name="link0" childclass="panda">'
    if root not in xml:
        raise RuntimeError("panda_nohand.xml の link0 が見つからない（Genesis の版が想定と違う）")
    xml = xml.replace(root, CHASSIS_XML + root, 1)
    xml = re.sub(r"</worldbody>", "</body></worldbody>", xml, count=1)
    out = path.with_name(path.stem + ("_mobile_palmthrough.xml" if palm_through_env else "_mobile.xml"))
    out.write_text(xml)
    return out


class PinchRobot(Robot):
    """PinchHand を付けたロボット。dof の番号は関節の名前から引く（台車があると腕の番号がずれるため）。"""

    def __init__(self, world: World, entity, name: str, design: PinchHand,
                 base_pose: tuple[float, float, float] | None = None, base_xy=(0.0, 0.0)):
        self.design = design
        dof = lambda n: int(entity.get_joint(n).dofs_idx_local[0])  # noqa: E731
        self.idx = {n: dof(n) for n in GAINS}
        self.arm = np.array([dof(f"joint{k}") for k in range(1, 8)])
        self.mobile = base_pose is not None
        self.base = np.array([dof(n) for n in BASE_JOINTS]) if self.mobile else None
        n_dofs = entity.n_dofs
        home = np.zeros(n_dofs)
        home[self.arm] = HOME[:7]
        home[self.idx["finger_l_joint"]] = design.finger_open_m
        home[self.idx["finger_r_joint"]] = design.finger_open_m
        if self.mobile:
            home[self.base] = base_pose
            base_xy = base_pose[:2]
        super().__init__(world, entity, name, home=home, base_xy=base_xy)
        self.kp, self.kv, self.force = np.zeros(n_dofs), np.zeros(n_dofs), np.zeros(n_dofs)
        self.kp[self.arm], self.kv[self.arm], self.force[self.arm] = KP[:7], KV[:7], ARM_FORCE
        for n, (kp, kv, f) in GAINS.items():
            i = self.idx[n]
            self.kp[i], self.kv[i], self.force[i] = kp, kv, f
        if self.mobile:
            self.kp[self.base], self.kv[self.base] = BASE_GAINS[0], BASE_GAINS[1]
            self.force[self.base] = [BASE_GAINS[2], BASE_GAINS[2], BASE_GAINS[2]]

    def setup(self, grip_limit_n=None) -> None:
        dofs = np.arange(len(self.home))
        self.e.set_dofs_kp(self.kp, dofs)
        self.e.set_dofs_kv(self.kv, dofs)
        self.e.set_dofs_force_range(-self.force, self.force, dofs)
        self.force_now = np.tile(self.force, (self.n, 1))  # いま効いている力の上限（ACT のデモの行動に記録する）
        self.e.set_qpos(self.target)
        self.e.control_dofs_position(self.target)

    def set_forces(self, **named: float) -> None:
        f = self.force.copy()
        for n, v in named.items():
            f[self.idx[n]] = v
        self.e.set_dofs_force_range(-f, f, np.arange(len(f)))
        self.force_now = np.tile(f, (self.n, 1))

    def set_forces_per_env(self, f: np.ndarray) -> None:
        """環境ごとに違う力の上限（[n, n_dofs]）。ACT の方策が出す力をそのまま効かせる。"""
        self.e.set_dofs_force_range(-f, f, np.arange(f.shape[1]))
        self.force_now = f.copy()

    def i(self, name: str) -> int:
        return self.idx[name]

    def q(self, name: str) -> np.ndarray:
        return self.e.get_qpos().cpu().numpy()[:, self.idx[name]]

    def ik(self, pos: np.ndarray, quat: np.ndarray, init_arm: np.ndarray | None = None) -> np.ndarray:
        """腕の IK。init_arm を渡すと、今の姿勢の代わりにそれを初期値にする（大きく姿勢を変えるとき。今の姿勢から
        少しずつ解くと、局所解に入って手の向きが 82° ずれたまま止まった）。"""
        init = self.e.get_qpos()
        if init_arm is not None:
            init = init.clone()
            init[:, self.arm] = init.new_tensor(np.broadcast_to(init_arm, (self.n, len(self.arm))).copy())
        q, err = self.e.inverse_kinematics(link=self.gripper, pos=pos, quat=quat, dofs_idx_local=self.arm,
                                           init_qpos=init, return_error=True)
        e = err.cpu().numpy().reshape(self.n, -1)
        self.ik_err_max = np.maximum(self.ik_err_max, np.linalg.norm(e[:, :3], axis=1))
        if e.shape[1] >= 6:
            self.ik_rot_err_max = np.maximum(getattr(self, "ik_rot_err_max", np.zeros(self.n)), np.linalg.norm(e[:, 3:6], axis=1))
        return q.cpu().numpy()[:, self.arm]

    def base_pose(self) -> np.ndarray:
        if not self.mobile:
            raise RuntimeError(f"{self.name} は台車に載っていない")
        return self.e.get_qpos().cpu().numpy()[:, self.base]


# ---------- 歩みを譲る動き ----------

def g_wait(k: int):
    yield k


def g_joints(r: PinchRobot, idx, goal: np.ndarray, steps: int):
    start = r.target[:, idx].copy()
    for k in range(1, steps + 1):
        r.target[:, idx] = start + (goal - start) * (k / steps)
        yield 1


def g_joints_smooth(r: PinchRobot, idx, goal: np.ndarray, steps: int):
    """関節空間で、最小ジャークの時間配分で動かす。"""
    start = r.target[:, idx].copy()
    for k in range(1, steps + 1):
        r.target[:, idx] = start + (goal - start) * min_jerk(k / steps)
        yield 1


def min_jerk(a: float) -> float:
    return 10 * a ** 3 - 15 * a ** 4 + 6 * a ** 5


def g_cart(r: PinchRobot, pos: np.ndarray, quat: np.ndarray, steps: int, n_way: int = 12, profile=lambda a: a):
    """手先を直線上の中継点に沿って動かす（中継点ごとに IK、その間は関節空間で補間）。"""
    p0 = r.gripper.get_pos().cpu().numpy()
    q0 = r.gripper.get_quat().cpu().numpy()
    q1 = np.where((np.sum(q0 * quat, axis=1) < 0)[:, None], -quat, quat)
    per = max(1, steps // n_way)
    prev = r.target[:, r.arm].copy()
    for w in range(1, n_way + 1):
        a = profile(w / n_way)
        qw = q0 + (q1 - q0) * a
        qw /= np.linalg.norm(qw, axis=1, keepdims=True)
        goal = r.ik(p0 + (pos - p0) * a, qw)
        for k in range(1, per + 1):
            r.target[:, r.arm] = prev + (goal - prev) * (k / per)
            yield 1
        prev = goal


def g_cart_chain(r: PinchRobot, pos: np.ndarray, quat: np.ndarray, steps: int, n_way: int = 12, profile=lambda a: a):
    """g_cart と同じく直線上の中継点を通るが、各中継点の IK を「前の中継点の解」から解く（今の実際の姿勢からは解かない）。
    手を水平に向けた姿勢で、今の姿勢から解くと中継点ごとに解の枝が飛び、手が 165 mm 下がった（20261020T033732）。"""
    p0 = r.gripper.get_pos().cpu().numpy()
    q0 = r.gripper.get_quat().cpu().numpy()
    q1 = np.where((np.sum(q0 * quat, axis=1) < 0)[:, None], -quat, quat)
    per = max(1, steps // n_way)
    prev = r.target[:, r.arm].copy()
    for w in range(1, n_way + 1):
        a = profile(w / n_way)
        qw = q0 + (q1 - q0) * a
        qw /= np.linalg.norm(qw, axis=1, keepdims=True)
        goal = r.ik(p0 + (pos - p0) * a, qw, init_arm=prev)
        for k in range(1, per + 1):
            r.target[:, r.arm] = prev + (goal - prev) * (k / per)
            yield 1
        prev = goal


def g_correct(r: PinchRobot, pos: np.ndarray, quat: np.ndarray, out: dict, iters: int = 2, steps: int = 30):
    """重力によるたわみを、実測の誤差だけ目標をずらして打ち消す。"""
    aim = pos.copy()
    for _ in range(iters):
        yield 40
        err = pos - r.gripper.get_pos().cpu().numpy()
        norm = np.linalg.norm(err, axis=1, keepdims=True)
        err = np.where(norm > 0.02, err * (0.02 / np.maximum(norm, 1e-9)), err)
        aim = aim + err
        start = r.target[:, r.arm].copy()
        goal = r.ik(aim, quat, init_arm=start)  # 指令中の関節角から解く（解の枝を飛ばない）
        for k in range(1, steps + 1):
            r.target[:, r.arm] = start + (goal - start) * (k / steps)
            yield 1
    out["err"] = np.linalg.norm(pos - r.gripper.get_pos().cpu().numpy(), axis=1)
    out["aim"] = aim


def g_home(r: PinchRobot, steps: int = 120):
    yield from g_joints(r, r.arm, np.tile(r.home[r.arm], (r.n, 1)), steps)


def g_drive(r: PinchRobot, goal: tuple[float, float, float] | np.ndarray, max_speed: float = 0.5,
            max_yaw_rate: float = 1.0, dt: float = 0.01):
    """台車を目標の (x, y, 向き) へ動かす（最小ジャークで補間。腕の関節はそのまま＝持った本も台車と一緒に動く）。"""
    start = r.target[:, r.base].copy()
    goal = np.broadcast_to(np.asarray(goal, dtype=float), start.shape).copy()
    dyaw = (goal[:, 2] - start[:, 2] + math.pi) % (2 * math.pi) - math.pi
    goal[:, 2] = start[:, 2] + dyaw
    dist = np.linalg.norm(goal[:, :2] - start[:, :2], axis=1).max()
    t = max(dist / max_speed * 1.875, np.abs(dyaw).max() / max_yaw_rate * 1.875, 0.5)  # 最小ジャークの最高速は平均の 1.875 倍
    steps = int(math.ceil(t / dt))
    for k in range(1, steps + 1):
        r.target[:, r.base] = start + (goal - start) * min_jerk(k / steps)
        yield 1
    yield 30


def run_generators(world: World, gens: list) -> None:
    waits = [0] * len(gens)
    alive = [True] * len(gens)
    while any(alive):
        for i, g in enumerate(gens):
            while alive[i] and waits[i] == 0:
                try:
                    waits[i] = int(next(g))
                except StopIteration:
                    alive[i] = False
        if any(alive):
            world.step(1)
            waits = [max(0, w - 1) for w in waits]


# ---------- 姿勢と測り方 ----------

def book_frame(body) -> tuple[np.ndarray, np.ndarray]:
    return body.get_pos().cpu().numpy(), quat_to_mat(body.get_quat().cpu().numpy())


def to_book(p_world, pos, mat):
    return np.einsum("nji,nj->ni", mat, p_world - pos)


def mats_to_quat(m: np.ndarray) -> np.ndarray:
    """回転行列 (n, 3, 3) → quat (n, 4)（w, x, y, z）。対角の最大の成分から求める（Shepperd の方法）。
    w を先に求めて x・y・z の符号を非対角の差から決める方法は、180° 近い回転で符号が崩れる（本を立てる姿勢が 174° ずれた）。"""
    m = np.asarray(m, dtype=float)
    out = np.zeros((len(m), 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(len(m)):
        a = m[i]
        k = int(np.argmax([tr[i], a[0, 0], a[1, 1], a[2, 2]]))
        if k == 0:
            r = np.sqrt(1 + tr[i]) * 2
            q = [r / 4, (a[2, 1] - a[1, 2]) / r, (a[0, 2] - a[2, 0]) / r, (a[1, 0] - a[0, 1]) / r]
        elif k == 1:
            r = np.sqrt(1 + a[0, 0] - a[1, 1] - a[2, 2]) * 2
            q = [(a[2, 1] - a[1, 2]) / r, r / 4, (a[0, 1] + a[1, 0]) / r, (a[0, 2] + a[2, 0]) / r]
        elif k == 2:
            r = np.sqrt(1 + a[1, 1] - a[0, 0] - a[2, 2]) * 2
            q = [(a[0, 2] - a[2, 0]) / r, (a[0, 1] + a[1, 0]) / r, r / 4, (a[1, 2] + a[2, 1]) / r]
        else:
            r = np.sqrt(1 + a[2, 2] - a[0, 0] - a[1, 1]) * 2
            q = [(a[1, 0] - a[0, 1]) / r, (a[0, 2] + a[2, 0]) / r, (a[1, 2] + a[2, 1]) / r, r / 4]
        out[i] = q
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def lip_side_quat(e: np.ndarray) -> np.ndarray:
    """真下向きで、右の指（ゴムと爪）が水平方向 e の側に来る姿勢（gripper の +y = -e）。"""
    gy = -e
    psi = np.arctan2(gy[:, 0], -gy[:, 1])
    return quat_mul(quat_z(psi), np.broadcast_to(Q_DOWN, psi.shape + (4,)))


def claw_points(r: PinchRobot) -> list[np.ndarray]:
    lk = r.e.get_link("claw")
    p = lk.get_pos().cpu().numpy()
    m = quat_to_mat(lk.get_quat().cpu().numpy())
    return [p + m @ np.array(r.design.claw_probe_local(s)) for s in (-1, +1)]


def claw_top_z(r: PinchRobot) -> np.ndarray:
    """爪の上面の世界 z（手は真下向き）。"""
    d = r.design
    return r.e.get_link("claw").get_pos().cpu().numpy()[:, 2] - (d.finger_len_m - d.claw_t_m)


def book_lowest_z(book, size) -> np.ndarray:
    pb, mb = book_frame(book)
    w, h, t = size
    corners = np.array([[sx * w / 2, sy * h / 2, sz * t / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (pb[:, None, 2] + np.einsum("nj,kj->nk", mb[:, 2, :], corners)).min(axis=1)


def claw_path_points(spec: BookSpec, d: PinchHand, reach: float) -> np.ndarray:
    w = spec.width_m
    xs = np.linspace(w / 2, w / 2 - reach, 8)
    ys = np.array([-d.claw_x_out, -d.claw_x_in, d.claw_x_in, d.claw_x_out])
    return np.array([[x, y] for x in xs for y in ys])


def finger_footprint_points(spec: BookSpec, d: PinchHand) -> np.ndarray:
    w = spec.width_m
    ys = np.linspace(-d.span_m / 2, d.span_m / 2, 5)
    xs = (list(np.linspace(w / 2, w / 2 + MARGIN_SPINE + 0.012 + d.claw_body_len_m, 8))
          + list(np.linspace(-w / 2, -(w / 2 + MARGIN_FORE + d.wall_t_m), 3)))
    return np.array([[x, y] for x in xs for y in ys])


def claw_rear_points(spec: BookSpec, d: PinchHand) -> np.ndarray:
    w = spec.width_m
    xs = np.linspace(w / 2, w / 2 + MARGIN_SPINE + 0.012 + d.claw_body_len_m, 8)
    ys = np.array([-d.claw_x_out, -d.claw_x_in, d.claw_x_in, d.claw_x_out])
    return np.array([[x, y] for x in xs for y in ys])


def bottom_z_over(book, size, pts: np.ndarray) -> np.ndarray:
    pb, mb = book_frame(book)
    loc = np.concatenate([pts, np.full((len(pts), 1), -size[2] / 2)], axis=1)
    return (pb[:, None, 2] + np.einsum("nj,kj->nk", mb[:, 2, :], loc)).min(axis=1)


def top_z_under(lower, size_lower, book, size, pts: np.ndarray) -> np.ndarray:
    """取る本の座標の点 pts の真下で、下の本の上面の世界 z の最大（環境ごと）。"""
    pb, mb = book_frame(book)
    loc = np.concatenate([pts, np.full((len(pts), 1), -size[2] / 2)], axis=1)
    wp = pb[:, None, :] + np.einsum("nij,kj->nki", mb, loc)
    pl, ml = book_frame(lower)
    nrm = ml[:, :, 2]
    c = pl + nrm * size_lower[2] / 2 * np.sign(nrm[:, 2:3])
    nz = np.where(np.abs(nrm[:, 2]) < 1e-6, 1e-6, nrm[:, 2])
    z = c[:, None, 2] - (nrm[:, None, 0] * (wp[..., 0] - c[:, None, 0])
                         + nrm[:, None, 1] * (wp[..., 1] - c[:, None, 1])) / nz[:, None]
    return z.max(axis=1)


def touching(r: PinchRobot, book) -> np.ndarray:
    ci = r.e.get_contacts(with_entity=book)
    return ci["valid_mask"].cpu().numpy().any(axis=1)


def robot_contacts_env0(r: PinchRobot) -> list[str]:
    """環境 0 で、ロボットのリンクが触れている相手（"自分のリンク>相手のリンク"）。原因を調べるための記録。"""
    solver = r.e._solver
    cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
    la, lb = cd["link_a"].cpu().numpy(), cd["link_b"].cpu().numpy()
    if la.ndim == 2:
        la, lb = la[0], lb[0]
    names = {}
    for ent in solver.entities:
        for lk in ent.links:
            names[lk.idx] = f"{ent.idx}:{lk.name}"
    mine = set(range(r.e.link_start, r.e.link_end))
    out = set()
    for a, b in zip(la, lb):
        if a < 0 or b < 0:
            continue
        if a in mine and b not in mine:
            out.add(f"{names[a].split(':')[1]}>{names[b]}")
        elif b in mine and a not in mine:
            out.add(f"{names[b].split(':')[1]}>{names[a]}")
    return sorted(out)


def palm_shelf_intrusion(r: PinchRobot, shelf, keeper) -> np.ndarray:
    """手のひら（箱）が本棚の後ろの壁・指の範囲（棚の座標: y > 指の先, z < 指の軸＋ハブ）にどれだけ入っているか（m、環境ごと）。
    手のひらの 8 つの角を棚の座標に写し、壁・指の範囲の箱との重なりの深さ（y と z の小さい方）を返す。"""
    d = r.design
    g = r.gripper.get_pos().cpu().numpy()
    m = quat_to_mat(r.gripper.get_quat().cpu().numpy())
    hx, hy, hz = d.span_m / 2, d.finger_open_m + PALM_MARGIN, 0.02
    corners = np.array([[sx * hx, sy * hy, 0.02 + sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pw = g[:, None, :] + np.einsum("nij,kj->nki", m, corners)
    ps = shelf.to_shelf(pw)
    lo = np.array([0.0, keeper.piv_y - 0.064, 0.0])  # 指の先から奥・軸の高さまで（後ろの壁を含む）
    hi = np.array([keeper.width, keeper.depth, keeper.piv_z + 0.0075])
    a_lo, a_hi = ps.min(axis=1), ps.max(axis=1)
    ov = np.clip(np.minimum(a_hi, hi) - np.maximum(a_lo, lo), 0.0, None)
    return np.where((ov > 0).all(axis=1), ov[:, 1:].min(axis=1), 0.0)


# ---------- 持っている本の状態 ----------

@dataclass
class Grip:
    """g_pick が作り、g_place_flat・g_shelve が使う。"""

    book: object
    spec: BookSpec
    quat: np.ndarray  # 持ち上げたときの手の姿勢（真下向き）
    e: np.ndarray  # 背の向き（世界・水平）
    path: np.ndarray  # 爪が通る範囲（本の座標の点）
    f_sq: float
    f_hold: float
    f_plate: float
    ok: np.ndarray  # 持てている環境
    rec: dict = field(default_factory=dict)


def squeeze_forces(spec: BookSpec, d: PinchHand) -> tuple[float, float, float]:
    """(挟む力, 爪に載せた後の挟む力, 板の力)。A3 と試行8 の値。"""
    w_est = spec.weight_estimated_kg * GRAVITY
    f_sq = float(np.clip(w_est / (2 * d.rubber_mu) * 2.0, 8.0, 40.0))
    f_hold = float(max(2.0, SIDE_HOLD_RATIO * w_est))
    f_plate = float(max(PLATE_MIN_N, 0.75 * w_est))
    return f_sq, f_hold, f_plate


def g_pick(r: PinchRobot, book, spec: BookSpec, out: dict, *, surface_z: float | None = None,
           lower: list[tuple[object, BookSpec]] = (), carry_bottom_z: float = 0.25,
           plate_mode: str = "ii", claw_mode: str = "center", grip_off: float = 0.0):
    """本を持ち上げ、本の底が carry_bottom_z になるまで上げる。out["grip"] に Grip を入れる。
    下に本がある（山の上）なら lower に (本, 寸法) を下から順に渡す。無ければ surface_z が載せている面の高さ。
    grip_off: 手の中心を本の天（+y）の側へずらす量（A1 は天地の中央。本棚へ立てて入れるときだけずらす。shelve_grip_offset）"""
    d = r.design
    n = r.n
    rec: dict = {}
    w, h, t = spec.size_m
    f_sq, f_hold, f_plate = squeeze_forces(spec, d)
    FL, FR, LI, CL, PY, PZ = (r.i(k) for k in ("finger_l_joint", "finger_r_joint", "finger_r_lift", "claw_joint",
                                                "plate_y_joint", "plate_z_joint"))
    fo = d.finger_open_m
    t0 = r.world.sim_time

    p0, m0 = book_frame(book)
    e = m0[:, :, 0].copy()  # 背の向き（本の +x）
    e[:, 2] = 0.0
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    quat = lip_side_quat(e)
    z_s = p0[:, 2] - t / 2
    q_r0 = np.full(n, w / 2 + MARGIN_SPINE)
    q_l0 = np.full(n, w / 2 + MARGIN_FORE)
    rec["fits"] = (q_r0 <= fo) & (q_l0 <= fo)
    fp = finger_footprint_points(spec, d) + np.array([0.0, grip_off])
    if lower:
        under_f = np.max(np.stack([top_z_under(b, s.size_m, book, spec.size_m, fp) for b, s in lower]), axis=0)
    else:
        under_f = z_s
    palm = p0.copy()
    palm[:, :2] += m0[:, :2, 1] * grip_off  # 本の +y（天）の向きへ
    palm[:, 2] = np.maximum(z_s, under_f) + TIP_CLEAR + d.fingertip_z
    r.set_forces(finger_l_joint=f_sq, finger_r_joint=SPINE_ANCHOR_N)
    for k, v in ((FL, q_l0), (FR, q_r0)):
        r.target[:, k] = v
    for k in (LI, CL, PY, PZ, r.i("plunger_joint")):
        r.target[:, k] = 0.0
    above = palm.copy()
    above[:, 2] += 0.12
    yield from g_cart(r, above, quat, 150)
    yield from g_cart(r, palm, quat, 100, n_way=6)
    cor: dict = {}
    yield from g_correct(r, palm, quat, cor, iters=4)
    rec["pos_err_mm"] = cor["err"] * 1000
    aim = cor["aim"]

    # 挟む（背の側の指は位置の基準、小口の側の壁で力を出す）
    g_now = r.gripper.get_pos().cpu().numpy()
    pb = book.get_pos().cpu().numpy()
    s_book = np.einsum("ni,ni->n", pb - g_now, -e)
    yield from g_joints(r, FR, np.clip(w / 2 - s_book - SPINE_PRESS, 0, fo), 40)
    yield from g_joints(r, FL, np.clip(s_book + w / 2 - SQUEEZE, 0, fo), 50)
    yield 30

    # 浮かせる（爪は面の上に残す）
    reach = w / 2 + CLAW_PAST_CENTER if claw_mode == "center" else CLAW_SHORT
    path = claw_path_points(spec, d, reach) + np.array([0.0, grip_off])
    path_floor = np.concatenate([path, claw_rear_points(spec, d) + np.array([0.0, grip_off])])
    if lower:
        floor_fn = lambda: np.max(np.stack([top_z_under(b, s.size_m, book, spec.size_m, path_floor) for b, s in lower]), axis=0)  # noqa: E731
    else:
        sz = z_s if surface_z is None else np.full(n, surface_z)
        floor_fn = lambda: sz  # noqa: E731
    gap_fn = lambda: bottom_z_over(book, spec.size_m, path) - floor_fn()  # noqa: E731
    li = LI
    gz0 = r.gripper.get_pos().cpu().numpy()[:, 2]
    l0 = r.target[:, li].copy()
    tgt = aim.copy()
    need = np.full(n, LIFT_GAP)
    for it in range(8):
        tgt[:, 2] += np.clip(need, 0.0, 0.01)
        goal = r.ik(tgt, quat)
        start = r.target[:, r.arm].copy()
        n_st = LIFT_GAP_STEPS if it == 0 else LIFT_GAP_STEPS // 2
        for k in range(1, n_st + 31):
            if k <= n_st:
                r.target[:, r.arm] = start + (goal - start) * (k / n_st)
            r.target[:, li] = l0 + (r.gripper.get_pos().cpu().numpy()[:, 2] - gz0)
            yield 1
        bottom = book.get_pos().cpu().numpy()[:, 2] - t / 2
        need = np.minimum(LIFT_GAP - gap_fn(), MAX_LIFT - (bottom - z_s))
        if (need < 0.0005).all():
            break
    pb1, mb1 = book_frame(book)
    rec["lift_gap_mm"] = (gap_fn()) * 1000
    rec["lift_tilt_deg"] = np.degrees(np.arccos(np.clip(np.abs(mb1[:, 2, 2]), 0, 1)))

    # 爪の高さを合わせてから出す
    low = bottom_z_over(book, spec.size_m, path)
    floor = floor_fn()
    down = claw_top_z(r) - (low - CLAW_UNDER_CLEAR)
    floor_room = (claw_top_z(r) - d.claw_t_m) - (floor + CLAW_FLOOR_CLEAR)
    down = np.minimum(down, floor_room)
    yield from g_joints(r, LI, r.target[:, LI] + down, 20)
    yield 10
    ext = min(reach + CLAW_RECESS + SQUEEZE, d.claw_stroke_m)
    yield from g_joints(r, CL, np.full(n, ext / 2), CLAW_STEPS // 2)
    yield from g_joints(r, CL, np.full(n, ext), CLAW_STEPS // 2)
    yield 10
    pb2, mb2 = book_frame(book)
    rec["claw_shift_mm"] = np.abs(np.einsum("ni,ni->n", (pb2 - pb1)[:, :2], e[:, :2])) * 1000
    under = np.ones(n, bool)
    for p in claw_points(r):
        loc = to_book(p, pb2, mb2)
        under &= (np.abs(loc[:, 0]) < w / 2 - 0.002) & (loc[:, 2] < -t / 2 + 0.0005) & (np.abs(loc[:, 1]) < h / 2)
    rec["inserted"] = under

    # 爪に載せる（先にゴムを緩める）
    r.set_forces(finger_l_joint=f_hold, finger_r_joint=SPINE_ANCHOR_N)
    up_by = np.clip(bottom_z_over(book, spec.size_m, path) - claw_top_z(r) + CLAW_PRELOAD, 0.0, 0.012)
    yield from g_joints(r, LI, r.target[:, LI] - up_by, 40)
    yield 15

    # 板で押さえる
    q_r_now = r.q("finger_r_joint")
    g_now = r.gripper.get_pos().cpu().numpy()
    pb, _ = book_frame(book)
    s_book = np.einsum("ni,ni->n", pb - g_now, -e)
    y_plate = -q_r_now + 0.002 + d.plate_len_m / 2 if plate_mode == "i" else s_book
    yield from g_joints(r, PY, y_plate, 30)
    top_gz = np.einsum("ni,ni->n", pb - g_now, r.approach_axis()) - t / 2
    r.set_forces(plate_z_joint=f_plate, finger_l_joint=f_hold, finger_r_joint=SPINE_ANCHOR_N)
    yield from g_joints(r, PZ, np.clip(top_gz - (PLATE_BASE_Z + d.plate_t_m) + PLATE_PRESS, 0, d.plate_stroke_m), 60)
    yield 30

    # 持ち上げる
    g_now = r.gripper.get_pos().cpu().numpy()
    pb = book.get_pos().cpu().numpy()
    up = g_now.copy()
    up[:, 2] = carry_bottom_z + (g_now[:, 2] - (pb[:, 2] - t / 2))
    yield from g_cart(r, up, quat, LIFT_STEPS, n_way=10, profile=min_jerk)
    yield 20
    pb = book.get_pos().cpu().numpy()
    lifted = (pb[:, 2] - p0[:, 2] > 0.03) & touching(r, book)
    rec["lifted"] = lifted
    rec["pick_s"] = r.world.sim_time - t0
    out["grip"] = Grip(book, spec, quat, e, path, f_sq, f_hold, f_plate, ok=lifted & under, rec=rec)


def g_move_book(r: PinchRobot, grip: Grip, center_xy: np.ndarray, steps: int = 140, quat: np.ndarray | None = None,
                bottom_z: float | None = None):
    """持った本を、本の中心が center_xy の真上に来るまで運ぶ（姿勢は quat、省略時は持ったまま）。
    高さは今のまま。bottom_z を渡すと、本の底がその高さになるようにする。"""
    pb = grip.book.get_pos().cpu().numpy()
    g_now = r.gripper.get_pos().cpu().numpy()
    q_now = grip.quat if quat is None else quat
    m_now = quat_to_mat(r.gripper.get_quat().cpu().numpy())
    m_new = quat_to_mat(q_now)
    off_local = np.einsum("nji,nj->ni", m_now, pb - g_now)  # 手の座標での本の中心
    off_new = np.einsum("nij,nj->ni", m_new, off_local)
    goal = g_now.copy()
    goal[:, :2] = np.asarray(center_xy)[:, :2] - off_new[:, :2]
    if bottom_z is not None:
        goal[:, 2] += bottom_z - (pb[:, 2] - grip.spec.thickness_m / 2)
    yield from g_cart(r, goal, q_now, steps, n_way=24, profile=min_jerk)
    yield 20


def g_move_book_joint(r: PinchRobot, grip: Grip, center_xy: np.ndarray, quat: np.ndarray, bottom_z,
                      steps: int = 300, out: dict | None = None):
    """g_move_book と同じ行き先へ、関節空間で動かす。行き先の関節角は待機姿勢を初期値にして解く。
    手の向きを大きく変える移動を手先の直線で補間すると、中継点ごとに IK の解の枝が飛んで本を落とした（20261023T004512）。"""
    pb = grip.book.get_pos().cpu().numpy()
    g_now = r.gripper.get_pos().cpu().numpy()
    m_now = quat_to_mat(r.gripper.get_quat().cpu().numpy())
    off_local = np.einsum("nji,nj->ni", m_now, pb - g_now)  # 手の座標での本の中心
    off_new = np.einsum("nij,nj->ni", quat_to_mat(quat), off_local)
    goal = np.empty_like(g_now)
    goal[:, :2] = np.asarray(center_xy)[:, :2] - off_new[:, :2]
    goal[:, 2] = bottom_z + grip.spec.thickness_m / 2 - off_new[:, 2]
    q_goal = r.ik(goal, quat, init_arm=r.home[r.arm])
    yield from g_joints_smooth(r, r.arm, q_goal, steps)
    yield 40
    if out is not None:
        mg = quat_to_mat(r.gripper.get_quat().cpu().numpy())
        mw = quat_to_mat(quat)
        out["goal_world"] = goal
        out["book_off_world_mm"] = off_new * 1000
        out["before_correct_err_mm"] = (r.gripper.get_pos().cpu().numpy() - goal) * 1000
        out["rot_err_deg"] = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", mg, mw) - 1) / 2, -1, 1)))
        out["ik_rot_err_max"] = getattr(r, "ik_rot_err_max", None)
    cor: dict = {}
    yield from g_correct(r, goal, quat, cor, iters=2)
    # 本の中心で合わせる（手の中で本が少しずれているぶんと、腕のたわみを、本の実測で打ち消す）
    aim = cor["aim"].copy()
    for _ in range(2):
        err = np.zeros_like(aim)
        err[:, :2] = np.asarray(center_xy)[:, :2] - grip.book.get_pos().cpu().numpy()[:, :2]
        if not (np.linalg.norm(err, axis=1) > 0.002).any():
            break
        aim += np.clip(err, -0.02, 0.02)
        start = r.target[:, r.arm].copy()
        q_aim = r.ik(aim, quat, init_arm=start)
        yield from g_joints(r, r.arm, q_aim, 30)
        yield 40
    if out is not None:
        out["book_xy_err_mm"] = (grip.book.get_pos().cpu().numpy()[:, :2] - np.asarray(center_xy)[:, :2]) * 1000
        out["move_err_mm"] = cor["err"] * 1000
        out["held_after_move"] = touching(r, grip.book)


def g_place_flat(r: PinchRobot, grip: Grip, surface_z: float, out: dict, release_open: float | None = None):
    """手の真下の平らな面（高さ surface_z）へ下ろして放す（試行8 の 9）。本の xy は今の位置のまま。
    release_open: 放すときに指を今の位置から開く量（省略時は全開）。本棚では全開にすると小口の側の壁が奥へ約 9 cm 動き、
    本棚の指を跳ね上げた（20261023T012135）"""
    d = r.design
    n = r.n
    book, spec = grip.book, grip.spec
    FL, FR, LI, CL, PZ = (r.i(k) for k in ("finger_l_joint", "finger_r_joint", "finger_r_lift", "claw_joint",
                                            "plate_z_joint"))
    quat = grip.quat
    rec = out.setdefault("place", {})
    pw_start = book.get_pos().cpu().numpy()
    g_now = r.gripper.get_pos().cpu().numpy()
    lift_q = r.q("finger_r_lift")
    place = g_now.copy()
    place[:, 2] = surface_z + PLACE_CLEAR + d.fingertip_z + lift_q
    yield from g_cart(r, place, quat, LOWER_STEPS, n_way=8, profile=min_jerk)
    yield 30
    pw0 = book.get_pos().cpu().numpy()
    r.set_forces(plate_z_joint=WITHDRAW_PLATE_N, finger_l_joint=grip.f_sq, finger_r_joint=SPINE_ANCHOR_N)
    yield 20
    # 爪を本の底から離してから抜く（試行8 の本試験で、爪が本の底にめり込んで抜けなかった）
    rise = np.zeros(n)
    for _ in range(WITHDRAW_ITERS):
        gap = bottom_z_over(book, spec.size_m, grip.path) - claw_top_z(r)
        need = np.clip(WITHDRAW_GAP - gap, 0.0, 0.002)
        if not (need > 0.0001).any():
            break
        rise += need
        g_up = place.copy()
        g_up[:, 2] += rise
        yield from g_cart(r, g_up, quat, 15, n_way=2)
        room = claw_top_z(r) - d.claw_t_m - surface_z
        yield from g_joints(r, LI, r.q("finger_r_lift") + np.clip(room - 0.0002, 0.0, need + 0.001), 15)
        yield 5
    yield from g_joints(r, CL, np.zeros(n), WITHDRAW_STEPS)
    yield 20
    pw1 = book.get_pos().cpu().numpy()
    rec["withdraw_shift_mm"] = np.linalg.norm((pw1 - pw0)[:, :2], axis=1) * 1000
    # ゴムで持ったまま面に下ろす
    g_dn = place.copy()
    g_dn[:, 2] += rise
    for _ in range(3):
        gap = book_lowest_z(book, spec.size_m) - surface_z
        if not (gap > 0.0003).any():
            break
        g_dn[:, 2] -= np.clip(gap + SET_DOWN_PRESS, 0.0, None)
        yield from g_cart(r, g_dn, quat, 30, n_way=3)
        yield 15
    rec["set_down_gap_mm"] = (book_lowest_z(book, spec.size_m) - surface_z) * 1000
    yield from g_joints(r, PZ, np.zeros(n), 30)
    if release_open is None:
        opened = np.full((n, 2), d.finger_open_m)
    else:
        # 挟んでいる間の目標は本の内側（力の上限で押している）なので、実測の位置から開く
        q_now = np.stack([r.q("finger_l_joint"), r.q("finger_r_joint")], -1)
        opened = np.minimum(q_now + release_open, d.finger_open_m)
        r.target[:, [FL, FR]] = q_now
    yield from g_joints(r, [FL, FR], opened, 30)
    yield 10
    leave = r.gripper.get_pos().cpu().numpy()
    for k in range(1, 9):
        lk = leave.copy()
        lk[:, 2] += 0.12 * k / 8
        yield from g_cart(r, lk, quat, 25 if k <= 2 else 10, n_way=1)
    r.target[:, LI] = 0.0
    r.set_forces(finger_l_joint=GAINS["finger_l_joint"][2], plate_z_joint=GAINS["plate_z_joint"][2])
    yield 20
    rec["moved_from_start_mm"] = np.linalg.norm((book.get_pos().cpu().numpy() - pw_start)[:, :2], axis=1) * 1000


# ---------- 本棚へ立てて差し込む ----------

def shelve_grip_offset(spec: BookSpec, d: PinchHand, keeper, clearance: float = 0.003, edge: float = 0.005) -> float | None:
    """本を立てて本棚の床まで下ろしたとき、手のひら（指の開閉のレール）と小口の側の壁の指が、本棚の後ろの壁と指の軸（ハブ）の
    上を通るために要る「手の中心を本の天へずらす量」。手の指が本の天からはみ出さずにずらせなければ None（この手では入れられない）。

    立てた姿勢では、レールと壁の指が本の横で水平になり、奥行の向きに ±(指の最大の開き+2.5 cm) 伸びるので、後ろの壁（高さ 100 mm）と
    指の軸（約 114 mm）の上を越えなければならない（20261020T034731 で、腕・手が本棚に当たって差し込めなかった）。"""
    from .keeper import BASE_T, F_HUB_R

    w, h, t = spec.size_m
    need = (keeper.piv_z + F_HUB_R + clearance) - (BASE_T + h / 2 - d.span_m / 2)
    max_off = h / 2 - d.span_m / 2 - edge
    off = max(0.0, need)
    return off if off <= max_off else None



def standing_quat(r: PinchRobot, grip: Grip, shelf) -> tuple[np.ndarray, np.ndarray]:
    """本を立てて差し込むときの手の姿勢（世界の quat）と、手のひらが棚の x のどちら側に来るか（-1 / +1）。

    手の座標: y = 右の指（背）→ 左の壁（小口）の向き、x = 本の天地の向き、z = 手のひら → 指先。
    差し込む向き（小口から奥へ）= 棚の +y を手の +y に、本の天を上に向けるので、手の x は ±鉛直（今の持ち方で、
    本の天が手の x の + 側か − 側かで決まる）。z = x × y。手のひらは本から見て −z の側。"""
    mg = quat_to_mat(r.gripper.get_quat().cpu().numpy())
    mb = quat_to_mat(grip.book.get_quat().cpu().numpy())
    top_g = np.einsum("nij,ni->nj", mg, mb[:, :, 1])  # 本の天（+y）を手の座標で
    s = np.sign(top_g[:, 0])
    s[s == 0] = 1.0
    y_w = np.broadcast_to(shelf.dir_world([0.0, 1.0, 0.0]), (r.n, 3))
    x_w = np.zeros((r.n, 3))
    x_w[:, 2] = s
    z_w = np.cross(x_w, y_w)
    m = np.stack([x_w, y_w, z_w], axis=-1)  # 列が手の x・y・z
    palm_dir_shelf_x = -np.einsum("ni,i->n", z_w, shelf.dir_world([1.0, 0.0, 0.0]))
    return mats_to_quat(m), np.sign(palm_dir_shelf_x)


def g_shelve(r: PinchRobot, grip: Grip, shelf, keeper, slot: np.ndarray, quat_stand: np.ndarray, out: dict,
             push_back: float = 0.004, snap=None):
    """_g_shelve を動かしながら、手のひらが本棚に入り込んだ量の最大を毎歩測る（palm_through_env の仮の扱いを結果に残す）。"""
    worst = np.zeros(r.n)
    for v in _g_shelve(r, grip, shelf, keeper, slot, quat_stand, out, push_back, snap):
        worst = np.maximum(worst, palm_shelf_intrusion(r, shelf, keeper))
        yield v
    out["shelve"]["palm_shelf_intrusion_mm"] = worst * 1000


def _g_shelve(r: PinchRobot, grip: Grip, shelf, keeper, slot: np.ndarray, quat_stand: np.ndarray, out: dict,
              push_back: float = 0.004, snap=None):
    """本棚の前で本を立て、小口から差し込んで床に下ろし、放して手を横へ抜く（B4-3）。
    slot は環境ごとの棚の x の範囲 (n, 2)（本の厚みの向き）。手は台車で棚の正面にいること。"""
    d = r.design
    n = r.n
    book, spec = grip.book, grip.spec
    w, h, t = spec.size_m
    FL, FR, CL, PZ = (r.i(k) for k in ("finger_l_joint", "finger_r_joint", "claw_joint", "plate_z_joint"))
    rec = out.setdefault("shelve", {})
    user_snap = snap

    def snap(label: str) -> None:  # 各段階の本の位置（棚の座標）と、手に触れているか
        rec.setdefault("phase_book_shelf_mm", {})[label] = np.round(shelf.to_shelf(book.get_pos().cpu().numpy()) * 1000, 1)
        rec.setdefault("phase_touch", {})[label] = touching(r, book)
        if user_snap is not None:
            user_snap(label)

    t0 = r.world.sim_time
    # 手の座標での本の中心（持っている間は変わらない）
    pb = book.get_pos().cpu().numpy()
    g_now = r.gripper.get_pos().cpu().numpy()
    mg = quat_to_mat(r.gripper.get_quat().cpu().numpy())
    c_g = np.einsum("nji,nj->ni", mg, pb - g_now)
    m_st = quat_to_mat(quat_stand)
    rec["target_hand_z"] = m_st[:, :, 2]
    rec["target_hand_x"] = m_st[:, :, 0]
    if r.mobile:
        rec["base_pose_at_start"] = r.base_pose()
    off = np.einsum("nij,nj->ni", m_st, c_g)  # 立てた姿勢での、手から本の中心へのベクトル（世界）

    x_c = np.asarray(slot, dtype=float).mean(axis=1)

    def grip_pos_for(y: float, z: float) -> np.ndarray:
        c = np.stack([x_c, np.full(n, y), np.full(n, z)], -1)
        return shelf.to_world(c) - off

    from .keeper import BASE_T, WT  # 循環を避けるためここで読む

    y_final = keeper.depth - WT - d.wall_t_m - push_back - w / 2
    z_up = BASE_T + h / 2 + 0.025  # 差し込む高さ（本の底を床の 25 mm 上）。6 mm では腕のたわみで本の底が台の手前の角に当たった
    # 立てている間は、両側のゴムを強く挟んで重さを摩擦で受ける（本の重さが手の x の向きにかかるため）
    r.set_forces(finger_l_joint=grip.f_sq, finger_r_joint=SPINE_ANCHOR_N, plate_z_joint=grip.f_plate)
    yield 20
    # 1. 棚の手前の空中で立てる
    snap("s0 before standing")
    def traced(gen, label: str, every: int = 10):
        """環境 0 の、手の中の本のずれ（手の座標）・指の位置と力を every 歩ごとに残す（本が手から落ちる原因を探すため）。"""
        tr = rec.setdefault("trace_env0", {}).setdefault(label, [])
        k_step = 0
        for v in gen:
            k_step += 1
            if k_step % every == 0:
                q = r.e.get_qpos().cpu().numpy()[0]
                cf = r.e.get_dofs_control_force().cpu().numpy()[0]
                mg_k = quat_to_mat(r.gripper.get_quat().cpu().numpy())
                c_k = np.einsum("nji,nj->ni", mg_k, book.get_pos().cpu().numpy() - r.gripper.get_pos().cpu().numpy())
                tr.append({"k": k_step, "touch": bool(touching(r, book)[0]),
                           "slip_g_mm": [round(float(v_), 1) for v_ in (c_k[0] - c_g[0]) * 1000],
                           "q_fl_fr_li_mm": [round(float(q[i_]) * 1000, 1) for i_ in (FL, FR, r.i("finger_r_lift"))],
                           "f_fl_fr_pz": [round(float(cf[i_]), 1) for i_ in (FL, FR, PZ)],
                           "q_claw_plate_mm": [round(float(q[CL]) * 1000, 1), round(float(q[PZ]) * 1000, 1)],
                           "contacts": robot_contacts_env0(r)})
            yield v

    # 立てた姿勢の関節角は、待機姿勢を初期値にして解き、関節空間で動かす（手先の直線補間で少しずつ解くと局所解に入った）
    pre_pos = grip_pos_for(-w / 2 - 0.14, z_up + 0.02)
    q_pre = r.ik(pre_pos, quat_stand, init_arm=r.home[r.arm])
    yield from traced(g_joints_smooth(r, r.arm, q_pre, 300), "stand")
    yield 30
    cor: dict = {}
    yield from g_correct(r, pre_pos, quat_stand, cor, iters=2)
    snap("s1 stood up")
    mb = quat_to_mat(book.get_quat().cpu().numpy())
    rec["stood_up_tilt_deg"] = np.degrees(np.arccos(np.clip(mb[:, 2, 1], -1, 1)))  # 本の天（+y）と鉛直のなす角
    mg2 = quat_to_mat(r.gripper.get_quat().cpu().numpy())
    rec["hand_rot_err_deg"] = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", mg2, m_st) - 1) / 2, -1, 1)))
    c_g2 = np.einsum("nji,nj->ni", mg2, book.get_pos().cpu().numpy() - r.gripper.get_pos().cpu().numpy())
    rec["book_slip_in_hand_mm"] = np.linalg.norm(c_g2 - c_g, axis=1) * 1000
    rec["held_after_standing"] = touching(r, book)
    rec["ik_err_max_mm"] = r.ik_err_max * 1000
    # 2. 小口を棚の手前の縁へ
    p1 = grip_pos_for(-w / 2 - 0.01, z_up)
    yield from traced(g_cart_chain(r, p1, quat_stand, 80, n_way=8, profile=min_jerk), "to_front", every=5)
    snap("s1b at front edge")
    rec["front_edge_hand_err_mm"] = (r.gripper.get_pos().cpu().numpy() - p1) * 1000
    # 3. 奥へ差し込む（ゆっくり）
    p2 = grip_pos_for(y_final, z_up)
    yield from traced(g_cart_chain(r, p2, quat_stand, 220, n_way=22, profile=min_jerk), "insert", every=20)
    yield 20
    snap("s2 inserted")
    # 4. 床に下ろす（腕のたわみがあるので測りながら 3 回まで）
    floor_z = shelf.stand_h + BASE_T
    aim = p2.copy()
    for _ in range(3):
        gap = book_lowest_z(book, spec.size_m) - floor_z
        if not (gap > 0.0008).any():
            break
        aim[:, 2] -= np.clip(gap - 0.0003, 0.0, 0.02)
        yield from traced(g_cart_chain(r, aim, quat_stand, 40, n_way=4), "lower", every=10)
        yield 15
    rec["floor_gap_mm"] = (book_lowest_z(book, spec.size_m) - floor_z) * 1000
    snap("s3 on floor")
    # 5. 放す: 板を戻し、爪を引っ込め、挟む力を抜く。左の壁は奥の壁の手前にあるので開けない（開くと奥の壁に当たる）
    yield from g_joints(r, PZ, np.zeros(n), 30)
    yield from g_joints(r, CL, np.zeros(n), WITHDRAW_STEPS)
    r.set_forces(finger_l_joint=1.0, finger_r_joint=5.0)
    yield from g_joints(r, FR, r.target[:, FR] + 0.006, 20)  # 背の側のゴムを背から離す
    yield 30
    # 6. 手を手のひらの側（−z）へ抜く
    zg = m_st[:, :, 2]
    out_pos = r.gripper.get_pos().cpu().numpy() - zg * (d.fingertip_z + 0.03)
    yield from g_cart_chain(r, out_pos, quat_stand, 120, n_way=12, profile=min_jerk)
    back = out_pos + shelf.dir_world([0.0, -1.0, 0.0]) * 0.15
    yield from g_cart_chain(r, back, quat_stand, 80, n_way=8, profile=min_jerk)
    snap("s4 hand out")
    r.set_forces(finger_l_joint=GAINS["finger_l_joint"][2], finger_r_joint=GAINS["finger_r_joint"][2],
                 plate_z_joint=GAINS["plate_z_joint"][2])
    r.target[:, FL] = d.finger_open_m
    r.target[:, FR] = d.finger_open_m
    r.target[:, r.i("finger_r_lift")] = 0.0
    yield 60
    rec["shelve_s"] = r.world.sim_time - t0
