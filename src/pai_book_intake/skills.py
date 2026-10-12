"""ロボットの技能（スクリプト）。すべての環境で同じ手順を同時に進め、目標は環境ごとの配列で持つ。

flat_pick      平置きの本を、吸盤で浮かせてからグリッパで挟んで持ち上げる（試行2-6: 表面が滑らかな本で 16/16）
place_by_plunger  持った本を目標の位置・向きの真上まで運び、指を開いて吸盤だけで支え、
                  プランジャを伸ばして本を面へ下ろしてから吸着を切る（指先が本の底より 2 cm 下に出ているため、
                  指で挟んだまま下ろすと指が先に面に当たる）。面が高い位置にあれば（廃棄箱）その高さから落とす
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .books import SURFACES
from .geometry import down_quat_with_finger_axis, quat_to_mat, wrap_axis_angle, yaw_of
from .scene import BookBody
from .tool import CUP_TIP_EXTENDED_Z, FINGER_OPEN, FINGERS, FINGERTIP_Z, LIP_T, PLUNGER, PLUNGER_STROKE, Robot


def book_pose(body: BookBody) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(中心, 横幅の向きの yaw, 本の z 軸の鉛直成分)。"""
    q = body.entity.get_quat().cpu().numpy()
    return body.entity.get_pos().cpu().numpy(), yaw_of(q), quat_to_mat(q)[:, 2, 2]


@dataclass
class PickResult:
    held: np.ndarray
    cup_touch: np.ndarray
    sealed: np.ndarray
    raised_by_suction_m: np.ndarray
    z_gain_m: np.ndarray
    opening_m: np.ndarray

    def per_env(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def flat_pick(r: Robot, body: BookBody, above_dz: float = 0.12, lift_dz: float = 0.25, hold_steps: int = 30,
              suction_n: float | None = None) -> PickResult:
    spec, book, n = body.spec, body.entity, r.n
    r.suction.attach(book, SURFACES[spec.surface]["suction_max_n"] if suction_n is None else suction_n)
    r.target[:, FINGERS] = FINGER_OPEN
    r.target[:, PLUNGER] = PLUNGER_STROKE
    p0, yaw, _ = book_pose(body)
    quat = down_quat_with_finger_axis(yaw)  # 指の開閉軸を本の横幅の向きに合わせる
    book_top = p0[:, 2] + spec.thickness_m / 2

    above = p0.copy()
    above[:, 2] = book_top + CUP_TIP_EXTENDED_Z + above_dz
    r.move_arm(above, quat, 120)
    touch = p0.copy()
    touch[:, 2] = book_top + CUP_TIP_EXTENDED_Z - 0.012  # PD のたわみを見込んで押し付ける（試行2-6 と同じ）
    r.move_arm(touch, quat, 80)
    r.step(15)
    _, cup_touch = r.touching(book)

    r.suction.on[:] = True
    r.step(10)
    r.move_joints(PLUNGER, np.zeros(n), 80)  # 吸盤を引き込む → 本が指の間へ
    r.step(10)
    raised = book.get_pos().cpu().numpy()[:, 2] - p0[:, 2]
    r.move_joints(FINGERS, np.zeros((n, 2)), 60)  # 閉じる（本の横幅で止まる）
    r.step(20)

    lift = p0.copy()
    lift[:, 2] = book_top + CUP_TIP_EXTENDED_Z + lift_dz
    r.move_arm(lift, quat, 120)
    r.step(hold_steps)
    z_gain = book.get_pos().cpu().numpy()[:, 2] - p0[:, 2]
    opening = r.finger_opening()
    f_touch, _ = r.touching(book)
    held = (z_gain > 0.6 * lift_dz) & f_touch & (np.abs(opening - spec.width_m) < 0.02)
    return PickResult(held, cup_touch, r.suction.ever_sealed.copy(), raised, z_gain, opening)


@dataclass
class PlaceResult:
    pos_err_m: np.ndarray  # 下ろす直前の手先の位置の誤差
    ik_err_max_m: np.ndarray  # 運ぶ間の IK の残差の最大（大きければ解けていない＝腕が振り回された疑い）
    drop_gap_m: np.ndarray  # 吸盤で下ろし始める前の、本の底と面の隙間
    center: np.ndarray
    yaw: np.ndarray
    up_z: np.ndarray

    def per_env(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def place_by_plunger(r: Robot, body: BookBody, target_center: np.ndarray, target_axis_yaw: np.ndarray,
                     surface_top: np.ndarray, carry_z: float, settle_steps: int = 40) -> PlaceResult:
    """target_center は置いたあとの本の中心（x, y だけ使う）。surface_top は置く面の高さ（環境ごと）。"""
    spec, book, n = body.spec, body.entity, r.n
    surface_top = np.broadcast_to(np.asarray(surface_top, dtype=float), (n,))
    g, gq = r.gripper_pose()
    rg = quat_to_mat(gq)
    off_local = np.einsum("nji,nj->ni", rg, book.get_pos().cpu().numpy() - g)  # 手の中の本の中心（gripper 座標）
    # 本の横幅の向きと指の開閉軸（gripper の y 軸）のずれを測り、置くときの手首の向きで打ち消す
    finger_yaw = np.arctan2(rg[:, 1, 1], rg[:, 0, 1])
    _, b_yaw, _ = book_pose(body)
    delta = wrap_axis_angle(b_yaw - finger_yaw)
    axis = np.asarray(target_axis_yaw, dtype=float) - delta
    q_place = r.down_quat_toward(axis, target_center[:, :2])
    rp = quat_to_mat(q_place)
    off_world = np.einsum("nij,nj->ni", rp, off_local)
    hover = np.empty((n, 3))
    hover[:, :2] = target_center[:, :2] - off_world[:, :2]
    hover[:, 2] = surface_top + FINGERTIP_Z + LIP_T + 0.006  # 指先（右の爪を含む）を面の 6 mm 上に

    r.ik_err_max = np.zeros(n)
    up = g.copy()
    up[:, 2] = carry_z
    r.move_linear(up, gq, 60)
    above = hover.copy()
    above[:, 2] = carry_z
    r.move_linear(above, q_place, 200, n_way=16)
    r.move_linear(hover, q_place, 100)
    pos_err = r.correct(hover, q_place)
    ik_err = r.ik_err_max.copy()

    # 指を開いて吸盤だけで支える → プランジャで面まで下ろす → 吸着を切る
    r.move_joints(FINGERS, np.full((n, 2), FINGER_OPEN), 60)
    r.step(10)
    bottom = book.get_pos().cpu().numpy()[:, 2] - spec.thickness_m / 2
    gap = bottom - surface_top
    r.move_joints(PLUNGER, np.clip(gap + 0.004, 0.0, PLUNGER_STROKE), 80)
    r.step(10)
    r.suction.on[:] = False
    r.step(20)
    r.move_joints(PLUNGER, np.zeros(n), 40)
    leave = r.gripper.get_pos().cpu().numpy()
    leave[:, 2] = carry_z
    r.move_linear(leave, q_place, 80)
    r.step(settle_steps)
    c, yaw, up_z = book_pose(body)
    return PlaceResult(pos_err, ik_err, gap, c, yaw, up_z)
