"""姿勢計算（quat は w, x, y, z）。numpy だけに依存する（Genesis なしでテストできるように）。

spikes/trial2_common.py の同名関数を本体へ移したもの。spikes 側はそのまま残す。
"""

from __future__ import annotations

import math

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=float), -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_z(yaw) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=float)
    z = np.zeros_like(yaw)
    return np.stack([np.cos(yaw / 2), z, z, np.sin(yaw / 2)], axis=-1)


def quat_x(angle) -> np.ndarray:
    a = np.asarray(angle, dtype=float)
    z = np.zeros_like(a)
    return np.stack([np.cos(a / 2), np.sin(a / 2), z, z], axis=-1)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=float), -1, 0)
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


def down_quat_with_finger_axis(yaw_of_axis) -> np.ndarray:
    """真下向きで、指の開閉軸を水平方向 yaw_of_axis に合わせる姿勢。軸は ±180° 同値なので手首が回りすぎない側を選ぶ。"""
    psi = np.asarray(yaw_of_axis, dtype=float) + math.pi / 2
    psi = (psi + math.pi / 2) % math.pi - math.pi / 2
    return quat_mul(quat_z(psi), np.broadcast_to(Q_DOWN, psi.shape + (4,)))


def down_quat_branch(axis_yaw, psi_hint) -> np.ndarray:
    """真下向きで指の開閉軸を axis_yaw に合わせる姿勢のうち、手首の回転角 psi が psi_hint に近いほうの枝。
    （down_quat_with_finger_axis は世界の yaw で枝を選ぶので、回して置いたロボットでは手首が限界まで回ることがある）"""
    psi0 = np.asarray(axis_yaw, dtype=float) + math.pi / 2
    k = np.round((np.asarray(psi_hint, dtype=float) - psi0) / math.pi)
    psi = psi0 + k * math.pi
    return quat_mul(quat_z(psi), np.broadcast_to(Q_DOWN, psi.shape + (4,)))


def wrist_psi(q: np.ndarray) -> np.ndarray:
    """真下向きの gripper の、z 軸まわりの回転角 psi（down_quat_* の psi と同じ定義）。"""
    m = quat_to_mat(q)
    return np.arctan2(m[..., 1, 0], m[..., 0, 0])


def wrap_pi(a) -> np.ndarray:
    return (np.asarray(a, dtype=float) + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q: np.ndarray) -> np.ndarray:
    m = quat_to_mat(q)
    return np.arctan2(m[..., 1, 0], m[..., 0, 0])


def wrap_axis_angle(a) -> np.ndarray:
    """向きを持たない軸（±180° 同値）の角度差を [-90°, 90°) に畳む。"""
    a = np.asarray(a, dtype=float)
    return (a + math.pi / 2) % math.pi - math.pi / 2


class Frame2D:
    """床の上に置いた台座の座標系（原点と z 軸まわりの向き）。ロボット B のように回して置いた装置の
    「ローカルの位置」を世界座標に直すのに使う。"""

    def __init__(self, origin_xy: tuple[float, float], yaw: float):
        self.origin = np.array([origin_xy[0], origin_xy[1]], dtype=float)
        self.yaw = float(yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        self.rot = np.array([[c, -s], [s, c]])

    def to_world_xy(self, local_xy) -> np.ndarray:
        p = np.asarray(local_xy, dtype=float)
        return self.origin + p @ self.rot.T

    def to_world(self, local_xyz) -> np.ndarray:
        p = np.asarray(local_xyz, dtype=float)
        xy = self.to_world_xy(p[..., :2])
        return np.concatenate([xy, p[..., 2:3]], axis=-1)

    def to_local_xy(self, world_xy) -> np.ndarray:
        p = np.asarray(world_xy, dtype=float) - self.origin
        return p @ self.rot

    def yaw_to_world(self, local_yaw):
        return np.asarray(local_yaw, dtype=float) + self.yaw
