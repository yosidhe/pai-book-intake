"""ライン全体の配置（数値はすべてここに集める）と、受け渡し台への置き方の基準（置き方の標準）。

世界座標（単位 m）。床は z = 0。

  ロボット A（取り出し）  台座 (0, 0)、+x 向き。段ボール箱は A の正面 (0.55, 0)（試行2・3と同じ位置関係）
  受け渡し台（審査台）     A の左手（+y 側）。A が本を置き、真上のカメラでバーコードを読む
  ロボット B（仕分け）     台座 (0, 1.15)、-y 向き（台をはさんで A と向き合う）。
                           B から見て右手（世界 -x）に廃棄箱、左手（世界 +x）に本棚

受け渡し台はコンベアではなく固定の台にした（理由は scene.py の冒頭）。

置き方の標準（裏返し工程のため。試行2-7 の前提を引き継ぐ）
  - 表紙の向きは箱から取ったまま（裏返しは inspect-flip の仕事）
  - 本の横幅（width）の向きを、台の B 側の縁に直交させる（＝本の縦（天地）が縁と平行）
  - 本は B 側の縁から B の方へ overhang だけはみ出す。overhang = clamp(0.35 × 横幅, 30 mm, 50 mm)
      130 mm 幅 → 45.5 mm、105 mm（文庫）→ 36.75 mm、148 mm 以上 → 50 mm。
      重心は必ず台の上（横幅/2 − overhang ≥ 15 mm）。試行2-7 は 40〜60 mm を想定していた
  - 本の中心は台の中心線上（B から見て左右 0）
  - 許容差: はみ出し ±10 mm、向き ±8°（試行2-7 のばらつきと同じ幅）、左右 ±30 mm
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Frame2D, wrap_axis_angle

# ---------- ロボット ----------
ROBOT_A_BASE = (0.0, 0.0)
ROBOT_A_YAW = 0.0
ROBOT_B_BASE = (0.0, 1.15)
ROBOT_B_YAW = -math.pi / 2
FRAME_A = Frame2D(ROBOT_A_BASE, ROBOT_A_YAW)
FRAME_B = Frame2D(ROBOT_B_BASE, ROBOT_B_YAW)

# ---------- 段ボール箱（A の座標 = 世界座標） ----------
BOX_C = (0.55, 0.0)
BOX_IN = (0.42, 0.42)
BOX_WALL_H = 0.16
BOX_WALL_T = 0.01
BOX_FLOOR = 0.02
FLAT_STACK_C = (0.55, -0.04)  # 平積みの山の中心（立てた本は +y の壁に寄せる）

# ---------- 受け渡し台（B の座標で決める。B から見て正面） ----------
TABLE_TOP = 0.30
TABLE_EDGE_B = 0.58  # B の台座から台の B 側の縁まで（B のローカル x）
TABLE_DEPTH = 0.30  # 縁から奥（A 側）へ
TABLE_HALF_W = 0.20  # 左右の半幅
OVERHANG_RATIO = 0.35
OVERHANG_MIN = 0.030
OVERHANG_MAX = 0.050
TOL_OVERHANG = 0.010
TOL_YAW = math.radians(8)
TOL_LATERAL = 0.030
BARCODE_CAM_H = 0.30  # 台の上面からカメラまで（試行2-8 と同じ）

# ---------- 本棚（B の座標）: 上が開いた 1 段。本は平置きでスロットに並べる ----------
# 本の横幅を棚の奥行き方向（B のローカル y）に向け、天地を棚の長さ方向に並べる（背を B に向けて寝かせる）。
# 横幅の向きが B から見た放射方向に近いので、台から棚へ運ぶ間に手首をほとんど回さずに済む
SHELF_BOARD_TOP = 0.30
SHELF_DEPTH_C = 0.50  # B のローカル y（本棚の奥行きの中心）
SHELF_DEPTH = 0.26
SHELF_SLOT_X = (-0.18, 0.04, 0.26)  # スロットの中心（B のローカル x）。3 冊で満杯
SHELF_SLOT_HALF = 0.12  # スロットの半幅（天地 194 mm までの本が入る）
SHELF_BOOK_AXIS_LOCAL = math.pi / 2  # 棚に置く本の横幅の向き（B のローカル yaw）
SHELF_PANEL_H = 0.06  # 棚板から上の背板・側板の高さ（運ぶ本の底が引っかからない低さ）

# ---------- 廃棄箱（B の座標） ----------
BIN_C = (0.10, -0.50)
BIN_IN = (0.34, 0.30)
BIN_WALL_H = 0.18
BIN_FLOOR_T = 0.03  # 底は厚めに（薄いと落とした本がすり抜ける）

# 待機姿勢: B は台から離れた側（廃棄箱の上）へ肩を回しておく（A が台に置く間にぶつからないように）
B_PARK_JOINT1 = -1.0


def overhang_for(width_m: float) -> float:
    return float(min(OVERHANG_MAX, max(OVERHANG_MIN, OVERHANG_RATIO * width_m)))


@dataclass(frozen=True)
class HandoffPose:
    center_world: np.ndarray  # 本の中心（x, y, z）
    width_axis_yaw_world: float  # 本の横幅の向き（世界の yaw）
    overhang_m: float


def handoff_pose(width_m: float, thickness_m: float) -> HandoffPose:
    """置き方の標準に従った、本の中心の位置と向き。"""
    oh = overhang_for(width_m)
    x_local = TABLE_EDGE_B - oh + width_m / 2  # B 側の縁から oh だけ B の方へ出す
    xy = FRAME_B.to_world_xy((x_local, 0.0))
    z = TABLE_TOP + thickness_m / 2
    return HandoffPose(np.array([xy[0], xy[1], z]), float(FRAME_B.yaw_to_world(0.0)), oh)


@dataclass(frozen=True)
class PlacementCheck:
    overhang_m: float
    overhang_err_m: float
    yaw_err_rad: float
    lateral_err_m: float
    on_table: bool
    within_tolerance: bool

    def features(self) -> dict:
        return {"overhang_err_mm": abs(self.overhang_err_m) * 1000, "yaw_err_deg": abs(math.degrees(self.yaw_err_rad)),
                "lateral_err_mm": abs(self.lateral_err_m) * 1000, "on_table": self.on_table}


def check_placement(center_world, yaw_world: float, width_m: float, height_m: float, thickness_m: float,
                    up_z: float = 1.0) -> PlacementCheck:
    """置かれた本の実測の姿勢を、置き方の標準と比べる（1 環境ぶん）。up_z は本の z 軸の鉛直成分。"""
    c = np.asarray(center_world, dtype=float)
    local = FRAME_B.to_local_xy(c[:2])
    yaw_local = float(yaw_world) - FRAME_B.yaw
    yaw_err = float(wrap_axis_angle(yaw_local))  # 横幅の軸が B のローカル x に沿っているか
    # 縁と直交する向きの、本の中心から B 側の端までの距離（向きがずれると角が先に出る）
    half_extent = abs(math.cos(yaw_err)) * width_m / 2 + abs(math.sin(yaw_err)) * height_m / 2
    near_edge = local[0] - half_extent
    overhang = TABLE_EDGE_B - near_edge
    oh_std = overhang_for(width_m)
    on_table = bool(abs(c[2] - (TABLE_TOP + thickness_m / 2)) < 0.01 and abs(up_z) > 0.9
                    and TABLE_EDGE_B <= local[0] <= TABLE_EDGE_B + TABLE_DEPTH and abs(local[1]) <= TABLE_HALF_W)
    ok = (on_table and abs(overhang - oh_std) <= TOL_OVERHANG and abs(yaw_err) <= TOL_YAW
          and abs(local[1]) <= TOL_LATERAL)
    return PlacementCheck(float(overhang), float(overhang - oh_std), yaw_err, float(local[1]), on_table, bool(ok))


def shelf_slot_center(slot: int, thickness_m: float) -> np.ndarray:
    """本棚のスロットに平置きしたときの本の中心（世界）。本の横幅は棚の奥行き方向。"""
    xy = FRAME_B.to_world_xy((SHELF_SLOT_X[slot], SHELF_DEPTH_C))
    return np.array([xy[0], xy[1], SHELF_BOARD_TOP + thickness_m / 2])


def shelf_axis_yaw_world() -> float:
    return float(FRAME_B.yaw_to_world(SHELF_BOOK_AXIS_LOCAL))


def bin_axis_yaw_world() -> float:
    """廃棄箱へは、B から見た放射方向に横幅を向けて落とす（向きは問わないので手首を回さない向き）。"""
    return float(FRAME_B.yaw_to_world(math.atan2(BIN_C[1], BIN_C[0])))


def shelf_capacity() -> int:
    return len(SHELF_SLOT_X)


def bin_center_world() -> np.ndarray:
    xy = FRAME_B.to_world_xy(BIN_C)
    return np.array([xy[0], xy[1], 0.0])


def in_bin(center_world) -> bool:
    c = np.asarray(center_world, dtype=float)
    local = FRAME_B.to_local_xy(c[:2]) - np.array(BIN_C)
    return bool(abs(local[0]) <= BIN_IN[0] / 2 and abs(local[1]) <= BIN_IN[1] / 2 and BIN_FLOOR_T <= c[2] < BIN_WALL_H)


def on_shelf(center_world, thickness_m: float) -> bool:
    c = np.asarray(center_world, dtype=float)
    local = FRAME_B.to_local_xy(c[:2])
    x0, x1 = SHELF_SLOT_X[0] - SHELF_SLOT_HALF, SHELF_SLOT_X[-1] + SHELF_SLOT_HALF
    return bool(x0 <= local[0] <= x1 and abs(local[1] - SHELF_DEPTH_C) <= SHELF_DEPTH / 2
                and abs(c[2] - (SHELF_BOARD_TOP + thickness_m / 2)) < 0.015)
