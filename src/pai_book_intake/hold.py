"""保留棚（査定できなかった本を積む平らな台）に、次の本を積んでも崩れないかの判定。Genesis に依存しない。

02_確認事項 B14 の回答: 保留棚は B の待機位置の近くの平らな台。廃棄箱と同じ置き方で積む。
崩れるおそれがあれば積まない（その本は取らずに査定台に残し、オブザーバが箱ごと止める）。

判定（B が台から取る**前**に確かめる。どれか 1 つでも満たさなければ積まない）
  1. 積んだ後の上面が、台の上面から max_stack_m 以下（本棚の平積みと同じ 0.12 m）
  2. 新しい本の重心が、いちばん上の本（置いた後に測った位置と向き）の面の縁から min_inset_m 以上内側に載る。
     置く位置はいちばん上の本の中心（1 冊目は台の中心）。置く位置の誤差（place_err_m）だけずれても満たすこと
  3. いちばん上の本の傾きが max_tilt_deg 以下
  4. いちばん上の本の中心が、台の縁から min_inset_m 以上内側にある（山がもう台の縁へずれていない）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HOLDABLE = ("barcode_unreadable", "isbn_not_in_catalog", "isbn_mismatch")  # 保留棚へ運ぶ理由（shelf_full は運ばない）
FULL_REASON = "hold_shelf_full"  # 積めないときに台に残す理由（オブザーバが箱ごと止める）


@dataclass(frozen=True)
class StackRule:
    max_stack_m: float = 0.12
    min_inset_m: float = 0.015
    place_err_m: float = 0.010
    max_tilt_deg: float = 3.0


@dataclass(frozen=True)
class TopBook:
    """保留棚のいちばん上の本（置いた後に測った値。世界座標）。"""
    center_xy: tuple[float, float]
    yaw: float
    size_xy: tuple[float, float]  # 上から見た大きさ（本の座標の x・y。背の向き・天地の向き）
    top_z: float
    tilt_deg: float


@dataclass(frozen=True)
class StackPlan:
    ok: bool
    reason: str
    target_xy: tuple[float, float] | None
    surface_z: float | None  # 置く面（いちばん上の本の上面。1 冊目は台の上面）
    inset_m: float | None


def inset_from_edge(point_xy, center_xy, yaw: float, size_xy) -> float:
    """点が、向き yaw・大きさ size_xy の長方形の縁から、どれだけ内側にあるか（外なら負）。"""
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(point_xy, float) - np.asarray(center_xy, float)
    lx, ly = c * d[0] + s * d[1], -s * d[0] + c * d[1]
    return float(min(size_xy[0] / 2 - abs(lx), size_xy[1] / 2 - abs(ly)))


def plan_stack(table_center, table_size, table_top: float, top: TopBook | None, new_thickness: float,
               rule: StackRule = StackRule()) -> StackPlan:
    """次の本（厚み new_thickness）を保留棚に積めるか。積めるなら置く位置と面を返す。"""
    if top is None:
        target, surface = (float(table_center[0]), float(table_center[1])), float(table_top)
        support = (tuple(table_center), 0.0, tuple(table_size))
    else:
        if top.tilt_deg > rule.max_tilt_deg:
            return StackPlan(False, "top_book_tilted", None, None, None)
        if inset_from_edge(top.center_xy, table_center, 0.0, table_size) < rule.min_inset_m:
            return StackPlan(False, "top_book_off_table", None, None, None)  # 下の山がもう台の縁へずれている
        target, surface = (float(top.center_xy[0]), float(top.center_xy[1])), float(top.top_z)
        support = (top.center_xy, top.yaw, top.size_xy)
    if surface + new_thickness - table_top > rule.max_stack_m + 1e-9:
        return StackPlan(False, "too_high", None, None, None)
    inset = inset_from_edge(target, *support) - rule.place_err_m
    if inset < rule.min_inset_m:
        return StackPlan(False, "center_of_mass_near_edge", None, None, inset)
    return StackPlan(True, "ok", target, surface, inset)
