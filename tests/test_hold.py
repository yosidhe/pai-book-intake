"""保留棚に積んでも崩れないかの判定（hold.py。02_確認事項 B14）。"""

import math

import pytest

from pai_book_intake.hold import StackRule, TopBook, inset_from_edge, plan_stack

TABLE_C, TABLE_S, TOP = (1.05, 1.12), (0.40, 0.34), 0.10


def top_book(xy=TABLE_C, yaw=0.0, size=(0.105, 0.148), top_z=TOP + 0.012, tilt=0.5):
    return TopBook(center_xy=xy, yaw=yaw, size_xy=size, top_z=top_z, tilt_deg=tilt)


def test_inset_from_edge():
    assert inset_from_edge((0.0, 0.0), (0.0, 0.0), 0.0, (0.2, 0.1)) == pytest.approx(0.05)
    assert inset_from_edge((0.09, 0.0), (0.0, 0.0), 0.0, (0.2, 0.1)) == pytest.approx(0.01)
    assert inset_from_edge((0.0, 0.08), (0.0, 0.0), 0.0, (0.2, 0.1)) == pytest.approx(-0.03)  # 外は負
    # 向きを 90° 回すと、長い辺が y に沿う
    assert inset_from_edge((0.0, 0.08), (0.0, 0.0), math.pi / 2, (0.2, 0.1)) == pytest.approx(0.02)


def test_first_book_goes_to_the_table_center():
    p = plan_stack(TABLE_C, TABLE_S, TOP, None, 0.012)
    assert p.ok and p.target_xy == TABLE_C and p.surface_z == TOP


def test_next_book_goes_on_the_top_book_center():
    tb = top_book(xy=(1.06, 1.11))
    p = plan_stack(TABLE_C, TABLE_S, TOP, tb, 0.015)
    assert p.ok and p.target_xy == (1.06, 1.11) and p.surface_z == tb.top_z
    assert p.inset_m == pytest.approx(0.105 / 2 - 0.010)  # 狭い辺の半分 − 置く位置の誤差


def test_too_high_is_refused():
    rule = StackRule(max_stack_m=0.12)
    assert plan_stack(TABLE_C, TABLE_S, TOP, top_book(top_z=TOP + 0.10), 0.020, rule).ok  # ちょうど上限
    p = plan_stack(TABLE_C, TABLE_S, TOP, top_book(top_z=TOP + 0.10), 0.021, rule)
    assert not p.ok and p.reason == "too_high"
    # 上限を小さくすると早く満杯になる（--hold-max-stack-m）
    p = plan_stack(TABLE_C, TABLE_S, TOP, top_book(top_z=TOP + 0.02), 0.012, StackRule(max_stack_m=0.03))
    assert not p.ok and p.reason == "too_high"


def test_tilted_top_book_is_refused():
    p = plan_stack(TABLE_C, TABLE_S, TOP, top_book(tilt=3.5), 0.012)
    assert not p.ok and p.reason == "top_book_tilted"
    assert plan_stack(TABLE_C, TABLE_S, TOP, top_book(tilt=3.0), 0.012).ok


def test_small_top_book_leaves_too_little_support():
    """いちばん上の本が小さく、置く位置の誤差を見込むと重心が縁から 15 mm 内側に入らない"""
    p = plan_stack(TABLE_C, TABLE_S, TOP, top_book(size=(0.045, 0.30)), 0.012)
    assert not p.ok and p.reason == "center_of_mass_near_edge" and p.inset_m < 0.015
    assert plan_stack(TABLE_C, TABLE_S, TOP, top_book(size=(0.051, 0.30)), 0.012).ok


def test_pile_already_off_the_table_edge_is_refused():
    p = plan_stack(TABLE_C, TABLE_S, TOP, top_book(xy=(TABLE_C[0] + 0.19, TABLE_C[1])), 0.012)
    assert not p.ok and p.reason == "top_book_off_table"
