"""物理的にありえるかの判定（Genesis なし）。"""

import math

import numpy as np
import pytest

from pai_book_intake.plausibility import Monitor, Rules, tilt_deg

DT = 0.02


def _run(mon, steps, **kw):
    for k in range(steps):
        mon.update(k * DT, DT, **kw)


def test_tilt_ignores_face_up_or_down():
    assert tilt_deg([1.0, -1.0, 0.0])[:2] == pytest.approx([0.0, 0.0])
    assert tilt_deg([0.0])[0] == pytest.approx(90.0)
    assert tilt_deg([math.cos(math.radians(30))])[0] == pytest.approx(30.0)


def test_flat_book_on_table_is_plausible():
    mon = Monitor(2, 1)
    _run(mon, 100, up_z=np.ones((2, 1)), hand=False, wall=False, support=True)
    assert mon.ok.all()


def test_book_standing_by_itself_is_rejected_at_once():
    # 2026-10-12 ユーザー決定: 支えなしで縁・角で立つ姿勢は一瞬でも失敗
    mon = Monitor(2, 1)
    up = np.array([[0.0], [1.0]])  # 環境 0 は長い辺で立っている
    mon.update(0.0, DT, up_z=up, hand=False, wall=False, support=True)
    assert list(mon.ok) == [False, True]
    assert mon.violated["edge_rest"][0] and mon.first_book["edge_rest"][0] == 0


def test_tilt_just_under_threshold_is_allowed_and_over_is_not():
    mon = Monitor(2, 1)
    up = np.cos(np.radians(np.array([[19.9], [20.1]])))
    _run(mon, 50, up_z=up, hand=False, wall=False, support=True)
    assert list(mon.ok) == [True, False]


def test_duration_tolerance_is_still_configurable():
    mon = Monitor(1, 1, Rules(rest_tilt_s=0.3))
    _run(mon, 15, up_z=0.0, hand=False, wall=False, support=True)  # 0.30 s ちょうどまでは数えない
    assert mon.ok.all()
    _run(mon, 2, up_z=0.0, hand=False, wall=False, support=True)
    assert not mon.ok.all()


def test_falling_over_through_the_tilt_is_rejected():
    # 倒れる途中でも、支えなしで 20° を超えて何かに載っていれば失敗（一瞬でも）
    mon = Monitor(1, 1)
    _run(mon, 2, up_z=np.cos(np.radians(60)), hand=False, wall=False, support=True)
    _run(mon, 50, up_z=1.0, hand=False, wall=False, support=True)
    assert not mon.ok.all()
    assert mon.max_rest_s[0] == pytest.approx(2 * DT)


@pytest.mark.parametrize("hand, wall", [(True, False), (False, True)])
def test_tilt_while_supported_by_hand_or_wall_is_allowed(hand, wall):
    mon = Monitor(1, 1)
    _run(mon, 100, up_z=0.0, hand=hand, wall=wall, support=True)
    assert mon.ok.all()


def test_airborne_book_is_rejected():
    mon = Monitor(1, 2)
    support = np.array([[True, False]])  # 2 冊目は何にも触れていない
    _run(mon, 12, up_z=1.0, hand=False, wall=False, support=support)
    assert mon.ok.all()  # 0.24 s
    _run(mon, 2, up_z=1.0, hand=False, wall=False, support=support)
    assert not mon.ok[0] and mon.first_book["airborne"][0] == 1


def test_held_in_hand_is_not_airborne():
    mon = Monitor(1, 1)
    _run(mon, 100, up_z=1.0, hand=True, wall=False, support=False)
    assert mon.ok.all()


def test_penetration_and_inside_and_peak_force():
    mon = Monitor(3, 1, Rules(penetration_m=0.002))
    mon.update(0.0, DT, up_z=1.0, hand=True, wall=False, support=False,
               penetration_m=np.array([[0.001], [0.0025], [0.0]]), inside=np.array([[False], [False], [True]]),
               hand_force_n=np.array([[5.0], [40.0], [1.0]]))
    assert list(mon.ok) == [True, False, False]
    assert mon.max_penetration_m == pytest.approx([0.001, 0.0025, 0.0])
    assert mon.peak_hand_force_n == pytest.approx([5.0, 40.0, 1.0])
    d = mon.per_env()
    assert list(d["violated_penetration"]) == [False, True, True]


def test_counter_resets_when_condition_breaks():
    mon = Monitor(1, 1, Rules(rest_tilt_s=0.3))
    for k in range(40):
        # 0.1 s 立つ → 0.02 s 手が触れる、を繰り返す（連続 0.3 s にはならない）
        hand = (k % 6) == 5
        mon.update(k * DT, DT, up_z=0.0, hand=hand, wall=False, support=True)
    assert mon.ok.all()


def test_records_when_and_which_book_tilted_most():
    mon = Monitor(1, 2, Rules(rest_tilt_s=0.3))
    mon.update(0.0, DT, up_z=np.array([[1.0, 1.0]]), hand=False, wall=False, support=True)
    mon.update(0.5, DT, up_z=np.array([[1.0, math.cos(math.radians(25))]]), hand=False, wall=False, support=True)
    mon.update(0.6, DT, up_z=np.array([[1.0, math.cos(math.radians(10))]]), hand=False, wall=False, support=True)
    assert mon.max_rest_tilt_deg[0] == pytest.approx(25.0)
    assert mon.max_rest_tilt_t[0] == pytest.approx(0.5) and mon.max_rest_tilt_book[0] == 1
