"""交渉ルールの境界値。"""

import math

import pytest

from pai_book_intake.negotiation import (BinnedSuccessEstimator, TaskTimer, praise_threshold,
                                         should_praise, should_request_fix)


@pytest.mark.parametrize("p, expected", [
    (0.80, False),  # ちょうど 0.80 は依頼しない
    (0.7999, True),
    (0.80001, False),
    (0.0, True),
    (1.0, False),
])
def test_fix_request_boundary(p, expected):
    assert should_request_fix(p) is expected


@pytest.mark.parametrize("p", [-0.01, 1.01, math.nan])
def test_fix_request_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        should_request_fix(p)


HIST = [9.0, 11.0, 9.0, 11.0, 10.0]  # 平均 10、標本標準偏差 1 → 基準 9.0


def test_praise_threshold_value():
    assert praise_threshold(HIST) == pytest.approx(9.0)


@pytest.mark.parametrize("t, expected", [
    (9.0, True),  # 平均 − 1σ ちょうどは好評価
    (8.5, True),
    (9.0001, False),
    (10.0, False),
])
def test_praise_boundary(t, expected):
    assert should_praise(t, HIST) is expected


@pytest.mark.parametrize("n", [0, 1, 4])
def test_no_praise_with_short_history(n):
    # 記録が 5 件未満なら、どれだけ速くても好評価しない
    assert should_praise(0.0, HIST[:n]) is False
    assert praise_threshold(HIST[:n]) is None


def test_praise_with_exactly_min_history():
    assert should_praise(9.0, HIST[:5]) is True


def test_zero_variance_history():
    hist = [10.0] * 5
    assert should_praise(10.0, hist) is True
    assert should_praise(10.0001, hist) is False


def test_praise_rejects_bad_history():
    with pytest.raises(ValueError):
        should_praise(1.0, [1.0, -2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        praise_threshold(HIST, min_history=1)


def test_task_timer_judges_before_recording():
    tt = TaskTimer()
    results = [tt.record_and_judge("robot-b-sort", t) for t in HIST]
    assert results == [False] * 5  # 最初の 5 件は記録が足りない
    assert tt.record_and_judge("robot-b-sort", 9.0) is True
    assert len(tt.history["robot-b-sort"]) == 6


def test_estimator_bins_and_update():
    est = BinnedSuccessEstimator()
    good = {"overhang_err_mm": 5.0, "yaw_err_deg": 3.0, "on_table": True}
    edge = {"overhang_err_mm": 10.0, "yaw_err_deg": 8.0, "on_table": True}  # 境界は内側の区分
    mid = {"overhang_err_mm": 15.0, "yaw_err_deg": 10.0, "on_table": True}
    off = {"overhang_err_mm": 0.0, "yaw_err_deg": 0.0, "on_table": False}
    assert est.estimate(good) == pytest.approx(0.95)
    assert est.estimate(edge) == pytest.approx(0.95)
    assert est.estimate(mid) == pytest.approx(0.80)
    assert should_request_fix(est.estimate(mid)) is False  # 0.80 ちょうど
    assert est.estimate(off) == 0.0
    est.update(mid, False)
    assert est.estimate(mid) < 0.80
    assert should_request_fix(est.estimate(mid)) is True
