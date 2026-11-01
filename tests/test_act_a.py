"""act_a の比べ方・外乱の向き（GPU を使わない部分）と、p2_line の checkpoint の記録のしかた。"""

from pathlib import Path

import numpy as np
import pytest

from pai_book_intake.act_a import compare, nudge_dirs, outcomes


def _res(run_id, ok_rows, seed=1, nudge=None, pile=("book-01", "book-02")):
    return {"run_id": run_id, "args": {"seed": seed}, "pile_bottom_to_top": list(pile), "n_envs": len(ok_rows[0]),
            "jitter": {"pos_m": 0.02, "yaw_deg": 10.0}, "nudge": nudge,
            "per_book": [{"book_id": b, "ok": ok} for b, ok in zip(("book-02", "book-01"), ok_rows)]}


def test_outcomes_by_book_and_env():
    r = _res("a", [[True, False], [False, True]])
    assert outcomes(r) == {("book-02", 0): True, ("book-02", 1): False, ("book-01", 0): False, ("book-01", 1): True}


def test_compare_lists_pairs_only_one_side_failed():
    a = _res("a", [[True, False, False], [True, True, True]])
    b = _res("b", [[True, True, False], [False, True, True]])
    c = compare(a, b)
    assert c["pairs"] == 6 and c["a_ok"] == 4 and c["b_ok"] == 4
    assert c["only_a_failed"] == [["book-02", 1]]
    assert c["only_b_failed"] == [["book-01", 0]]
    assert c["both_failed"] == [["book-02", 2]]


@pytest.mark.parametrize("change", [{"seed": 2}, {"nudge": {"mm": 15.0, "after_s": 1.5}},
                                    {"pile": ("book-03", "book-02")}])
def test_compare_refuses_different_conditions(change):
    a = _res("a", [[True], [True]])
    b = _res("b", [[True], [True]], **change)
    with pytest.raises(ValueError):
        compare(a, b)


def test_nudge_dirs_same_for_same_seed_and_length_is_mm():
    d1 = nudge_dirs(1000, 2, 50, 15.0)
    d2 = nudge_dirs(1000, 2, 50, 15.0)
    assert np.array_equal(d1, d2)
    assert np.allclose(np.linalg.norm(d1, axis=1), 0.015)
    assert not np.allclose(d1, nudge_dirs(1000, 3, 50, 15.0))  # 本ごとに向きが変わる
    assert not np.allclose(d1, nudge_dirs(1001, 2, 50, 15.0))


def test_p2_line_records_ckpt_without_absolute_path():
    from pai_book_intake.p2_line import ckpt_name

    assert ckpt_name(Path("/workspace/outputs/act_train/20261028T160045-6cf9ae/final")) == "20261028T160045-6cf9ae/final"
