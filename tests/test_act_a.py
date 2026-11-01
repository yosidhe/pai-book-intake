"""act_a の比べ方・外乱の向き（GPU を使わない部分）と、p2_line の checkpoint の記録のしかた・停止後の数え方。"""

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


def test_book_a_failed_is_still_counted_after_a_stop():
    """止まった後に台に残した本だけが「届かないのが正しい」。A が台に置けなかった本は失敗に数える"""
    from pai_book_intake.p2_line import left_on_tray_after_stop

    assert left_on_tray_after_stop(stopped=True, cleared=False, on_tray=True)
    assert not left_on_tray_after_stop(stopped=True, cleared=False, on_tray=False)  # A の失敗（not_on_table）
    assert not left_on_tray_after_stop(stopped=False, cleared=False, on_tray=True)
    assert not left_on_tray_after_stop(stopped=True, cleared=True, on_tray=True)


def test_face_down_book_is_given_to_act_as_face_up():
    """p2_line の裏向き（x 軸まわりに 180°）の本は、ACT に渡すときだけ表向きと同じ向きの数値になる。表向きはそのまま"""
    from pai_book_intake.act_a import face_up_frame
    from pai_book_intake.geometry import quat_mul, quat_to_mat, quat_z

    q = quat_z(np.radians([30.0, -8.0]))
    down = quat_mul(q, np.broadcast_to(np.array([0.0, 1.0, 0.0, 0.0]), (2, 4)))
    up_m = quat_to_mat(q)
    mixed = np.stack([quat_to_mat(down)[0], up_m[1]])  # 1 冊目は裏向き・2 冊目は表向き
    got = face_up_frame(mixed)
    assert np.allclose(got, up_m)
    assert np.allclose(got[:, :, 0], mixed[:, :, 0])  # 背の向き（x 軸）は変えない
    assert np.allclose(mixed[0, :, 2], [0.0, 0.0, -1.0])  # 渡す前の値は変えない（コピーして直す）


class _Tensor:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Entity:
    def __init__(self, vels):
        self.vels = iter(vels)
        self.vel = None

    def get_dofs_velocity(self):
        self.vel = next(self.vels)
        return _Tensor(self.vel)

    def get_qpos(self):
        return _Tensor([[0.5, 0.1]])


class _Robot:
    n = 1
    home = np.array([0.5, 0.1])
    force_now = np.array([[10.0, 20.0]])

    def __init__(self, vels):
        self.target = np.array([[0.0, 0.0]])
        self.e = _Entity(vels)


def test_settle_home_moves_all_joints_and_waits_until_still():
    from pai_book_intake.p2_line import g_settle_home

    r = _Robot([[[0.5, 0.0]], [[0.02, 0.0]], [[0.0, 0.009]]])
    out: dict = {}
    assert sum(g_settle_home(r, out, steps=4)) == 4 + 2
    assert np.allclose(r.target, r.home)
    assert out["settled"] is True and out["waited_steps"] == 2
    assert np.allclose(out["force_now"], r.force_now)


def test_settle_home_gives_up_and_records_it():
    from pai_book_intake.p2_line import g_settle_home

    r = _Robot([[[0.5, 0.0]]] * 4)
    out: dict = {}
    assert sum(g_settle_home(r, out, steps=1, max_wait=3)) == 1 + 3
    assert out["settled"] is False and out["waited_steps"] == 3


def test_act_uses_only_the_first_tray_slot():
    from pai_book_intake.p2_line import tray_slots_used
    from pai_book_intake.site import DEFAULT_SITE, load_site

    slots = load_site(DEFAULT_SITE)["appraisal_table"]["slots"]
    assert slots[0][0] == 0.25
    assert tray_slots_used("act", slots) == 1
    assert tray_slots_used("scripted", slots) == len(slots)
