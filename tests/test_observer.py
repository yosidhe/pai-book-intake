"""オブザーバ: 止める条件（しきい値の手前と到達）と報告書。"""

import pytest

from pai_book_intake.bus import Bus, read_log
from pai_book_intake.observer import Observer, ObserverConfig
from pai_book_intake.runlog import classify


def setup(tmp_path, **cfg):
    bus = Bus("run-obs", tmp_path / "bus.jsonl")
    return bus, Observer(bus, ObserverConfig(**cfg))


def result(bus, outcome, env=0, dest=None, book="b", value=0, reason=""):
    bus.publish("robot-b-sort", "observer", "event", "book.result",
                {"env": env, "book_id": book, "title": f"t-{book}", "isbn": "9784000000017", "outcome": outcome,
                 "destination": dest, "value_jpy": value, "reason": reason})


def barcode(bus, ok, env=0):
    bus.publish("inspect-flip", "appraisal", "event", "barcode.read", {"env": env, "ok": ok})


def stops(bus):
    return [m for m in bus.history if m.topic == "stop"]


def test_consecutive_failures_stop(tmp_path):
    bus, obs = setup(tmp_path, max_consecutive_failures=3)
    result(bus, "rejected")
    result(bus, "failed")
    assert not obs.is_stopped(0)
    result(bus, "succeeded", dest="discard")  # 成功で数え直し
    result(bus, "rejected")
    result(bus, "rejected")
    assert not obs.is_stopped(0)
    result(bus, "failed")
    assert obs.is_stopped(0) and obs.box(0).stop_reason == "consecutive_failures"
    result(bus, "failed")
    assert len(stops(bus)) == 1  # stop は 1 回だけ
    s = stops(bus)[0]
    assert s.from_task == "observer" and s.to_task == "*" and s.payload["env"] == 0


def test_shelf_full_stop(tmp_path):
    bus, obs = setup(tmp_path, shelf_capacity=2)
    result(bus, "succeeded", dest="shelf", book="b1", value=100)
    assert not obs.is_stopped()
    result(bus, "succeeded", dest="shelf", book="b2", value=200)
    assert obs.is_stopped() and obs.box(0).stop_reason == "shelf_full"


def test_barcode_unreadable_stop_is_consecutive(tmp_path):
    bus, obs = setup(tmp_path, barcode_unreadable_limit=2)
    barcode(bus, False)
    barcode(bus, True)
    barcode(bus, False)
    assert not obs.is_stopped()
    barcode(bus, False)
    assert obs.is_stopped() and obs.box(0).stop_reason == "barcode_unreadable"
    assert obs.box(0).barcode_unreadable == 3 and obs.box(0).barcode_reads == 4


def test_envs_are_separate_boxes(tmp_path):
    bus, obs = setup(tmp_path, max_consecutive_failures=1)
    result(bus, "failed", env=1)
    assert obs.is_stopped(1) and not obs.is_stopped(0)


def test_invalid_outcome_raises(tmp_path):
    bus, obs = setup(tmp_path)
    with pytest.raises(ValueError):
        result(bus, "ok")


def test_report(tmp_path):
    bus, obs = setup(tmp_path)
    result(bus, "succeeded", dest="shelf", book="b1", value=160)
    result(bus, "succeeded", dest="discard", book="b2", value=0)
    result(bus, "rejected", book="b3", reason="not_on_table")
    bus.publish("inspect-flip", "robot-a-pick", "feedback", "fix_request", {"env": 0, "p": 0.5})
    bus.publish("robot-b-sort", "inspect-flip", "feedback", "praise", {"env": 0})
    bus.publish("robot-b-sort", "inspect-flip", "feedback", "praise", {"env": 0})
    paths = obs.write_reports(tmp_path, title="テスト")
    assert [p.name for p in paths] == ["observer_report_env0.md"]
    md = paths[0].read_text(encoding="utf-8")
    assert "処理した本: 3 冊（成功 2 / 不合格 1 / 異常 0）" in md
    assert "| b1 | t-b1 |" in md and "| b2 | t-b2 |" in md
    assert "160 円" in md
    assert "| fix_request | inspect-flip → robot-a-pick | 1 |" in md
    assert "| praise | robot-b-sort → inspect-flip | 2 |" in md
    assert "not_on_table" in md
    bus.close()
    assert len(read_log(tmp_path / "bus.jsonl")) == 6


def test_empty_report_gives_instruction(tmp_path):
    bus, obs = setup(tmp_path)
    md = obs.report_markdown(0)
    assert "処理した本: 0 冊" in md and "まだありません" in md
    bus.close()


@pytest.mark.parametrize("ok, err, expected", [
    (True, None, "succeeded"), (False, None, "rejected"), (True, RuntimeError("x"), "failed"),
    (False, RuntimeError("x"), "failed"),
])
def test_run_status(ok, err, expected):
    assert classify(ok, err) == expected


def test_config_validation():
    with pytest.raises(ValueError):
        ObserverConfig(shelf_capacity=0)


def test_book_held_for_no_shelf_space_stops_the_box(tmp_path):
    """運搬許可が「本棚に空きがない」で本を台に残したら、箱ごと止める（人が本棚を空けるまで進めない）。
    ほかの理由で台に残った本（読めないなど）では止めない（それは barcode_unreadable などの規則で見る）"""
    bus, obs = setup(tmp_path, shelf_capacity=20)
    bus.publish("appraisal", "observer", "event", "tray.held", {"env": 0, "book_id": "b1", "slot": 0,
                                                                "reason": "barcode_not_read"})
    assert not obs.is_stopped(0)
    bus.publish("appraisal", "observer", "event", "tray.held", {"env": 0, "book_id": "b2", "slot": 1,
                                                                "reason": "shelf_full"})
    assert obs.is_stopped(0) and obs.box(0).stop_reason == "shelf_full"
    assert stops(bus)[0].payload["book_id"] == "b2"


def test_with_a_hold_table_unreadable_books_do_not_stop_the_box(tmp_path):
    """保留棚があるとき（barcode_unreadable_limit=None）は、読めない本が続いても止めない（B14）。
    保留棚に積めない（hold_shelf_full）で本を台に残したら、箱ごと止める"""
    bus, obs = setup(tmp_path, barcode_unreadable_limit=None, shelf_capacity=20)
    for k in range(5):
        barcode(bus, False)
        result(bus, "succeeded", dest="hold", book=f"b{k}", reason="barcode_unreadable")
    assert not obs.is_stopped(0)
    assert [b["book_id"] for b in obs.box(0).hold] == [f"b{k}" for k in range(5)]
    bus.publish("appraisal", "observer", "event", "tray.held", {"env": 0, "book_id": "b5", "slot": 0,
                                                                "reason": "hold_shelf_full"})
    assert obs.is_stopped(0) and obs.box(0).stop_reason == "hold_shelf_full"
    assert stops(bus)[0].payload["book_id"] == "b5"
    md = obs.report_markdown(0)
    assert "## 保留棚" in md and "| b0 |" in md and "barcode_unreadable" in md


def test_without_a_hold_table_unreadable_books_still_stop_the_box(tmp_path):
    bus, obs = setup(tmp_path, barcode_unreadable_limit=2)
    barcode(bus, False)
    assert not obs.is_stopped(0)
    barcode(bus, False)
    assert obs.is_stopped(0) and obs.box(0).stop_reason == "barcode_unreadable"
