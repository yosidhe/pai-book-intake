"""バス: 記録が先・記録の形・配る順序・不正な値は拒む。"""

import json

import numpy as np
import pytest

from pai_book_intake.bus import Bus, read_log


def make_bus(tmp_path):
    return Bus("run-x", tmp_path / "bus.jsonl", sim_clock=lambda: 0.0)


def test_every_message_is_logged_with_required_fields(tmp_path):
    bus = make_bus(tmp_path)
    bus.publish("robot-a-pick", "inspect-flip", "event", "place.done", {"env": 0, "x": 1})
    bus.publish("robot-b-sort", "inspect-flip", "feedback", "fix_request", {"env": 0, "p": 0.5})
    bus.close()
    rows = read_log(tmp_path / "bus.jsonl")
    assert len(rows) == 2
    for r in rows:
        for k in ("run_id", "msg_id", "seq", "sim_time", "wall_time", "from_task", "to_task", "type", "topic",
                  "payload", "storage"):
            assert k in r
        assert r["run_id"] == "run-x"
    assert [r["seq"] for r in rows] == [1, 2]
    assert rows[0]["msg_id"] != rows[1]["msg_id"]


def test_storage_record_mapping(tmp_path):
    bus = make_bus(tmp_path)
    ev = bus.publish("robot-a-pick", "inspect-flip", "event", "place.done", {"env": 0})
    fb = bus.publish("robot-b-sort", "inspect-flip", "feedback", "praise", {"env": 0})
    bus.close()
    # event は送った側、feedback は受け取る側の records に書く
    assert ev.to_storage_record()["task"] == "robot-a-pick"
    assert fb.to_storage_record()["task"] == "inspect-flip"
    rec = fb.to_storage_record()["record"]
    assert set(rec) == {"type", "from_task", "run_id", "payload", "occurred_at"}
    assert rec["type"] == "feedback" and rec["from_task"] == "robot-b-sort" and rec["run_id"] == "run-x"
    assert rec["payload"]["_msg"]["topic"] == "praise"
    assert rec["occurred_at"] == fb.wall_time


def test_log_is_written_before_dispatch_even_if_handler_raises(tmp_path):
    bus = make_bus(tmp_path)

    def boom(msg):
        raise RuntimeError("handler failed")

    bus.subscribe(boom)
    with pytest.raises(RuntimeError):
        bus.publish("appraisal", "robot-b-sort", "event", "decision", {})
    bus.close()
    assert len(read_log(tmp_path / "bus.jsonl")) == 1


def test_routing_and_order(tmp_path):
    bus = make_bus(tmp_path)
    got_all, got_b = [], []
    bus.subscribe(lambda m: got_all.append(m.topic))
    bus.subscribe(lambda m: got_b.append(m.topic), to_task="robot-b-sort")

    def reply(m):
        if m.topic == "ping":
            bus.publish("robot-b-sort", "appraisal", "event", "pong", {})

    bus.subscribe(reply, to_task="robot-b-sort", topic="ping")
    bus.publish("appraisal", "robot-b-sort", "event", "ping", {})
    bus.publish("observer", "*", "event", "stop", {})
    bus.publish("appraisal", "inspect-flip", "event", "other", {})
    bus.close()
    # 返信（pong）は ping を全員に配り終えてから配られる
    assert got_all == ["ping", "pong", "stop", "other"]
    assert got_b == ["ping", "stop"]


def test_sim_time_and_numpy_payload(tmp_path):
    t = [1.23456]
    bus = Bus("r", tmp_path / "b.jsonl", sim_clock=lambda: t[0])
    m = bus.publish("robot-a-pick", "inspect-flip", "sample", "pose", {"v": np.array([1.0, 2.0]), "f": np.float32(0.5)})
    bus.close()
    assert m.sim_time == 1.2346
    row = json.loads((tmp_path / "b.jsonl").read_text(encoding="utf-8"))
    assert row["payload"]["v"] == [1.0, 2.0] and row["payload"]["f"] == 0.5


@pytest.mark.parametrize("kw", [
    dict(from_task="robot-x", to_task="inspect-flip", type="event", topic="t"),
    dict(from_task="robot-a-pick", to_task="nobody", type="event", topic="t"),
    dict(from_task="robot-a-pick", to_task="inspect-flip", type="log", topic="t"),
    dict(from_task="robot-a-pick", to_task="inspect-flip", type="event", topic=""),
])
def test_rejects_invalid(tmp_path, kw):
    bus = make_bus(tmp_path)
    with pytest.raises(ValueError):
        bus.publish(**kw)
    bus.close()
    assert read_log(tmp_path / "bus.jsonl") == []
