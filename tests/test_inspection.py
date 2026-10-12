"""検品エリアのやりとり（inspect-scan ⇄ robot-a-pick → observer）と、バーコードの向きの読み取り。"""

import numpy as np
import pytest
from PIL import Image

from pai_book_intake.bus import Bus, read_log
from pai_book_intake.inspection import (ESCALATED, MAX_FLIPS, READ, WAIT_FLIP, WAIT_SCAN, InspectionSession,
                                        image_yaw_to_world, scan_result, wrap_deg)
from pai_book_intake.observer import Observer

ISBN = "9784000000017"


def session(tmp_path, **kw):
    bus = Bus("run-insp", tmp_path / "bus.jsonl", sim_clock=lambda: 0.0)
    return bus, InspectionSession(bus, env=kw.pop("env", 0), book_id="b1", **kw)


def topics(bus):
    return [(m.from_task, m.to_task, m.type, m.topic) for m in bus.history]


def test_visible_on_first_scan_sends_orientation_and_ends(tmp_path):
    bus, s = session(tmp_path)
    assert s.on_scan(ISBN, 30.0) == READ and s.done
    bus.close()
    assert topics(bus) == [("inspect-scan", "robot-a-pick", "event", "scan.result")]
    p = bus.history[0].payload
    assert {k: p[k] for k in ("barcode_visible", "isbn", "barcode_yaw_deg", "attempt")} == \
        {"barcode_visible": True, "isbn": ISBN, "barcode_yaw_deg": 30.0, "attempt": 0}
    assert p["env"] == 0 and p["book_id"] == "b1"
    assert read_log(tmp_path / "bus.jsonl")[0]["payload"]["barcode_yaw_deg"] == 30.0


def test_not_visible_requests_flip_then_reads(tmp_path):
    bus, s = session(tmp_path)
    assert s.on_scan(None, None) == WAIT_FLIP
    fix = bus.history[-1]
    assert (fix.type, fix.topic, fix.to_task) == ("feedback", "fix_request", "robot-a-pick")
    assert fix.payload["action"] == "flip" and fix.payload["attempt"] == 1
    # 修正依頼は受け取る側（robot-a-pick）の records に書く
    assert fix.to_storage_record()["task"] == "robot-a-pick"
    assert s.on_flip_done() == WAIT_SCAN
    assert s.on_scan(ISBN, -170.0) == READ
    bus.close()
    scans = [m.payload for m in bus.history if m.topic == "scan.result"]
    assert [(p["barcode_visible"], p["attempt"]) for p in scans] == [(False, 0), (True, 1)]
    assert scans[0]["isbn"] is None and scans[0]["barcode_yaw_deg"] is None
    assert [m.topic for m in bus.history] == ["scan.result", "fix_request", "flip.done", "scan.result"]


def test_escalates_after_two_failed_flips(tmp_path):
    bus, s = session(tmp_path, env=3)
    seen = []
    obs = Observer(bus)
    bus.subscribe(lambda m: seen.append(m.topic), to_task="observer")
    assert s.on_scan(None, None) == WAIT_FLIP
    s.on_flip_done()
    assert s.on_scan(None, None) == WAIT_FLIP
    s.on_flip_done(ok=False, detail={"reason": "dropped"})
    assert s.on_scan(None, None) == ESCALATED and s.done
    bus.close()
    fixes = [m.payload["attempt"] for m in bus.history if m.topic == "fix_request"]
    assert fixes == [1, 2] == list(range(1, MAX_FLIPS + 1))
    esc = [m for m in bus.history if m.topic == "escalation"]
    assert len(esc) == 1
    assert (esc[0].from_task, esc[0].to_task, esc[0].type) == ("inspect-scan", "observer", "event")
    assert esc[0].payload == {"env": 3, "book_id": "b1", "reason": "barcode_unreadable", "scans": 3, "flips": 2}
    assert seen == ["escalation"]
    # 同じ本の撮り直しで、オブザーバが箱ごと止めてはいけない（barcode.read は出していない）
    assert not obs.is_stopped(3) and obs.box(3).barcode_reads == 0


def test_protocol_order_is_enforced(tmp_path):
    bus, s = session(tmp_path)
    with pytest.raises(RuntimeError):
        s.on_flip_done()  # 依頼していないのに完了
    s.on_scan(None, None)
    with pytest.raises(RuntimeError):
        s.on_scan(ISBN, 0.0)  # 裏返しの完了を待たずに撮り直し
    s.on_flip_done()
    s.on_scan(ISBN, 0.0)
    with pytest.raises(RuntimeError):
        s.on_scan(ISBN, 0.0)  # 読めた後
    bus.close()


def test_zero_flips_allowed_escalates_immediately(tmp_path):
    bus, s = session(tmp_path, max_flips=0)
    assert s.on_scan(None, None) == ESCALATED
    bus.close()
    assert [m.topic for m in bus.history] == ["scan.result", "escalation"]


@pytest.mark.parametrize("isbn,yaw,attempt", [
    (ISBN, None, 0),  # 読めたのに向きが無い
    ("9784000000018", 0.0, 0),  # チェックディジットが違う
    (None, 10.0, 0),  # 見えないのに向き
    (ISBN, 0.0, -1),
    (ISBN, float("nan"), 0),
])
def test_scan_result_rejects_inconsistent(isbn, yaw, attempt):
    with pytest.raises(ValueError):
        scan_result(isbn, yaw, attempt)


def test_angles():
    assert wrap_deg(190.0) == -170.0 and wrap_deg(-180.0) == 180.0 and wrap_deg(540.0) == 180.0
    assert scan_result(ISBN, 350.0, 0).barcode_yaw_deg == -10.0
    # カメラの上が世界の +x（yaw 0°）: 画像の右は世界の -y（yaw -90°）、画像の上は世界の +x
    assert image_yaw_to_world(0.0, 0.0) == -90.0
    assert image_yaw_to_world(90.0, 0.0) == 0.0
    assert image_yaw_to_world(90.0, 90.0) == 90.0


# ---------- バーコードの向き（zxing-cpp。Genesis は要らない） ----------

def _barcode_canvas(angle_ccw: float) -> np.ndarray:
    import zxingcpp  # Docker のイメージに入っている（docker/requirements.txt）

    bc = np.asarray(zxingcpp.write_barcode_to_image(zxingcpp.create_barcode(ISBN, zxingcpp.BarcodeFormat.EAN13),
                                                    scale=4))
    img = Image.fromarray(bc).convert("L")
    canvas = Image.new("L", (900, 900), 255)
    canvas.paste(img, (450 - img.width // 2, 450 - img.height // 2))
    return np.asarray(canvas.rotate(angle_ccw, fillcolor=255).convert("RGB"))


@pytest.mark.parametrize("angle", [0, 17, 30, 90, 145, 180, 250, -63])
def test_read_isbn_with_yaw_recovers_rotation(angle):
    from pai_book_intake.vision import read_isbn_with_yaw

    isbn, yaw = read_isbn_with_yaw(_barcode_canvas(angle))
    assert isbn == ISBN
    assert abs(wrap_deg(yaw - angle)) <= 4.0  # 細かい刻み 2° ＋ 読める範囲の端の揺れ


def test_read_isbn_with_yaw_blank():
    from pai_book_intake.vision import read_isbn_with_yaw

    assert read_isbn_with_yaw(np.full((300, 300, 3), 255, np.uint8)) == (None, None)
