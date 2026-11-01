"""運搬許可（release.py）の試験。査定が済んでいない本を意図的に作り、ロボット B が持っていかないことを確かめる。"""

from __future__ import annotations

import pytest

from pai_book_intake.appraisal import Decision, appraise
from pai_book_intake.books import DEFAULT_CATALOG, load_catalog
from pai_book_intake.bus import Bus, read_log
from pai_book_intake.release import (DELIVERED, HELD, PLACED, RELEASED, SCANNED, NotCleared, ReleaseGate)

CAT = load_catalog(DEFAULT_CATALOG)


def isbn(book_id: str) -> str:
    return CAT.by_id(book_id).isbn13


def make(tmp_path, n_slots=6, shelf_capacity=3):
    bus = Bus("run-x", tmp_path / "bus.jsonl", sim_clock=lambda: 0.0)
    return bus, ReleaseGate(bus, CAT, n_slots=n_slots, shelf_capacity=shelf_capacity)


def ready(g: ReleaseGate, book_id: str, appraised: bool = True) -> None:
    """台に置いて読む。appraised なら査定も済ませる。"""
    g.place(book_id)
    g.on_scan(book_id, isbn(book_id))
    if appraised:
        g.on_appraisal(book_id, appraise(isbn(book_id), CAT))


class FakeRobotB:
    """台の上の本を運ぶロボット B の代わり（物理なし）。許可の出た本だけを、許可の順に運ぶ。

    tray は「台の上に実際にある本」。B が運んだ本だけがここから消える。
    """

    def __init__(self, gate: ReleaseGate, tray: set[str]):
        self.g, self.tray, self.carried = gate, tray, []

    def cycle(self) -> list[str]:
        moved = []
        for it in self.g.pickable():
            dest = self.g.take(it.book_id, it.clearance_id)
            self.tray.remove(it.book_id)  # ここで初めて物理的に台から持ち上げる
            self.carried.append((it.book_id, dest))
            self.g.on_delivered(it.book_id, True)
            moved.append(it.book_id)
        return moved


def topics(bus: Bus, tmp_path, topic: str) -> list[dict]:
    bus.close()
    return [m for m in read_log(tmp_path / "bus.jsonl") if m["topic"] == topic]


def test_unappraised_book_is_not_carried(tmp_path):
    bus, g = make(tmp_path)
    ready(g, "book-03", appraised=False)  # 読めたが査定がまだ
    tray = {"book-03"}
    b = FakeRobotB(g, tray)
    assert b.cycle() == []
    assert tray == {"book-03"}  # 台に残っている
    assert g.items["book-03"].state == SCANNED
    with pytest.raises(NotCleared, match="not_released:scanned"):
        g.take("book-03", None)  # 直接持っていこうとしても止まる
    denied = topics(bus, tmp_path, "pick.denied")
    assert denied and denied[-1]["payload"]["book_id"] == "book-03"


def test_mixed_tray_only_cleared_books_leave(tmp_path):
    """査定が済んだ本・済んでいない本が混ざった台。B は済んだ本だけを、許可の順に運ぶ。"""
    bus, g = make(tmp_path)
    ready(g, "book-03")  # 完了（棚）
    ready(g, "book-05", appraised=False)  # 査定待ち
    ready(g, "book-04")  # 完了（廃棄）
    g.place("book-08")  # 置いただけ（まだ読んでいない）
    ready(g, "book-02", appraised=False)
    g.on_flip_requested("book-02")  # 裏返し待ち
    g.on_appraisal("book-02", appraise(isbn("book-02"), CAT))  # 査定は出たが、裏返し待ちなので運ばない
    ready(g, "book-01", appraised=False)
    g.on_appraisal("book-01", appraise(isbn("book-06"), CAT))  # 別の本の査定結果が届いた

    tray = {"book-03", "book-05", "book-04", "book-08", "book-02", "book-01"}
    b = FakeRobotB(g, tray)
    assert b.cycle() == ["book-03", "book-04"]
    assert b.carried == [("book-03", "shelf"), ("book-04", "discard")]
    assert tray == {"book-05", "book-08", "book-02", "book-01"}
    st = {it.book_id: (it.state, it.hold_reason) for it in g.remaining()}
    assert st == {"book-05": (SCANNED, None), "book-08": (PLACED, None), "book-02": (SCANNED, None),
                  "book-01": (HELD, "appraisal_for_other_book")}
    assert b.cycle() == []  # 何度回しても持っていかない

    # 査定が届けば次の回で運ぶ。裏返しが終われば運ぶ。人の判断待ちの本は運ばない
    g.on_appraisal("book-05", appraise(isbn("book-05"), CAT))
    g.on_flip_done("book-02")
    g.on_scan("book-02", isbn("book-02"))  # 裏返したあと読み直す
    assert b.cycle() == ["book-05", "book-02"]
    assert tray == {"book-08", "book-01"}


def test_release_order_is_clearance_order_not_placement_order(tmp_path):
    _, g = make(tmp_path)
    for k in ("book-03", "book-04", "book-05"):
        ready(g, k, appraised=False)
    g.on_appraisal("book-05", appraise(isbn("book-05"), CAT))
    g.on_appraisal("book-03", appraise(isbn("book-03"), CAT))
    assert [it.book_id for it in g.pickable()] == ["book-05", "book-03"]


def test_unread_barcode_is_never_sent_to_discard(tmp_path):
    """以前のデモは、査定の結果が無いと廃棄箱へ運んでいた。読めない本は運ばない。"""
    _, g = make(tmp_path)
    g.place("book-03")
    g.on_scan("book-03", None)
    g.on_appraisal("book-03", appraise(None, CAT))  # isbn_unread → discard という結果
    assert g.pickable() == []
    g.on_escalated("book-03", "barcode_unreadable")
    assert g.items["book-03"].state == HELD and g.items["book-03"].hold_reason == "barcode_unreadable"
    assert FakeRobotB(g, {"book-03"}).cycle() == []


def test_decision_without_destination_is_not_released(tmp_path):
    _, g = make(tmp_path)
    ready(g, "book-03", appraised=False)
    g.on_appraisal("book-03", Decision("unknown", 0, "manual_review", "book-03"))
    assert g.pickable() == [] and g.items["book-03"].state == SCANNED


def test_wrong_book_barcode_is_held(tmp_path):
    _, g = make(tmp_path)
    g.place("book-03")
    g.on_scan("book-03", isbn("book-04"))  # 置いた本と違う ISBN が読めた
    assert g.items["book-03"].state == HELD and g.items["book-03"].hold_reason == "isbn_mismatch"


def test_flip_request_revokes_clearance(tmp_path):
    bus, g = make(tmp_path)
    ready(g, "book-03")
    old = g.items["book-03"].clearance_id
    assert old is not None
    g.on_flip_requested("book-03")
    assert g.pickable() == []
    assert not g.authorize("book-03", old).ok
    assert topics(bus, tmp_path, "release.revoked")[0]["payload"]["clearance_id"] == old


def test_clearance_id_must_match(tmp_path):
    _, g = make(tmp_path)
    ready(g, "book-03")
    c = g.items["book-03"].clearance_id
    with pytest.raises(NotCleared, match="clearance_mismatch"):
        g.take("book-03", c + 100)
    assert g.items["book-03"].state == RELEASED  # 拒否されても状態は変わらない
    assert g.take("book-03", c) == "shelf"


def test_shelf_full_holds_shelf_books_but_discard_proceeds(tmp_path):
    _, g = make(tmp_path, shelf_capacity=1)
    ready(g, "book-03")  # 棚（1 冊目: 入る）
    ready(g, "book-05")  # 棚（2 冊目: 満杯）
    ready(g, "book-04")  # 廃棄
    assert g.items["book-05"].state == HELD and g.items["book-05"].hold_reason == "shelf_full"
    tray = {"book-03", "book-05", "book-04"}
    assert FakeRobotB(g, tray).cycle() == ["book-03", "book-04"]
    assert tray == {"book-05"}


def test_delivery_failure_is_held_and_counted(tmp_path):
    _, g = make(tmp_path)
    ready(g, "book-03")
    it = g.items["book-03"]
    g.take("book-03", it.clearance_id)
    g.on_delivered("book-03", False)
    assert it.state == HELD and it.hold_reason == "delivery_failed" and g.shelf_used == 0
    ready(g, "book-05")
    g.take("book-05", g.items["book-05"].clearance_id)
    g.on_delivered("book-05", True)
    assert g.items["book-05"].state == DELIVERED and g.shelf_used == 1


def test_tray_slots(tmp_path):
    _, g = make(tmp_path, n_slots=2)
    ready(g, "book-03")
    ready(g, "book-04", appraised=False)
    with pytest.raises(RuntimeError, match="満杯"):
        g.place("book-05")
    g.take("book-03", g.items["book-03"].clearance_id)  # 持っていけばスロットが空く
    assert g.place("book-05").slot == g.items["book-03"].slot


def test_every_step_is_on_the_bus(tmp_path):
    bus, g = make(tmp_path)
    ready(g, "book-03", appraised=False)
    g.authorize("book-03", None)
    g.on_appraisal("book-03", appraise(isbn("book-03"), CAT))
    g.take("book-03", g.items["book-03"].clearance_id)
    bus.close()
    seq = [m["topic"] for m in read_log(tmp_path / "bus.jsonl")]
    assert seq == ["tray.placed", "tray.scanned", "pick.request", "pick.denied", "release", "pick.request", "pick.cleared",
                   "tray.picked"]


def test_observer_reports_books_left_on_tray(tmp_path):
    from pai_book_intake.observer import Observer

    bus, g = make(tmp_path)
    obs = Observer(bus)
    ready(g, "book-03")
    ready(g, "book-05", appraised=False)
    g.take("book-03", g.items["book-03"].clearance_id)
    g.authorize("book-05", None)  # 許可の無い本を持っていこうとした
    md = obs.report_markdown(0)
    assert "査定台に残った本: 1 冊、許可の無い本を持っていこうとして断られた回数: 1" in md
    assert "| book-05 | scanned |" in md and "book-03 |" not in md.split("査定台に残った本（")[1]


def test_place_at_the_slot_a_chose(tmp_path):
    """A は取りに行く前に置き場所を選ぶ。並行のときは、A が運んでいる間に B が別の本を持っていき、前のスロットが
    空く。台帳には A が実際に置いたスロットを記録する（p2_line 20261028T160151-8c1914 で book-04 を book-10 の上に
    重ねた）"""
    _, g = make(tmp_path, n_slots=2)
    ready(g, "book-03")
    chosen = g.free_slot()  # A が book-10 を取りに行く前に選んだ場所
    assert chosen == 1
    g.take("book-03", g.items["book-03"].clearance_id)  # その間に B が book-03 を持っていく
    assert g.place("book-10", slot=chosen).slot == 1
    assert g.free_slot() == 0  # 次の本は、book-10 の上ではなく空いた方へ
    with pytest.raises(RuntimeError, match="ふさがって"):
        g.place("book-04", slot=1)


# ---------- 保留棚（02_確認事項 B14） ----------

def unreadable(g: ReleaseGate, book_id: str = "book-03") -> None:
    g.place(book_id)
    g.on_scan(book_id, None)
    g.on_escalated(book_id, "barcode_unreadable")


def test_unreadable_book_can_be_sent_to_the_hold_table(tmp_path):
    """読めない本（held）は release_to_hold で保留棚行きの許可が出て、B が持っていける。届いたら delivered"""
    bus, g = make(tmp_path)
    unreadable(g)
    assert [it.book_id for it in g.holdable()] == ["book-03"]
    assert g.pickable() == []  # 保留棚へ送ると決めるまでは持っていけない
    it = g.release_to_hold("book-03")
    assert it.state == RELEASED and it.destination == "hold" and it.hold_cause == "barcode_unreadable"
    assert g.holdable() == []
    assert g.take("book-03", it.clearance_id) == "hold"
    g.on_delivered("book-03", True)
    assert g.items["book-03"].state == DELIVERED and g.shelf_used == 0
    assert g.free_slot() == 0  # 台の置き場所が空く
    rel = [m for m in read_log(tmp_path / "bus.jsonl") if m["topic"] == "release"]
    assert rel[-1]["payload"]["destination"] == "hold" and rel[-1]["payload"]["reason"] == "barcode_unreadable"


@pytest.mark.parametrize("reason", ["isbn_not_in_catalog", "isbn_mismatch"])
def test_unknown_or_mismatched_isbn_goes_to_the_hold_table(tmp_path, reason):
    bus, g = make(tmp_path)
    g.place("book-03")
    g.on_scan("book-03", "9784000000017" if reason == "isbn_not_in_catalog" else isbn("book-05"))
    assert g.items["book-03"].state == HELD and g.items["book-03"].hold_reason == reason
    assert g.release_to_hold("book-03").destination == "hold"


def test_hold_release_is_not_revoked_by_later_scan_or_appraisal(tmp_path):
    """保留棚行きの許可は、あとから届く読み取りや査定の結果で取り消さない"""
    bus, g = make(tmp_path)
    unreadable(g)
    it = g.release_to_hold("book-03")
    g.on_scan("book-03", None)
    g.on_appraisal("book-03", None)
    assert it.state == RELEASED and it.destination == "hold"
    assert g.authorize("book-03", it.clearance_id).ok


def test_shelf_full_book_is_not_sent_to_the_hold_table(tmp_path):
    """本棚が満杯で台に残した本（shelf_full）は、保留棚に載せない（B14-3）"""
    bus, g = make(tmp_path, shelf_capacity=0)
    ready(g, "book-03")
    assert g.items["book-03"].hold_reason == "shelf_full"
    assert g.holdable() == []
    with pytest.raises(ValueError):
        g.release_to_hold("book-03")


def test_book_that_cannot_be_stacked_stays_on_the_tray(tmp_path):
    """保留棚に積めないときは hold_shelf_full で台に残す。もう保留棚へは送れない"""
    bus, g = make(tmp_path)
    unreadable(g)
    g.on_escalated("book-03", "hold_shelf_full")
    it = g.items["book-03"]
    assert it.state == HELD and it.hold_reason == "hold_shelf_full"
    assert g.holdable() == [] and g.remaining() == [it]
    with pytest.raises(ValueError):
        g.release_to_hold("book-03")


def test_hold_delivery_failure_leaves_the_book_held(tmp_path):
    bus, g = make(tmp_path)
    unreadable(g)
    it = g.release_to_hold("book-03")
    g.take("book-03", it.clearance_id)
    g.on_delivered("book-03", False)
    assert it.state == HELD and it.hold_reason == "delivery_failed"


def test_outcome_after_appraisal(tmp_path):
    """appraisal.done で A へ知らせる行き先: 運んでよい本は shelf・discard、査定できなかった本は保留棚があれば hold・
    無ければ tray、本棚が満杯の本は保留棚があっても tray（B14-3）"""
    _, g = make(tmp_path, shelf_capacity=1)
    ready(g, "book-03")  # 棚
    ready(g, "book-05")  # 棚（満杯）
    ready(g, "book-04")  # 廃棄
    unreadable(g, "book-06")
    assert g.outcome("book-03", hold_table=True) == ("shelf", None)
    assert g.outcome("book-04", hold_table=True) == ("discard", None)
    assert g.outcome("book-05", hold_table=True) == ("tray", "shelf_full")
    assert g.outcome("book-06", hold_table=True) == ("hold", "barcode_unreadable")
    assert g.outcome("book-06", hold_table=False) == ("tray", "barcode_unreadable")


def test_outcome_before_appraisal_is_not_decided(tmp_path):
    _, g = make(tmp_path)
    ready(g, "book-03", appraised=False)
    assert g.outcome("book-03", hold_table=True) == (None, "scanned")
