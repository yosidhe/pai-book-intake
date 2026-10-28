"""査定台から本棚・廃棄箱へ運んでよいかの判定（運搬許可）。Genesis に依存しない。

査定台には、ロボット A が箱から出した本が順に並んでいく（スロットごとに 1 冊）。ロボット B は移動が多いので、
台の上の本を「運んでよい」と許可が出たものから、許可が出た順に運ぶ。許可が出ていない本には手を出さない。

本ごとの状態（TrayItem.state）
  placed     台に置かれた（まだバーコードを読んでいない）
  scanned    バーコードを読めた（カタログにある本と分かった）。査定待ち
  released   運んでよい（許可番号 clearance_id と行き先が付く）
  held       運ばない（人の判断を待つ）。理由は hold_reason
  picked     B が許可を示して持っていった
  delivered  行き先に届いた

許可（released）の条件。1 つでも欠けたら許可しない
  1. バーコードが読めていて、その ISBN がカタログにある
  2. 裏返しの依頼が終わっている（裏返し待ちではない）
  3. 査定の結果があり、その本（book_id）が読めた ISBN の本と一致する
  4. 行き先が shelf か discard（査定が行き先を決められなかった本は運ばない）
  5. 行き先が本棚なら、本棚に空きがある（許可済みでまだ届いていない本の分も数える）。無ければ held（満杯）

B は運ぶ直前に authorize(book_id, clearance_id) で確かめる。許可が無い・番号が違う・取り消されたときは
pick.denied を出して、その本には触れない（take は例外を投げる）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .appraisal import Decision
from .books import Catalog
from .bus import Bus

TASK = "appraisal"  # 運搬許可を出すのは査定台（査定の工程）
SORTER = "robot-b-sort"

PLACED, SCANNED, RELEASED, HELD, PICKED, DELIVERED = "placed", "scanned", "released", "held", "picked", "delivered"
DESTINATIONS = ("shelf", "discard")
# 待っても直らない理由（held にして人の判断を待つ）。それ以外（査定待ち・裏返し待ち・未読）は待てば進みうる
HOLD_REASONS = ("shelf_full", "isbn_not_in_catalog", "isbn_mismatch", "appraisal_for_other_book")


class NotCleared(RuntimeError):
    """許可の無い本を持っていこうとした。"""


@dataclass
class TrayItem:
    book_id: str
    slot: int
    placed_seq: int
    state: str = PLACED
    isbn: str | None = None
    flip_pending: bool = False
    decision: Decision | None = None
    clearance_id: int | None = None
    released_seq: int | None = None
    destination: str | None = None
    hold_reason: str | None = None


@dataclass(frozen=True)
class Clearance:
    ok: bool
    book_id: str
    clearance_id: int | None
    destination: str | None
    reason: str


@dataclass
class ReleaseGate:
    """査定台 1 つ（環境 1 つ）ぶんの運搬許可。"""

    bus: Bus
    catalog: Catalog
    n_slots: int
    shelf_capacity: int
    env: int = 0
    items: dict[str, TrayItem] = field(default_factory=dict)
    shelf_used: int = 0
    _seq: int = 0
    _clearance: int = 0

    # ---- 台に置く・読む・裏返す・査定する ----

    def place(self, book_id: str, slot: int | None = None) -> TrayItem:
        """台に置いた本を記録する。slot は A が実際に置いた場所（省略すると空いている最初の場所）。
        A は取りに行く前に場所を選ぶので、並行のときはその間に別の場所が空くことがある。選び直さずに記録する"""
        if book_id in self.items and self.items[book_id].state not in (DELIVERED,):
            raise ValueError(f"{book_id} はもう台の上にある")
        if slot is None:
            slot = self.free_slot()
            if slot is None:
                raise RuntimeError("査定台が満杯（空いたスロットが無い）")
        elif slot in self._used_slots():
            raise RuntimeError(f"スロット {slot} はふさがっている")
        self._seq += 1
        it = self.items[book_id] = TrayItem(book_id, slot, self._seq)
        self._event("tray.placed", it)
        return it

    def _used_slots(self) -> set[int]:
        return {it.slot for it in self.items.values() if it.state not in (PICKED, DELIVERED)}

    def free_slot(self) -> int | None:
        used = self._used_slots()
        return next((s for s in range(self.n_slots) if s not in used), None)

    def on_scan(self, book_id: str, isbn: str | None) -> None:
        """バーコードを読んだ結果。読めなければ None（裏返しや人の判断は検品の側で扱う）。"""
        it = self._get(book_id)
        it.isbn = isbn
        b = self.catalog.by_isbn(isbn) if isbn else None
        if b is not None and b.book_id == book_id and it.state == PLACED:
            it.state = SCANNED
            self._event("tray.scanned", it)
        self._evaluate(it)

    def on_flip_requested(self, book_id: str) -> None:
        it = self._get(book_id)
        it.flip_pending = True
        self._revoke(it, "flip_requested")

    def on_flip_done(self, book_id: str) -> None:
        self._get(book_id).flip_pending = False

    def on_escalated(self, book_id: str, reason: str) -> None:
        """検品で読めないまま上げた本。人が判断するまで運ばない。"""
        it = self._get(book_id)
        self._hold(it, reason)

    def on_appraisal(self, book_id: str, decision: Decision | None) -> None:
        """査定の結果（まだ出ていなければ呼ばれない／None）。"""
        it = self._get(book_id)
        it.decision = decision
        self._evaluate(it)

    # ---- B から ----

    def pickable(self) -> list[TrayItem]:
        """運んでよい本を、許可が出た順に返す。"""
        return sorted((it for it in self.items.values() if it.state == RELEASED), key=lambda it: it.released_seq)

    def authorize(self, book_id: str, clearance_id: int | None) -> Clearance:
        """B が持っていく直前の確認。許可が無ければ pick.denied を出す（B はその本に触れない）。"""
        self.bus.publish(SORTER, TASK, "event", "pick.request",
                         {"env": self.env, "book_id": book_id, "clearance_id": clearance_id})
        it = self.items.get(book_id)
        if it is None:
            return self._deny(book_id, clearance_id, "not_on_tray")
        if it.state != RELEASED:
            return self._deny(book_id, clearance_id, f"not_released:{it.state}"
                              + (f":{it.hold_reason}" if it.hold_reason else ""))
        if clearance_id != it.clearance_id:
            return self._deny(book_id, clearance_id, "clearance_mismatch")
        c = Clearance(True, book_id, it.clearance_id, it.destination, "cleared")
        self.bus.publish(TASK, SORTER, "event", "pick.cleared",
                         {"env": self.env, "book_id": book_id, "clearance_id": it.clearance_id,
                          "destination": it.destination})
        return c

    def take(self, book_id: str, clearance_id: int | None) -> str:
        """B が持っていった。許可が無ければ NotCleared（呼んだ側の誤り。運んではいけない）。戻り値は行き先。"""
        c = self.authorize(book_id, clearance_id)
        if not c.ok:
            raise NotCleared(f"{book_id}: {c.reason}")
        it = self.items[book_id]
        it.state = PICKED
        self._event("tray.picked", it)
        return c.destination

    def on_delivered(self, book_id: str, delivered: bool) -> None:
        it = self._get(book_id)
        if it.state != PICKED:
            raise RuntimeError(f"{book_id}: 持っていっていない本の到着（状態 {it.state}）")
        it.state = DELIVERED if delivered else HELD
        if not delivered:
            it.hold_reason = "delivery_failed"
        elif it.destination == "shelf":
            self.shelf_used += 1
        self._event("tray.delivered" if delivered else "tray.delivery_failed", it)

    def remaining(self) -> list[TrayItem]:
        """台に残っている本（運ばなかった本）。"""
        return [it for it in self.items.values() if it.state in (PLACED, SCANNED, RELEASED, HELD)]

    # ---- 内部 ----

    def _get(self, book_id: str) -> TrayItem:
        it = self.items.get(book_id)
        if it is None:
            raise KeyError(f"{book_id} は台の上に無い")
        return it

    def _shelf_reserved(self) -> int:
        return sum(1 for it in self.items.values() if it.destination == "shelf" and it.state in (RELEASED, PICKED))

    def _evaluate(self, it: TrayItem) -> None:
        if it.state in (PICKED, DELIVERED, HELD):
            return
        reason = self._why_not(it)
        if reason is None:
            if it.state != RELEASED:
                self._clearance += 1
                self._seq += 1
                it.state, it.clearance_id, it.released_seq = RELEASED, self._clearance, self._seq
                it.destination, it.hold_reason = it.decision.destination, None
                self.bus.publish(TASK, SORTER, "event", "release",
                                 {"env": self.env, "book_id": it.book_id, "slot": it.slot,
                                  "clearance_id": it.clearance_id, "destination": it.destination,
                                  "value_jpy": it.decision.value_jpy})
        elif reason in HOLD_REASONS:  # 待っても直らない（人の判断が要る）
            self._hold(it, reason)
        else:
            self._revoke(it, reason)

    def _why_not(self, it: TrayItem) -> str | None:
        if it.isbn is None:
            return "barcode_not_read"
        b = self.catalog.by_isbn(it.isbn)
        if b is None:
            return "isbn_not_in_catalog"
        if b.book_id != it.book_id:
            return "isbn_mismatch"
        if it.flip_pending:
            return "flip_pending"
        d = it.decision
        if d is None:
            return "appraisal_pending"
        if d.book_id != it.book_id:
            return "appraisal_for_other_book"
        if d.destination not in DESTINATIONS:
            return f"no_destination:{d.destination}"
        if d.destination == "shelf" and it.state != RELEASED \
                and self.shelf_used + self._shelf_reserved() >= self.shelf_capacity:
            return "shelf_full"
        return None

    def _revoke(self, it: TrayItem, reason: str) -> None:
        """許可を取り消す（まだ持っていかれていない場合だけ）。状態は読めたかどうかで戻す。"""
        if it.state == RELEASED:
            self.bus.publish(TASK, SORTER, "event", "release.revoked",
                             {"env": self.env, "book_id": it.book_id, "clearance_id": it.clearance_id,
                              "reason": reason})
            it.clearance_id, it.released_seq, it.destination = None, None, None
        if it.state in (RELEASED, SCANNED, PLACED):
            b = self.catalog.by_isbn(it.isbn) if it.isbn else None
            it.state = SCANNED if b is not None and b.book_id == it.book_id else PLACED

    def _hold(self, it: TrayItem, reason: str) -> None:
        self._revoke(it, reason)
        it.state, it.hold_reason = HELD, reason
        self.bus.publish(TASK, "observer", "event", "tray.held",
                         {"env": self.env, "book_id": it.book_id, "slot": it.slot, "reason": reason})

    def _deny(self, book_id: str, clearance_id: int | None, reason: str) -> Clearance:
        self.bus.publish(TASK, SORTER, "event", "pick.denied",
                         {"env": self.env, "book_id": book_id, "clearance_id": clearance_id, "reason": reason})
        return Clearance(False, book_id, None, None, reason)

    def _event(self, topic: str, it: TrayItem) -> None:
        self.bus.publish(TASK, "observer", "event", topic,
                         {"env": self.env, "book_id": it.book_id, "slot": it.slot, "state": it.state})
