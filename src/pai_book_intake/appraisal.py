"""査定（スタブ）。読めた ISBN をカタログと照らし、本棚か廃棄箱かと、架空の買取価格を返す。

規則はカタログの decision_rule をそのまま再現する: buyback_yen >= shelf_threshold_yen（50 円）なら shelf
TODO(P2 以降): 状態（A〜D）を画像から判定する。今はカタログの値（架空）をそのまま使う
"""

from __future__ import annotations

from dataclasses import dataclass

from .books import Catalog


@dataclass(frozen=True)
class Decision:
    destination: str  # "shelf" / "discard"
    value_jpy: int
    reason: str
    book_id: str | None


def appraise(isbn: str | None, catalog: Catalog) -> Decision:
    if isbn is None:
        return Decision("discard", 0, "isbn_unread", None)
    b = catalog.by_isbn(isbn)
    if b is None:
        return Decision("discard", 0, "isbn_not_in_catalog", None)
    th = catalog.shelf_threshold_yen
    if b.buyback_yen >= th:
        return Decision("shelf", b.buyback_yen, f"buyback {b.buyback_yen}>={th} (condition={b.condition})", b.book_id)
    return Decision("discard", b.buyback_yen, f"buyback {b.buyback_yen}<{th} (condition={b.condition})", b.book_id)
