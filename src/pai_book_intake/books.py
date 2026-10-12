"""本の仕様（大きさ・重さ・表面・ISBN・査定の入力）と、カタログ `assets/catalog/books_v1.json` の読み込み。

カタログ（別作業で作成。中身は assets/catalog/README.md）は入れ子の形をしており、ここで平らな BookSpec に写す
  id                              → book_id
  bib.title                       → title
  isbn13                          → isbn13
  physical.format_ja              → format_name
  physical.dims_mm.{w, h, t}      → width_m / height_m / thickness_m（mm → m）
  physical.weight_g_true          → weight_kg（物理の質量に使う。架空の「本当の重さ」）
  physical.weight_g_estimated     → weight_estimated_kg（つかむ前の判断＝把持力の選択に使う）
  physical.surface                → surface（smooth / rough）
  appraisal.condition             → condition（A〜D）
  appraisal.buyback_yen           → buyback_yen（架空の買取価格）
  appraisal.expected_decision     → expected_decision（正解の仕分け shelf / discard）
  decision_rule.shelf_threshold_yen → Catalog.shelf_threshold_yen
JSON も tools/ も変えない。形が合わなければ KeyError / ValueError で止める（黙って直さない）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# 本の表面ごとの性質（摩擦係数・吸着力の上限 N・表紙の色）。値は試行2・3で使ったもの
SURFACES = {
    "smooth": {"friction": 0.4, "suction_max_n": 12.0, "color": (0.75, 0.25, 0.25)},
    "rough": {"friction": 0.9, "suction_max_n": 3.0, "color": (0.45, 0.55, 0.30)},
}
CONDITIONS = ("A", "B", "C", "D")
DECISIONS = ("shelf", "discard")
GRAVITY = 9.81
DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "assets" / "catalog" / "books_v1.json"


def ean13_check_digit(body12: str) -> str:
    if len(body12) != 12 or not body12.isdigit():
        raise ValueError(f"EAN-13 の本体は 12 桁の数字: {body12!r}")
    s = sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(body12))
    return str((10 - s % 10) % 10)


def ean13(body12: str) -> str:
    return body12 + ean13_check_digit(body12)


def is_valid_isbn13(code: str) -> bool:
    return len(code) == 13 and code.isdigit() and code[:3] in ("978", "979") and ean13_check_digit(code[:12]) == code[12]


@dataclass(frozen=True)
class BookSpec:
    book_id: str
    title: str
    isbn13: str
    format_name: str  # 判型
    width_m: float
    height_m: float
    thickness_m: float
    weight_kg: float  # 本当の重さ（物理の質量）
    weight_estimated_kg: float  # つかむ前に判型とページ数から出した推定（把持力の選択に使う）
    surface: str
    condition: str = "B"  # A 美品 / B 良好 / C 可 / D 難あり
    buyback_yen: int = 0  # 架空の買取価格
    expected_decision: str | None = None  # 正解の仕分け（評価用。仕分けの判断には使わない）

    def __post_init__(self) -> None:
        if not is_valid_isbn13(self.isbn13):
            raise ValueError(f"{self.book_id}: ISBN-13 が不正: {self.isbn13!r}")
        if self.surface not in SURFACES:
            raise ValueError(f"{self.book_id}: surface は {list(SURFACES)} のどれか: {self.surface!r}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"{self.book_id}: condition は {CONDITIONS} のどれか: {self.condition!r}")
        if self.expected_decision is not None and self.expected_decision not in DECISIONS:
            raise ValueError(f"{self.book_id}: expected_decision は {DECISIONS} のどれか: {self.expected_decision!r}")
        for name in ("width_m", "height_m", "thickness_m", "weight_kg", "weight_estimated_kg"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{self.book_id}: {name} は正の値: {getattr(self, name)!r}")
        if self.buyback_yen < 0:
            raise ValueError(f"{self.book_id}: buyback_yen は 0 以上: {self.buyback_yen!r}")

    @property
    def size_m(self) -> tuple[float, float, float]:
        """シミュレーションの箱の寸法（本の座標: x=横幅, y=縦（天地）, z=厚み）。"""
        return (self.width_m, self.height_m, self.thickness_m)

    @property
    def volume_m3(self) -> float:
        return self.width_m * self.height_m * self.thickness_m

    @property
    def density(self) -> float:
        return self.weight_kg / self.volume_m3

    def to_dict(self) -> dict:
        return asdict(self)


def book_from_catalog_entry(d: dict) -> BookSpec:
    phys, appr, dims = d["physical"], d["appraisal"], d["physical"]["dims_mm"]
    return BookSpec(
        book_id=str(d["id"]), title=str(d["bib"]["title"]), isbn13=str(d["isbn13"]),
        format_name=str(phys["format_ja"]),
        width_m=float(dims["w"]) / 1000, height_m=float(dims["h"]) / 1000, thickness_m=float(dims["t"]) / 1000,
        weight_kg=float(phys["weight_g_true"]) / 1000, weight_estimated_kg=float(phys["weight_g_estimated"]) / 1000,
        surface=str(phys["surface"]), condition=str(appr["condition"]), buyback_yen=int(appr["buyback_yen"]),
        expected_decision=str(appr["expected_decision"]),
    )


@dataclass(frozen=True)
class Catalog:
    books: tuple[BookSpec, ...]
    shelf_threshold_yen: int

    def by_id(self, book_id: str) -> BookSpec:
        for b in self.books:
            if b.book_id == book_id:
                return b
        raise KeyError(book_id)

    def by_isbn(self, isbn: str) -> BookSpec | None:
        for b in self.books:
            if b.isbn13 == isbn:
                return b
        return None


def load_catalog(path: Path = DEFAULT_CATALOG) -> Catalog:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    books = tuple(book_from_catalog_entry(r) for r in data["books"])
    ids = [b.book_id for b in books]
    if len(set(ids)) != len(ids):
        raise ValueError(f"id が重複している: {ids}")
    threshold = int(data["decision_rule"]["shelf_threshold_yen"])
    for b in books:  # 正解の仕分けが、規則（buyback_yen >= しきい値 → shelf）と食い違っていないか
        rule = "shelf" if b.buyback_yen >= threshold else "discard"
        if b.expected_decision != rule:
            raise ValueError(f"{b.book_id}: expected_decision={b.expected_decision} が規則の結果 {rule} と違う")
    return Catalog(books, threshold)


def _ph(i: int, fmt: str, w: float, h: float, t: float, surface: str, cond: str, buyback: int) -> BookSpec:
    # 仮の本（カタログが無いとき・テスト用）。重さは密度 1000 kg/m^3（試行2・3 と同じ）、推定も同じ値
    m = round(w * h * t * 1000.0, 4)
    return BookSpec(book_id=f"ph{i:02d}", title=f"仮の本 {i:02d}", isbn13=ean13(f"9784000000{i:02d}"),
                    format_name=fmt, width_m=w, height_m=h, thickness_m=t, weight_kg=m, weight_estimated_kg=m,
                    surface=surface, condition=cond, buyback_yen=buyback,
                    expected_decision="shelf" if buyback >= 50 else "discard")


# 仮のカタログ。ph01 は試行2・3 で成功を確かめた大きさ
PLACEHOLDER_CATALOG = Catalog((
    _ph(1, "四六判(試行と同寸)", 0.130, 0.190, 0.020, "smooth", "A", 160),
    _ph(2, "四六判(試行と同寸)", 0.130, 0.190, 0.020, "rough", "B", 120),
    _ph(3, "B6判", 0.128, 0.182, 0.018, "smooth", "D", 0),
    _ph(4, "文庫", 0.105, 0.148, 0.015, "smooth", "B", 40),
    _ph(5, "新書", 0.103, 0.182, 0.012, "rough", "A", 90),
    _ph(6, "四六判(試行と同寸)", 0.130, 0.190, 0.020, "smooth", "A", 180),
), shelf_threshold_yen=50)
PLACEHOLDER_BOOKS = list(PLACEHOLDER_CATALOG.books)


def grip_plan(spec: BookSpec) -> dict:
    """つかむ前の判断（推定重さだけを使う。本当の重さは持ち上げた後にしか分からない）。
    - 把持力の上限: ざらざらの本は 25 N（試行3。本を傷めないよう抑える）、滑らかな本は 60 N（試行2 の上限）
    - 吸着だけで本を浮かせられるか: 推定の重さ × g が吸着力の上限未満か（足りなければ flat_pick の前提が崩れる）"""
    s = SURFACES[spec.surface]
    weight_n = spec.weight_estimated_kg * GRAVITY
    return {"grip_limit_n": 25.0 if spec.surface == "rough" else 60.0, "suction_max_n": s["suction_max_n"],
            "est_weight_n": round(weight_n, 3), "suction_can_lift": weight_n < s["suction_max_n"]}
