"""書籍カタログの計算規則（ISBN・判型・寸法・推定重さ・査定）。

build_catalog.py とテストの両方がここを使う。数値だけでなく式もここに置き、
ライン側（把持力の決定・査定）は books_v1.json の `weight_model` / `decision_rule` から同じ計算を再現できる。
標準ライブラリだけで書く。
"""

from __future__ import annotations

import math
import re

# ---------------------------------------------------------------- ISBN


def _digits(isbn: str) -> str:
    return re.sub(r"[^0-9Xx]", "", isbn).upper()


def isbn10_check(body9: str) -> str:
    s = sum((10 - i) * int(c) for i, c in enumerate(body9))
    r = (11 - s % 11) % 11
    return "X" if r == 10 else str(r)


def isbn13_check(body12: str) -> str:
    s = sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(body12))
    return str((10 - s % 10) % 10)


def is_valid_isbn10(isbn: str) -> bool:
    d = _digits(isbn)
    return bool(re.fullmatch(r"\d{9}[\dX]", d)) and isbn10_check(d[:9]) == d[9]


def is_valid_isbn13(isbn: str) -> bool:
    d = _digits(isbn)
    return bool(re.fullmatch(r"97[89]\d{10}", d)) and isbn13_check(d[:12]) == d[12]


def to_isbn13(isbn: str) -> str:
    """ISBN-10 / ISBN-13（ハイフン可）を検査して ISBN-13 の数字 13 桁にする。不正なら ValueError。"""
    d = _digits(isbn)
    if len(d) == 13:
        if not is_valid_isbn13(d):
            raise ValueError(f"ISBN-13 のチェックディジットが合わない: {isbn}")
        return d
    if len(d) == 10:
        if not is_valid_isbn10(d):
            raise ValueError(f"ISBN-10 のチェックディジットが合わない: {isbn}")
        body = "978" + d[:9]
        return body + isbn13_check(body)
    raise ValueError(f"ISBN の桁数が不正: {isbn}")


# ---------------------------------------------------------------- 書誌の「形態」欄（例: '563p ; 15cm'）


def parse_extent(extent: str) -> tuple[int, int]:
    """NDL の dcterms:extent からページ数と高さ(cm)を取り出す。

    ページ数は ';' より前の「数字p」の合計（'268, 9p' → 268+9）。'+ 8p' のような付録（';' より後）は数えない。
    高さは NDL の慣行どおり cm 単位・端数切り上げの値。
    """
    head, _, tail = extent.partition(";")
    nums = [int(n) for n in re.findall(r"\d+", head)]
    if not nums:
        raise ValueError(f"ページ数が読めない: {extent}")
    m = re.search(r"(\d+)\s*cm", tail)
    if not m:
        raise ValueError(f"大きさ(cm)が読めない: {extent}")
    return sum(nums), int(m.group(1))


# ---------------------------------------------------------------- 判型（NDL の高さ cm → 寸法）

# NDL の大きさは本の高さを cm で切り上げた値。高さから一般的な判型を当てはめる（仮定）。
# 製本（並製=ソフトカバー / 上製=ハードカバー）も判型ごとの典型で決める（仮定。書誌には無い）。
FORMATS: dict[int, dict] = {
    15: {"format": "bunko", "format_ja": "文庫（A6）", "h_mm": 148, "w_mm": 105, "binding": "soft"},
    16: {"format": "bunko", "format_ja": "文庫（やや大きめ）", "h_mm": 152, "w_mm": 106, "binding": "soft"},
    18: {"format": "shinsho", "format_ja": "新書判", "h_mm": 173, "w_mm": 105, "binding": "soft"},
    19: {"format": "shiroku", "format_ja": "四六判（並製）", "h_mm": 188, "w_mm": 127, "binding": "soft"},
    20: {"format": "shiroku_hard", "format_ja": "四六判（上製）", "h_mm": 194, "w_mm": 133, "binding": "hard"},
    21: {"format": "a5", "format_ja": "A5判（上製）", "h_mm": 210, "w_mm": 148, "binding": "hard"},
    26: {"format": "b5", "format_ja": "B5判（並製）", "h_mm": 257, "w_mm": 182, "binding": "soft"},
}

# ---------------------------------------------------------------- 推定重さ（つかむ前に判型とページ数だけで出す）

WEIGHT_MODEL = {
    "version": "weight_v1",
    # 本文紙 1 枚（2 ページ）の厚さ [mm]。文庫・新書は薄い紙、それ以外は一般的な書籍用紙（仮定）
    "leaf_thickness_mm": {"bunko": 0.075, "shinsho": 0.075, "default": 0.090},
    # 表紙 1 枚の厚さ [mm]。並製は厚紙、上製は板紙（仮定）
    "cover_thickness_mm": {"soft": 0.3, "hard": 2.5},
    # 本文紙の密度 [g/cm^3]（一般的な上質紙・書籍用紙の 0.7〜0.8 の中ほど。仮定）
    "paper_density_g_cm3": 0.75,
    # 表紙 1 枚の面密度 [g/m^2]（並製=厚紙 250 / 上製=板紙＋クロス 1200。仮定）
    "cover_areal_density_g_m2": {"soft": 250, "hard": 1200},
    "formula": (
        "leaves = ceil(pages / 2); "
        "block_mm = leaves * leaf_thickness_mm[format]; "
        "t_mm = block_mm + 2 * cover_thickness_mm[binding]; "
        "weight_g = (h_mm*w_mm*block_mm/1000) * paper_density_g_cm3 "
        "+ 2 * (h_mm*w_mm/1e6) * cover_areal_density_g_m2[binding]"
    ),
}


def format_for_height(height_cm: int) -> dict:
    if height_cm not in FORMATS:
        raise ValueError(f"判型の表に無い高さ: {height_cm}cm")
    return dict(FORMATS[height_cm])


def estimate_dims_weight(pages: int, height_cm: int, model: dict = WEIGHT_MODEL) -> dict:
    """判型（NDL の高さ cm）とページ数から、寸法 [mm] と推定重さ [g] を決める。乱数なし・決定的。"""
    f = format_for_height(height_cm)
    leaf = model["leaf_thickness_mm"].get(f["format"], model["leaf_thickness_mm"]["default"])
    binding = f["binding"]
    leaves = math.ceil(pages / 2)
    block_mm = leaves * leaf
    t_mm = block_mm + 2 * model["cover_thickness_mm"][binding]
    h, w = f["h_mm"], f["w_mm"]
    body_g = (h * w * block_mm / 1000.0) * model["paper_density_g_cm3"]
    cover_g = 2 * (h * w / 1e6) * model["cover_areal_density_g_m2"][binding]
    return {
        **f,
        "dims_mm": {"h": h, "w": w, "t": round(t_mm, 1)},
        "weight_g_estimated": round(body_g + cover_g, 1),
    }


def true_weight(estimated_g: float, error: float) -> float:
    """架空の「本当の重さ」。持ち上げた後に推定と比べるためのもの。"""
    return round(estimated_g * (1.0 + error), 1)


# ---------------------------------------------------------------- 査定（架空の買取価格 × 状態）


def appraise(base_buyback_yen: int, condition: str, rule: dict) -> dict:
    """buyback_yen = round(base × 状態係数)。しきい値以上なら shelf、未満なら discard。"""
    factor = rule["condition_factor"][condition]
    buyback = int(round(base_buyback_yen * factor))
    th = rule["shelf_threshold_yen"]
    return {
        "condition_factor": factor,
        "buyback_yen": buyback,
        "decision": "shelf" if buyback >= th else "discard",
        "borderline": abs(buyback - th) <= rule["borderline_margin_yen"],
    }
