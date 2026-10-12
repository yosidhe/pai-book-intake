"""書籍カタログ assets/catalog/books_v1.json の検査。

実行: docker compose run --rm sim python -m pytest tests/test_catalog.py -q
NDL への通信はしない（保存済みの JSON と手書きの YAML だけを見る）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import catalog_model as M  # noqa: E402

CATALOG = json.loads((ROOT / "assets" / "catalog" / "books_v1.json").read_text(encoding="utf-8"))
PRICING = yaml.safe_load((ROOT / "assets" / "catalog" / "pricing_v1.yaml").read_text(encoding="utf-8"))
BOOKS = CATALOG["books"]


def test_ten_books_unique():
    assert len(BOOKS) == 10
    assert len({b["id"] for b in BOOKS}) == 10
    assert len({b["isbn13"] for b in BOOKS}) == 10


@pytest.mark.parametrize("book", BOOKS, ids=lambda b: b["id"])
def test_isbn13_check_digit(book):
    isbn = book["isbn13"]
    assert re.fullmatch(r"97[89]\d{10}", isbn)
    # チェックディジットを catalog_model とは別に、ここで直接計算する
    s = sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(isbn[:12]))
    assert int(isbn[12]) == (10 - s % 10) % 10
    assert book["ean13"] == isbn
    # NDL に記録された ISBN（10 桁の場合もある）が同じ本を指す
    assert M.to_isbn13(book["bib"]["isbn_as_recorded"]) == isbn


def test_isbn_conversion_and_rejection():
    assert M.to_isbn13("4-00-310101-4") == "9784003101018"
    assert M.to_isbn13("480749905X") == "9784807499052"
    with pytest.raises(ValueError):
        M.to_isbn13("4-00-310101-5")  # チェックディジット違い
    with pytest.raises(ValueError):
        M.to_isbn13("9784003101019")


@pytest.mark.parametrize("book", BOOKS, ids=lambda b: b["id"])
def test_decision_rule_reproduces_expected(book):
    rule = CATALOG["decision_rule"]
    a = book["appraisal"]
    got = M.appraise(a["base_buyback_yen"], a["condition"], rule)
    assert got["buyback_yen"] == a["buyback_yen"]
    assert got["decision"] == a["expected_decision"]
    # 規則を式どおりに直接適用しても同じ
    direct = round(a["base_buyback_yen"] * rule["condition_factor"][a["condition"]])
    assert ("shelf" if direct >= rule["shelf_threshold_yen"] else "discard") == a["expected_decision"]


def test_catalog_matches_handwritten_pricing():
    by_id = {p["id"]: p for p in PRICING["books"]}
    assert CATALOG["decision_rule"]["shelf_threshold_yen"] == PRICING["decision_rule"]["shelf_threshold_yen"]
    assert CATALOG["decision_rule"]["condition_factor"] == PRICING["decision_rule"]["condition_factor"]
    for b in BOOKS:
        p = by_id[b["id"]]
        assert M.to_isbn13(p["isbn"]) == b["isbn13"]
        assert p["base_buyback_yen"] == b["appraisal"]["base_buyback_yen"]
        assert p["condition"] == b["appraisal"]["condition"]
        assert p["expected_decision"] == b["appraisal"]["expected_decision"]
        assert p["surface"] == b["physical"]["surface"]


def test_mix_of_decisions_formats_surfaces():
    decisions = [b["appraisal"]["expected_decision"] for b in BOOKS]
    assert decisions.count("shelf") == 6 and decisions.count("discard") == 4
    assert any(b["appraisal"]["borderline"] for b in BOOKS)
    formats = {b["physical"]["format"] for b in BOOKS}
    assert {"bunko", "shinsho"} <= formats
    assert formats & {"shiroku", "shiroku_hard"}
    assert formats & {"a5", "b5"}
    assert {b["physical"]["surface"] for b in BOOKS} == {"smooth", "rough"}


def test_weight_model_in_json_is_the_code_model():
    assert CATALOG["weight_model"] == json.loads(json.dumps(M.WEIGHT_MODEL))


@pytest.mark.parametrize("book", BOOKS, ids=lambda b: b["id"])
def test_weight_estimate_deterministic(book):
    ph = book["physical"]
    pages, height = M.parse_extent(book["bib"]["extent"])
    assert (pages, height) == (ph["pages"], ph["height_cm_ndl"])
    r1 = M.estimate_dims_weight(pages, height)
    r2 = M.estimate_dims_weight(pages, height)
    assert r1 == r2
    assert r1["weight_g_estimated"] == ph["weight_g_estimated"]
    assert r1["dims_mm"] == ph["dims_mm"]


def test_weight_estimate_hand_calculation():
    # 吾輩は猫である（岩波文庫, 563p, 15cm）を手で計算した値と一致すること
    # 紙: ceil(563/2)=282 枚 × 0.075mm = 21.15mm, 148×105×21.15/1000 cm^3 × 0.75 = 246.50 g
    # 表紙: 2 × 0.148×0.105 m^2 × 250 g/m^2 = 7.77 g → 254.27 g
    r = M.estimate_dims_weight(563, 15)
    assert r["weight_g_estimated"] == pytest.approx(254.3, abs=0.05)
    assert r["dims_mm"] == {"h": 148, "w": 105, "t": 21.8}


def test_parse_extent_variants():
    assert M.parse_extent("563p ; 15cm") == (563, 15)
    assert M.parse_extent("416p ; 20cm + 8p") == (416, 20)
    assert M.parse_extent("268, 9p ; 17cm") == (277, 17)


@pytest.mark.parametrize("book", BOOKS, ids=lambda b: b["id"])
def test_true_weight_differs_by_5_to_15_percent(book):
    ph = book["physical"]
    ratio = ph["weight_g_true"] / ph["weight_g_estimated"] - 1.0
    assert 0.045 <= abs(ratio) <= 0.155  # 0.1 g 単位の丸めの分だけ余裕
    assert ph["weight_g_true"] == M.true_weight(ph["weight_g_estimated"], ph["weight_true_error"])


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def test_no_absolute_paths_in_json():
    abs_patterns = [
        re.compile(r"(^|[\s\"'(])[A-Za-z]:[\\/]"),  # Windows のドライブ
        re.compile(r"\\\\"),  # UNC / バックスラッシュ区切り
        re.compile(r"(^|[\s\"'(])/(home|Users|root|workspace|mnt|tmp|var)(/|$)"),
        re.compile(r"OneDrive", re.IGNORECASE),
    ]
    for s in _strings(CATALOG):
        if re.match(r"^https?://", s):
            continue
        for pat in abs_patterns:
            assert not pat.search(s), f"絶対パスらしき文字列: {s!r}"
    prov = CATALOG["provenance"]
    assert not Path(prov["generatedBy"]).is_absolute()
    for p in prov["inputs"]:
        assert not Path(p).is_absolute()


def test_provenance_block():
    prov = CATALOG["provenance"]
    for key in ("artifactId", "generatedAt", "generatedBy", "sources", "notes"):
        assert prov[key]
    assert prov["generatedBy"] == "tools/build_catalog.py"
    ndl = prov["sources"][0]
    assert ndl["endpoint"].startswith("https://ndlsearch.ndl.go.jp/")
    assert "CC BY 4.0" in ndl["license"]
    assert "国立国会図書館" in ndl["attribution_text"]
