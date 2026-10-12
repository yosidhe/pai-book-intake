"""書籍カタログ assets/catalog/books_v1.json を作る。

入力:
- outputs/ndl_raw/<ISBN13>.xml と manifest.json … tools/fetch_ndl.py が保存した NDL サーチ SRU の生応答（書誌）
- assets/catalog/pricing_v1.yaml … 手書きの架空の値（買取価格・状態・表面・重さの誤差）と査定規則

検証（1 つでも外れたら書き出さずに終了コード 1）:
- ISBN のチェックディジット（NDL の ISBN-10 は ISBN-13 に変換してから再検査）
- 書誌が国立国会図書館作成（bibRecordCategory=R100000002）であること
- 査定規則が各本の expected_decision を再現すること
- 重さの誤差が ±5〜15% の範囲にあること

使い方: python tools/build_catalog.py [--check]   （--check は書き出さずに検証だけ）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog_model as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRICING = ROOT / "assets" / "catalog" / "pricing_v1.yaml"
RAW_DIR = ROOT / "outputs" / "ndl_raw"
OUT = ROOT / "assets" / "catalog" / "books_v1.json"

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "dcndl": "http://ndl.go.jp/dcndl/terms/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "srw": "http://www.loc.gov/zing/srw/",
}
RDF_ABOUT = f"{{{NS['rdf']}}}about"
RDF_DATATYPE = f"{{{NS['rdf']}}}datatype"
NDL_REPOSITORY = "R100000002"  # 国立国会図書館蔵書（国立国会図書館が作成した書誌）

ATTRIBUTION = {
    "name": "国立国会図書館サーチ（NDL サーチ）SRU API による国立国会図書館書誌データ",
    "endpoint": "https://ndlsearch.ndl.go.jp/api/sru",
    "dpid": "iss-ndl-opac",
    "license": "CC BY 4.0 互換（国立国会図書館が作成し提供する書誌データの二次利用条件）",
    "license_url": "https://creativecommons.org/licenses/by/4.0/deed.ja",
    "terms_urls": [
        "https://ndlsearch.ndl.go.jp/help/api",
        "https://ndlsearch.ndl.go.jp/help/api/provider",
        "https://www.ndl.go.jp/jp/data/data_service/quickguide/index.html",
    ],
    "attribution_text": (
        "出典：国立国会図書館書誌データ（{date}に取得）。国立国会図書館サーチのAPIを利用。"
        "題名・著者・出版者・出版年・ページ数・大きさの項目を抜き出して整形し、"
        "寸法・推定重さは大きさとページ数から本プロジェクトが計算した（国立国会図書館が作成したものではない）。"
    ),
}


def _text(el, path: str) -> str | None:
    found = el.find(path, NS)
    return found.text.strip() if found is not None and found.text else None


def parse_record(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    n = root.findtext("srw:numberOfRecords", namespaces=NS)
    if n != "1":
        raise ValueError(f"件数が 1 でない: numberOfRecords={n}")
    rdf = root.find(".//rdf:RDF", NS)
    admin = rdf.find("dcndl:BibAdminResource", NS)
    bib = next(b for b in rdf.findall("dcndl:BibResource", NS) if b.find("dcterms:title", NS) is not None)
    isbns = [e.text for e in bib.findall("dcterms:identifier", NS) if (e.get(RDF_DATATYPE) or "").endswith("/ISBN")]
    return {
        "ndl_url": admin.get(RDF_ABOUT),
        "ndl_record_category": _text(admin, "dcndl:bibRecordCategory"),
        "ndl_bib_id": next((e.text for e in bib.findall("dcterms:identifier", NS)
                            if (e.get(RDF_DATATYPE) or "").endswith("/NDLBibID")), None),
        "isbn_ndl": isbns[0] if isbns else None,
        "title": _text(bib, "dcterms:title"),
        "volume": _text(bib, "dcndl:volume/rdf:Description/rdf:value"),
        "series": _text(bib, "dcndl:seriesTitle/rdf:Description/rdf:value"),
        "creators": [e.text for e in bib.findall("dcterms:creator/foaf:Agent/foaf:name", NS)],
        "responsibility": [e.text for e in bib.findall("dc:creator", NS)],
        "publisher": _text(bib, "dcterms:publisher/foaf:Agent/foaf:name"),
        "date": _text(bib, "dcterms:date"),
        "year": int(_text(bib, "dcterms:issued")),
        "extent": _text(bib, "dcterms:extent"),
    }


def build(pricing: dict, raw_manifest: dict) -> tuple[list[dict], list[str]]:
    rule = pricing["decision_rule"]
    entries, errors = [], []
    for p in pricing["books"]:
        bid = p["id"]
        try:
            isbn13 = M.to_isbn13(p["isbn"])
        except ValueError as e:
            errors.append(f"{bid}: {e}")
            continue
        raw_path = RAW_DIR / f"{isbn13}.xml"
        if not raw_path.exists():
            errors.append(f"{bid}: 生応答が無い（先に tools/fetch_ndl.py）: {raw_path.relative_to(ROOT).as_posix()}")
            continue
        rec = parse_record(raw_path.read_bytes())
        if rec["ndl_record_category"] != NDL_REPOSITORY:
            errors.append(f"{bid}: 国立国会図書館作成の書誌ではない: {rec['ndl_record_category']}")
        if rec["isbn_ndl"] is None or M.to_isbn13(rec["isbn_ndl"]) != isbn13:
            errors.append(f"{bid}: NDL の ISBN {rec['isbn_ndl']} が指定 {p['isbn']} と一致しない")
        pages, height_cm = M.parse_extent(rec["extent"])
        phys = M.estimate_dims_weight(pages, height_cm)
        err = float(p["weight_true_error"])
        if not 0.05 <= abs(err) <= 0.15:
            errors.append(f"{bid}: weight_true_error {err} が ±0.05〜0.15 の外")
        appr = M.appraise(p["base_buyback_yen"], p["condition"], rule)
        if appr["decision"] != p["expected_decision"]:
            errors.append(f"{bid}: 査定規則の結果 {appr['decision']} が expected_decision {p['expected_decision']} と違う")
        fetched = raw_manifest.get("entries", {}).get(isbn13, {})
        entries.append({
            "id": bid,
            "isbn13": isbn13,
            "ean13": isbn13,  # 表紙のバーコード（EAN-13）は ISBN-13 そのもの
            "bib": {
                "title": rec["title"],
                "volume": rec["volume"],
                "series": rec["series"],
                "creators": rec["creators"],
                "responsibility": rec["responsibility"],
                "publisher": rec["publisher"],
                "date": rec["date"],
                "year": rec["year"],
                "extent": rec["extent"],
                "isbn_as_recorded": rec["isbn_ndl"],
                "ndl_bib_id": rec["ndl_bib_id"],
                "ndl_url": rec["ndl_url"],
                "fetched_at": fetched.get("fetched_at"),
                "raw_file": fetched.get("file", raw_path.relative_to(ROOT).as_posix()),
                "raw_sha256": fetched.get("sha256"),
            },
            "physical": {
                "pages": pages,
                "height_cm_ndl": height_cm,
                "format": phys["format"],
                "format_ja": phys["format_ja"],
                "binding": phys["binding"],
                "dims_mm": phys["dims_mm"],
                "weight_g_estimated": phys["weight_g_estimated"],
                "weight_g_true": M.true_weight(phys["weight_g_estimated"], err),
                "weight_true_error": err,
                "surface": p["surface"],
            },
            "appraisal": {
                "condition": p["condition"],
                "condition_factor": appr["condition_factor"],
                "base_buyback_yen": p["base_buyback_yen"],
                "buyback_yen": appr["buyback_yen"],
                "expected_decision": p["expected_decision"],
                "borderline": appr["borderline"],
            },
            "cover": {"label_ascii": p["label_ascii"]},
        })
    return entries, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="書き出さずに検証だけする")
    args = ap.parse_args()

    pricing = yaml.safe_load(PRICING.read_text(encoding="utf-8"))
    mpath = RAW_DIR / "manifest.json"
    raw_manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    entries, errors = build(pricing, raw_manifest)
    if len(entries) != 10:
        errors.append(f"冊数が 10 でない: {len(entries)}")
    if errors:
        for e in errors:
            print("[NG]", e, file=sys.stderr)
        sys.exit(1)

    fetched_dates = sorted({e["bib"]["fetched_at"][:10] for e in entries if e["bib"]["fetched_at"]})
    retrieved = fetched_dates[-1] if fetched_dates else "不明"
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    catalog = {
        "provenance": {
            "artifactId": "pai-book-intake/catalog/books",
            "version": "v1",
            "generatedAt": now,
            "generatedBy": "tools/build_catalog.py",
            "inputs": ["assets/catalog/pricing_v1.yaml", "outputs/ndl_raw/manifest.json"],
            "sources": [
                {**ATTRIBUTION, "retrievedOn": fetched_dates,
                 "attribution_text": ATTRIBUTION["attribution_text"].format(date=retrieved)},
                {"name": "assets/catalog/pricing_v1.yaml（手書き）",
                 "license": "このリポジトリの一部",
                 "note": "買取価格・状態・表面・本当の重さの誤差はすべて架空"},
            ],
            "status": "draft",
            "classification": "public",
            "notes": [
                "書誌（題名・著者・出版者・年・ページ数・大きさ）は国立国会図書館書誌データ。国立国会図書館が作成した書誌（R100000002）だけを使う。",
                "価格・状態・weight_g_true は架空。特定の店舗やマーケットプレイスの価格は参照せず、中古書の一般的な傾向を一般知識として大まかに写した。",
                "寸法は NDL の大きさ（高さ cm・切り上げ）から一般的な判型を当てはめ、厚さはページ数と紙厚の仮定から計算した。式は weight_model と tools/catalog_model.py。",
                "表紙はプロジェクトで作るダミー（tools/make_covers.py）。実在の書影は使わない。",
            ],
        },
        "weight_model": M.WEIGHT_MODEL,
        "format_table": {str(k): v for k, v in M.FORMATS.items()},
        "decision_rule": {
            **pricing["decision_rule"],
            "formula": "buyback_yen = round(base_buyback_yen * condition_factor[condition]); "
                       "decision = 'shelf' if buyback_yen >= shelf_threshold_yen else 'discard'",
        },
        "books": entries,
    }
    for e in entries:
        a, ph = e["appraisal"], e["physical"]
        print(f"{e['id']} {e['isbn13']} {ph['format']:<12} {ph['pages']:>4}p "
              f"{ph['dims_mm']['h']}x{ph['dims_mm']['w']}x{ph['dims_mm']['t']}mm "
              f"est={ph['weight_g_estimated']}g true={ph['weight_g_true']}g "
              f"buyback={a['buyback_yen']}円 {a['expected_decision']}{' (borderline)' if a['borderline'] else ''} "
              f"{e['bib']['title']}")
    if args.check:
        print("check OK（書き出しなし）")
        return
    OUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()} ({len(entries)} 冊)")


if __name__ == "__main__":
    main()
