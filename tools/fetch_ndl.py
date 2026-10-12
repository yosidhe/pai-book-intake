"""国立国会図書館サーチ（NDL サーチ）の SRU API から、カタログ用 10 冊の書誌を取得する。

- 対象は assets/catalog/pricing_v1.yaml の books[].isbn（ISBN-10 / 13）
- 検索は SRU（https://ndlsearch.ndl.go.jp/api/sru）、recordSchema=dcndl。
  データプロバイダを国立国会図書館蔵書（dpid=iss-ndl-opac）に限る＝国立国会図書館が作成した書誌だけを使う
- 生の応答を outputs/ndl_raw/<ISBN13>.xml に保存し、取得記録を outputs/ndl_raw/manifest.json に書く
- 行儀: 1 件ずつ・間隔 1 秒以上・User-Agent にプロジェクト名（個人名・連絡先は入れない）

使い方: python tools/fetch_ndl.py [--delay 1.5] [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog_model import to_isbn13  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRICING = ROOT / "assets" / "catalog" / "pricing_v1.yaml"
RAW_DIR = ROOT / "outputs" / "ndl_raw"
SRU = "https://ndlsearch.ndl.go.jp/api/sru"
DPID = "iss-ndl-opac"
USER_AGENT = "pai-book-intake-catalog/1.0 (course project; bibliographic lookup)"


def sru_url(isbn: str) -> str:
    q = {
        "operation": "searchRetrieve",
        "version": "1.2",
        "recordSchema": "dcndl",
        "recordPacking": "xml",
        "maximumRecords": "5",
        "query": f'isbn="{isbn}" AND dpid={DPID}',
    }
    return SRU + "?" + urllib.parse.urlencode(q)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def summarize(xml: str) -> str:
    n = re.search(r"<numberOfRecords>(\d+)</numberOfRecords>", xml)
    title = re.search(r"<dcterms:title>(.*?)</dcterms:title>", xml)
    extent = re.search(r"<dcterms:extent>(.*?)</dcterms:extent>", xml)
    return (f"records={n.group(1) if n else '?'} title={title.group(1) if title else '-'} "
            f"extent={extent.group(1) if extent else '-'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.5, help="リクエスト間隔 [秒]（1 未満は 1 にする）")
    ap.add_argument("--force", action="store_true", help="保存済みでも取り直す")
    args = ap.parse_args()
    delay = max(1.0, args.delay)

    books = yaml.safe_load(PRICING.read_text(encoding="utf-8"))["books"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"entries": {}}

    first = True
    for b in books:
        isbn13 = to_isbn13(b["isbn"])
        path = RAW_DIR / f"{isbn13}.xml"
        if path.exists() and not args.force:
            print(f"[skip] {b['id']} {isbn13} 保存済み: {summarize(path.read_text(encoding='utf-8'))}")
            continue
        if not first:
            time.sleep(delay)
        first = False
        url = sru_url(b["isbn"])  # NDL 側の登録どおりの ISBN（旧い本は ISBN-10）で引く
        body = fetch(url)
        text = body.decode("utf-8")
        if "<diagnostics>" in text or "<numberOfRecords>0<" in text:
            print(f"[miss] {b['id']} {b['isbn']}: {summarize(text)}", file=sys.stderr)
        path.write_bytes(body)
        manifest["entries"][isbn13] = {
            "id": b["id"],
            "query_isbn": b["isbn"],
            "url": url,
            "file": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"[ok]   {b['id']} {isbn13}: {summarize(text)}")

    manifest.update({"endpoint": SRU, "dpid": DPID, "user_agent": USER_AGENT})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path.relative_to(ROOT).as_posix()} ({len(manifest['entries'])} 件)")


if __name__ == "__main__":
    main()
