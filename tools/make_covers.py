"""カタログの 10 冊ぶん、自作のダミー表紙（PNG）を作り、バーコードが読み戻せるか確かめる。

- 実在の書影は使わない。単色の地＋ローマ字の短いラベル＋ISBN の文字＋EAN-13 バーコード（ISBN-13）だけ
- 日本語フォントは使わない・ダウンロードしない（Pillow の既定フォントで ASCII のみ）
- 大きさは本の実寸（dims_mm の h×w）に比例（--px-per-mm、既定 10 px/mm）
- バーコードは spikes/trial2_barcode.py と同じく表紙の右上、実寸 4.5 cm 幅
- 読み戻し: zxing-cpp で、原寸・1/2・1/4 縮小の 3 通りを読む。結果を outputs/covers/read_check.json に書く。
  合格条件は原寸で 10/10。縮小は「どこまで解像度を落とせるか」の参考値。
  二値化は既定（LocalAverage）→ GlobalHistogram → FixedThreshold の順に試し、読めた方式を記録する
  （縮小した表紙全体では、暗い地と白いラベルの境目で既定の局所二値化が崩れることがあるため）

出力: outputs/covers/<id>_<ISBN13>.png（outputs/ は git 管理外の named volume）
使い方: python tools/make_covers.py [--px-per-mm 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zxingcpp
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "assets" / "catalog" / "books_v1.json"
OUT_DIR = ROOT / "outputs" / "covers"
BARCODE_W_MM = 45.0
SCALES = (1, 2, 4)  # 読み戻しを試す縮小率（1/1, 1/2, 1/4）
# 地の色（彩度を抑えた中間色。どれでも白いラベル上のバーコードとは十分なコントラストがある）
PALETTE = [(120, 70, 60), (60, 90, 120), (70, 110, 80), (130, 110, 60), (90, 70, 110),
           (110, 110, 110), (140, 80, 50), (50, 100, 110), (100, 60, 80), (80, 80, 60)]


def font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def barcode_image(ean13: str, px_w: int) -> Image.Image:
    bc = np.asarray(zxingcpp.write_barcode_to_image(
        zxingcpp.create_barcode(ean13, zxingcpp.BarcodeFormat.EAN13), scale=10))
    img = Image.fromarray(bc).convert("RGB")
    return img.resize((px_w, int(px_w * img.height / img.width)), Image.LANCZOS)


def make_cover(book: dict, idx: int, px_per_mm: float) -> Image.Image:
    dims = book["physical"]["dims_mm"]
    W, H = int(dims["w"] * px_per_mm), int(dims["h"] * px_per_mm)
    cover = Image.new("RGB", (W, H), PALETTE[idx % len(PALETTE)])
    bc = barcode_image(book["ean13"], int(BARCODE_W_MM * px_per_mm))
    pad = int(4 * px_per_mm)
    label = Image.new("RGB", (bc.width + 2 * pad, bc.height + 2 * pad), (255, 255, 255))
    label.paste(bc, (pad, pad))
    margin = int(6 * px_per_mm)
    cover.paste(label, (W - label.width - margin, margin))
    # 題名の帯（ASCII のラベル）と ISBN の文字
    d = ImageDraw.Draw(cover)
    band_top = int(H * 0.55)
    band_h = int(24 * px_per_mm)
    d.rectangle([margin, band_top, W - margin, band_top + band_h], fill=(240, 236, 226))
    f_big = font(int(6 * px_per_mm))
    f_small = font(int(4 * px_per_mm))
    d.text((margin + pad, band_top + pad), book["cover"]["label_ascii"], fill=(20, 20, 20), font=f_big)
    d.text((margin + pad, band_top + pad + int(10 * px_per_mm)), f"ISBN {book['isbn13']}", fill=(20, 20, 20),
           font=f_small)
    d.text((margin, H - margin - int(5 * px_per_mm)), f"{book['id']}  dummy cover", fill=(235, 235, 235),
           font=f_small)
    return cover


BINARIZERS = (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram, zxingcpp.Binarizer.FixedThreshold)


def read_ean13(img: Image.Image) -> dict:
    gray = np.asarray(img.convert("L"))
    for bz in BINARIZERS:
        res = zxingcpp.read_barcodes(gray, formats=zxingcpp.BarcodeFormat.EAN13, binarizer=bz)
        if res:
            return {"text": res[0].text, "binarizer": bz.name}
    return {"text": None, "binarizer": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--px-per-mm", type=float, default=10.0)
    args = ap.parse_args()

    books = json.loads(CATALOG.read_text(encoding="utf-8"))["books"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, b in enumerate(books):
        cover = make_cover(b, i, args.px_per_mm)
        path = OUT_DIR / f"{b['id']}_{b['isbn13']}.png"
        cover.save(path)
        reads = {}
        for k in SCALES:
            img = cover if k == 1 else cover.resize((cover.width // k, cover.height // k), Image.LANCZOS)
            reads[f"1/{k}"] = read_ean13(img)
        rows.append({"id": b["id"], "ean13": b["ean13"], "file": path.relative_to(ROOT).as_posix(),
                     "size_px": list(cover.size), "reads": reads})
        print(f"{b['id']} {b['ean13']} {cover.size} "
              + " ".join(f"{s}={v['text']}({v['binarizer']})" for s, v in reads.items()))
    n = len(rows)
    rates = {f"1/{k}": sum(r["reads"][f"1/{k}"]["text"] == r["ean13"] for r in rows) / n for k in SCALES}
    result = {
        "catalog": CATALOG.relative_to(ROOT).as_posix(),
        "px_per_mm": args.px_per_mm,
        # zxing-cpp の EAN-13 画像は両端の余白込みで 113 モジュール幅（95 + 余白 9×2）
        "px_per_module_full": round(BARCODE_W_MM / 113 * args.px_per_mm, 2),
        "binarizers_tried": [b.name for b in BINARIZERS],
        "read_rate": rates,
        "read_rate_full": rates["1/1"],
        "rows": rows,
    }
    (OUT_DIR / "read_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("read rate: " + " ".join(f"{s}={v:.0%}" for s, v in rates.items()) + f" ({n} 冊)")
    if result["read_rate_full"] < 1.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
