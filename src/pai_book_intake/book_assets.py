"""本の見た目（表紙のテクスチャとメッシュ）を実行時に作る。生成物は実行の出力先（named volume）に置く。

試行2-8（バーコード 95%）と同じ解像度にする: 表紙は 1 mm あたり約 15.75 px（130 mm 幅で 2048 px）。
テクスチャは「表紙の領域 + 右端の細い単色の帯」。単色の帯は表紙以外の面（裏表紙・背・小口）に使う。
バーコードは表紙（本の +z 面）の右上に実寸 4.5 cm 幅で印字する（ISBN の EAN-13）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .books import SURFACES, BookSpec

PX_PER_M = 2048 / 0.130
STRIP_PX = 32  # 単色の帯の幅
BARCODE_W_M = 0.045


def cover_px(book: BookSpec) -> tuple[int, int]:
    return int(round(book.width_m * PX_PER_M)), int(round(book.height_m * PX_PER_M))


def cover_color(book: BookSpec) -> tuple[int, int, int]:
    # 表面の種類ごとの基本色に、本ごとの小さな明るさの差をつける（動画で見分けられるように）
    base = np.array(SURFACES[book.surface]["color"]) * 255
    shift = (int(book.isbn13[-4:-1]) % 7 - 3) * 12
    return tuple(int(v) for v in np.clip(base + shift, 0, 255))


def make_texture(book: BookSpec, path: Path) -> Path:
    import zxingcpp  # バーコードを描くときだけ要る

    cw, ch = cover_px(book)
    color = cover_color(book)
    img = Image.new("RGB", (cw + STRIP_PX, ch), color)
    bc = np.asarray(zxingcpp.write_barcode_to_image(
        zxingcpp.create_barcode(book.isbn13, zxingcpp.BarcodeFormat.EAN13), scale=10))
    bc_img = Image.fromarray(bc).convert("RGB")
    w = int(BARCODE_W_M * PX_PER_M)
    bc_img = bc_img.resize((w, int(w * bc_img.height / bc_img.width)), Image.LANCZOS)
    label = Image.new("RGB", (bc_img.width + 96, bc_img.height + 96), (255, 255, 255))
    label.paste(bc_img, (48, 48))
    img.paste(label, (cw - label.width - 80, 80))
    band_y = int(ch * 0.40)
    ImageDraw.Draw(img).rectangle([int(cw * 0.06), band_y, int(cw * 0.94), band_y + int(ch * 0.055)],
                                  fill=(240, 230, 210))  # 題名の帯（見た目だけ）
    img.save(path)
    return path


def make_obj(book: BookSpec, path: Path, tex_name: str) -> Path:
    """表紙(+z)だけ UV を表紙の領域に割り当てた直方体の OBJ（面は三角形に分ける）。"""
    hx, hy, hz = (v / 2 for v in book.size_m)
    cw, _ = cover_px(book)
    u_max = cw / (cw + STRIP_PX)
    u_plain = 1 - (STRIP_PX / 2) / (cw + STRIP_PX)
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    vt = [(0, 0), (u_max, 0), (u_max, 1), (0, 1), (u_plain, 0.5)]
    faces = [
        ((5, 1), (6, 2), (7, 3), (8, 4)),  # +z 表紙
        ((1, 5), (4, 5), (3, 5), (2, 5)),  # -z
        ((1, 5), (2, 5), (6, 5), (5, 5)),
        ((2, 5), (3, 5), (7, 5), (6, 5)),
        ((3, 5), (4, 5), (8, 5), (7, 5)),
        ((4, 5), (1, 5), (5, 5), (8, 5)),
    ]
    lines = [f"mtllib {tex_name}.mtl", *[f"v {a:.5f} {b:.5f} {c:.5f}" for a, b, c in v],
             *[f"vt {a:.5f} {b:.5f}" for a, b in vt], "usemtl cover"]
    for a, b_, c, d in faces:
        lines += ["f " + " ".join(f"{vi}/{ti}" for vi, ti in tri) for tri in ((a, b_, c), (a, c, d))]
    (path.parent / f"{tex_name}.mtl").write_text(f"newmtl cover\nKd 1 1 1\nmap_Kd {tex_name}.png\n")
    path.write_text("\n".join(lines) + "\n")
    return path


def make_book_assets(book: BookSpec, out_dir: Path) -> tuple[Path, Path]:
    d = Path(out_dir) / "books"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{book.book_id}_tex"
    tex = make_texture(book, d / f"{name}.png")
    obj = make_obj(book, d / f"{book.book_id}.obj", name)
    return obj, tex
