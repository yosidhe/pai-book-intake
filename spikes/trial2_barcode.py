"""試行2-8: 審査台の真上のカメラ画像から、本の表紙のバーコード（ISBN / EAN-13）を読む。

表紙（本の +z 面）の右上にバーコードを印字したテクスチャを貼り、位置と向き（0〜360°）をランダムにして撮る。
裏向き（バーコードが下）の本では、何も読めないこと（＝裏返しが必要と判断できること）も確かめる。
読み取りは zxing-cpp。斜めのバーコードに強くするため、画像を一定の刻み（--step-deg）で回して読む。

判定: 表向きの本で 90% 以上、正しい ISBN が読める。裏向きの本で誤検出が 0 件

使い方: python spikes/trial2_barcode.py --n-up 40 --n-down 20
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import zxingcpp
from PIL import Image, ImageDraw

import genesis as gs
import trial2_common as C

BOOK = (0.13, 0.19, 0.02)
TABLE_TOP = 0.40
CAM_H = 0.30
STEP_DEG = 10
CAM_RES = (1280, 960)
ATLAS = 4096
COVER_PX = (2048, 2992)  # 表紙の領域（幅・高さ）。本の 0.13 x 0.19 m に対応
PLAIN_UV = (0.93, 0.07)  # 表紙以外の面に使う単色の位置


def ean13(body12: str) -> str:
    s = sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(body12))
    return body12 + str((10 - s % 10) % 10)


def make_texture(isbn: str, path) -> None:
    img = Image.new("RGB", (ATLAS, ATLAS), (70, 90, 140))
    cover = Image.new("RGB", COVER_PX, (200, 70, 60))
    bc = np.asarray(zxingcpp.write_barcode_to_image(zxingcpp.create_barcode(isbn, zxingcpp.BarcodeFormat.EAN13),
                                                    scale=10))
    bc_img = Image.fromarray(bc).convert("RGB")
    w = int(COVER_PX[0] * 0.045 / BOOK[0])  # 実寸 4.5 cm 幅
    bc_img = bc_img.resize((w, int(w * bc_img.height / bc_img.width)), Image.LANCZOS)
    label = Image.new("RGB", (bc_img.width + 96, bc_img.height + 96), (255, 255, 255))
    label.paste(bc_img, (48, 48))
    cover.paste(label, (COVER_PX[0] - label.width - 80, 80))
    ImageDraw.Draw(cover).rectangle([120, 1200, 1920, 1360], fill=(240, 230, 210))  # 題名の帯（見た目だけ）
    img.paste(cover, (0, 0))
    img.save(path)


def make_obj(path, tex_name: str) -> None:
    """表紙(+z)だけ UV を表紙領域に割り当てた直方体の OBJ。"""
    hx, hy, hz = (v / 2 for v in BOOK)
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    u_max, v_top = COVER_PX[0] / ATLAS, 1 - COVER_PX[1] / ATLAS
    vt = [(0, v_top), (u_max, v_top), (u_max, 1), (0, 1), PLAIN_UV]
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
    for a, b_, c, d in faces:  # 三角形に分ける
        lines += ["f " + " ".join(f"{vi}/{ti}" for vi, ti in tri) for tri in ((a, b_, c), (a, c, d))]
    (path.parent / f"{tex_name}.mtl").write_text(f"newmtl cover\nKd 1 1 1\nmap_Kd {tex_name}.png\n")
    path.write_text("\n".join(lines) + "\n")


def decode(rgb: np.ndarray) -> str | None:
    img = Image.fromarray(rgb).convert("L")
    for ang in range(0, 180, STEP_DEG):
        res = zxingcpp.read_barcodes(np.asarray(img.rotate(ang, expand=True, fillcolor=255)),
                                     formats=zxingcpp.BarcodeFormat.EAN13)
        if res:
            return res[0].text
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-up", type=int, default=40)
    ap.add_argument("--n-down", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-deg", type=int, default=10)
    ap.add_argument("--cam-res", type=int, nargs=2, default=[1280, 960])
    args = ap.parse_args()
    global STEP_DEG, CAM_RES
    STEP_DEG, CAM_RES = args.step_deg, tuple(args.cam_res)
    run_id, out, meta = C.new_run("trial2_barcode", args)
    rng = np.random.default_rng(args.seed)
    isbn = ean13("978400000001")
    make_texture(isbn, out / "book_tex.png")
    make_obj(out / "book.obj", "book_tex")

    C.init_genesis(args.seed)
    scene = C.make_scene(1)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.5, 0.5, TABLE_TOP), pos=(0.6, 0.0, TABLE_TOP / 2), fixed=True),
                     surface=gs.surfaces.Default(color=(0.85, 0.85, 0.82)))
    book = scene.add_entity(
        gs.morphs.Mesh(file=str(out / "book.obj"), pos=(0.6, 0.0, TABLE_TOP + BOOK[2] / 2 + 0.002)),
        surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(out / "book_tex.png"))),
    )
    cam = scene.add_camera(res=CAM_RES, pos=(0.6, 0.0, TABLE_TOP + CAM_H), lookat=(0.6, 0.0, TABLE_TOP),
                           up=(1.0, 0.0, 0.0), fov=40)
    scene.build()

    flip = np.array([0.0, 1.0, 0.0, 0.0])  # x 軸まわり 180°（裏向き）
    rows = []
    for k in range(args.n_up + args.n_down):
        face_up = k < args.n_up
        yaw = rng.uniform(0, 2 * math.pi)
        q = C.quat_z(np.array(yaw))
        if not face_up:
            q = C.quat_mul(q, flip)
        book.set_pos(np.array([0.6 + rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02),
                               TABLE_TOP + BOOK[2] / 2 + 0.002]))
        book.set_quat(q)
        for _ in range(5):
            scene.step()
        rgb, *_ = cam.render()
        rgb = np.asarray(rgb)
        text = decode(rgb)
        rows.append({"face_up": face_up, "yaw_deg": round(math.degrees(yaw), 1), "read": text})
        if k in (0, args.n_up):
            Image.fromarray(rgb).save(out / f"sample_{'up' if face_up else 'down'}.png")

    up = [r for r in rows if r["face_up"]]
    down = [r for r in rows if not r["face_up"]]
    up_ok = sum(r["read"] == isbn for r in up) / max(1, len(up))
    down_false = sum(r["read"] is not None for r in down)
    result = {
        "criterion": "8_barcode_read",
        "isbn": isbn,
        "face_up_read_rate": up_ok,
        "face_down_false_reads": down_false,
        "pass": bool(up_ok >= 0.9 and down_false == 0),
        "failed_yaws_deg": [r["yaw_deg"] for r in up if r["read"] != isbn],
        "per_env": rows,
    }
    C.save_result(out, meta, result)


if __name__ == "__main__":
    main()
