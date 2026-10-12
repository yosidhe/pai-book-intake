"""バーコード（ISBN / EAN-13）の読み取り。試行2-8 と同じ方法: 画像を一定の刻みで回して zxing-cpp で読む。"""

from __future__ import annotations

import numpy as np
from PIL import Image

STEP_DEG = 10


def decode_isbn(rgb: np.ndarray, step_deg: int = STEP_DEG) -> str | None:
    import zxingcpp

    img = Image.fromarray(np.asarray(rgb)).convert("L")
    for ang in range(0, 180, step_deg):
        res = zxingcpp.read_barcodes(np.asarray(img.rotate(ang, expand=True, fillcolor=255)),
                                     formats=zxingcpp.BarcodeFormat.EAN13)
        if res:
            return res[0].text
    return None


def _read_at(img: Image.Image, ang: float):
    res = _zx().read_barcodes(np.asarray(img.rotate(ang, expand=True, fillcolor=255)),
                              formats=_zx().BarcodeFormat.EAN13)
    return res[0] if res else None


def _zx():
    import zxingcpp

    return zxingcpp


def read_isbn_with_yaw(rgb: np.ndarray, step_deg: int = STEP_DEG, fine_deg: int = 2,
                       window_deg: int = 48) -> tuple[str | None, float | None]:
    """ISBN と、画像の中でのバーコードの読む向き（右 0°・上 90°・反時計回り、度）を返す。読めなければ (None, None)。

    zxing-cpp は行か列に沿ったバーコードしか読まず、向きも 90° 刻みでしか返さない。そこで
      1. decode_isbn と同じ刻みで回して最初に読める角度を探し
      2. その前後 ±window_deg を fine_deg 刻みで回して、同じ向き（90° 刻み）で読める角度の範囲を求め
      3. 範囲の中央を「ちょうど 90° 刻みに揃った角度」とみなして向きを出す
    （PIL の rotate は反時計回り。zxing の orientation は時計回りなので符号を反転する）"""
    img = Image.fromarray(np.asarray(rgb)).convert("L")
    hit = None
    for ang in range(0, 180, step_deg):
        r = _read_at(img, ang)
        if r is not None:
            hit = (ang, r)
            break
    if hit is None:
        return None, None
    ang0, r0 = hit
    ok = [a for a in range(ang0 - window_deg, ang0 + window_deg + 1, fine_deg)
          if (r := _read_at(img, a)) is not None and r.text == r0.text and r.orientation == r0.orientation]
    center = (min(ok) + max(ok)) / 2  # ang0 自身が必ず入る（同じ画像・同じ角度）
    yaw = -float(r0.orientation) - center
    yaw = (yaw + 180.0) % 360.0 - 180.0
    return r0.text, yaw
