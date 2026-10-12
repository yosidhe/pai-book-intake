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
