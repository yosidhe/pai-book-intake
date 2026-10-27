"""買取ラインの配置データ（assets/site/site_v1.json）を読む。Genesis に依存しない。

本棚はラインから離れた場所にあり、その位置はデータとして保管する（2026-10-20 ユーザー指示）。
ロボット B は台車で本棚の前まで移動し、そこで空きを見て本を差し込む。

棚の座標: 原点は棚の左手前の角。x = 幅（左→右）、y = 奥行（手前 0 → 奥）、z = 上（棚の床の下面 = 台の上面）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SITE = Path(__file__).resolve().parents[2] / "assets" / "site" / "site_v1.json"


@dataclass(frozen=True)
class ShelfSite:
    shelf_id: str
    origin: tuple[float, float]
    yaw: float
    stand_h: float
    width: float
    depth: float
    n_fingers: int
    pitch: float
    front_standoff: float
    cam_height: float
    cam_forward: float

    def to_world(self, p_shelf) -> np.ndarray:
        """棚の座標の点（…, 3）を世界座標へ。"""
        p = np.asarray(p_shelf, dtype=float)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return np.stack([self.origin[0] + c * x - s * y, self.origin[1] + s * x + c * y, self.stand_h + z], -1)

    def to_shelf(self, p_world) -> np.ndarray:
        p = np.asarray(p_world, dtype=float)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = p[..., 0] - self.origin[0], p[..., 1] - self.origin[1]
        return np.stack([c * dx + s * dy, -s * dx + c * dy, p[..., 2] - self.stand_h], -1)

    def dir_world(self, d_shelf) -> np.ndarray:
        """棚の座標の向き（…, 3）を世界座標へ（平行移動なし）。"""
        d = np.asarray(d_shelf, dtype=float)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)

    def front_pose(self, x_shelf: float | None = None) -> tuple[float, float, float]:
        """台車を止める位置（世界の x, y, 向き）。棚の x_shelf（省略時は中央）の正面、front_standoff 手前で棚を向く。"""
        xs = self.width / 2 if x_shelf is None else x_shelf
        p = self.to_world([xs, -self.front_standoff, 0.0])
        return float(p[0]), float(p[1]), self.yaw + math.pi / 2


@dataclass(frozen=True)
class Site:
    site_id: str
    raw: dict
    shelves: tuple[ShelfSite, ...]

    def shelf(self, shelf_id: str | None = None) -> ShelfSite:
        if shelf_id is None:
            return self.shelves[0]
        for s in self.shelves:
            if s.shelf_id == shelf_id:
                return s
        raise KeyError(shelf_id)

    def __getitem__(self, key: str):
        return self.raw[key]


def load_site(path: Path = DEFAULT_SITE) -> Site:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != "pai-book-intake/site@1":
        raise ValueError(f"配置データの形式が違う: {raw.get('schema')}")
    shelves = []
    for s in raw["shelves"]:
        cam = s["camera"]
        shelves.append(ShelfSite(
            shelf_id=s["shelf_id"], origin=tuple(s["origin"]), yaw=float(s["yaw"]), stand_h=float(s["stand_h"]),
            width=float(s["width"]), depth=float(s["depth"]), n_fingers=int(s["n_fingers"]), pitch=float(s["pitch"]),
            front_standoff=float(s["front_standoff"]), cam_height=float(cam["height_above_floor"]),
            cam_forward=float(cam["forward"])))
    if not shelves:
        raise ValueError("本棚が 1 つも無い")
    return Site(raw["site_id"], raw, tuple(shelves))
