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


# ---------- B の台車の道（02_確認事項 B15） ----------

# 台車の床板（pinch.CHASSIS_XML と同じ。台車の中心から前へ +0.25・後ろへ -0.35、左右 ±0.24）
CART_SIZE = (0.60, 0.48)
CART_OFFSET = -0.05
ROUTE_MARGIN = 0.10  # 台車の床板と障害物の間に空ける距離


def rect_corners(center, size, yaw: float = 0.0, offset: float = 0.0) -> np.ndarray:
    """長方形の 4 隅（4, 2）。offset は長方形の中心を自分の向き（yaw）の前へずらす量。"""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size[0] / 2, size[1] / 2
    local = np.array([[offset - hx, -hy], [offset + hx, -hy], [offset + hx, hy], [offset - hx, hy]])
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(center, dtype=float)[:2] + local @ rot.T


def cart_corners(pose) -> np.ndarray:
    return rect_corners(pose[:2], CART_SIZE, float(pose[2]), CART_OFFSET)


def _seg_point_dist(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _overlap(P: np.ndarray, Q: np.ndarray) -> bool:
    """凸多角形どうしが重なるか（分離軸）。"""
    for poly in (P, Q):
        for k in range(len(poly)):
            e = poly[(k + 1) % len(poly)] - poly[k]
            ax = np.array([-e[1], e[0]])
            a, b = P @ ax, Q @ ax
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def polygon_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """凸多角形どうしの距離。重なれば 0。"""
    if _overlap(P, Q):
        return 0.0
    d = min(_seg_point_dist(p, Q[k], Q[(k + 1) % len(Q)]) for p in P for k in range(len(Q)))
    return min(d, min(_seg_point_dist(q, P[k], P[(k + 1) % len(P)]) for q in Q for k in range(len(P))))


def obstacle_rects(site: "Site") -> dict[str, np.ndarray]:
    """B の台車が近づいてはいけない物の外形（上から見た 4 隅）。廃棄箱は壁の外側まで"""
    raw = site.raw
    tb, bn = raw["appraisal_table"], raw["bin"]
    out = {"appraisal_table": rect_corners(tb["center"], tb["size"]),
           "bin": rect_corners(bn["center"], [v + 2 * bn["wall_t"] for v in bn["inner"]])}
    if "hold_table" in raw:
        ht = raw["hold_table"]
        out["hold_table"] = rect_corners(ht["center"], ht["size"])
    return out


def b_route(robot_b: dict, start, goal) -> list[tuple[float, float, float]]:
    """B の台車が start から goal へ行く道（中継点の列。どれも (x, y, 向き)）。
    待機位置の手前の点（station_approach）まで start の向きのまま進み、そこで向きを変えてから goal へ向かう。
    待機位置から本棚へ向きを変えながらまっすぐ進むと、腕の台座（link0）が廃棄箱に当たった（20261101T003116-6d04d8。B15）"""
    ap = robot_b.get("station_approach")
    start, goal = tuple(float(v) for v in start), tuple(float(v) for v in goal)
    if ap is None:
        return [start, goal]
    return [start, (float(ap[0]), float(ap[1]), start[2]), (float(ap[0]), float(ap[1]), goal[2]), goal]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def route_clearance(route, obstacles: dict[str, np.ndarray], step_m: float = 0.01, step_rad: float = 0.02) -> dict:
    """道（中継点の列）を、台車の動き（g_drive と同じく x・y・向きを同時に補間）どおりに細かくたどり、
    障害物ごとに台車の床板との最も近い距離を返す。{name: (距離, その時の姿勢)}"""
    best = {k: (math.inf, None) for k in obstacles}
    for a, b in zip(route[:-1], route[1:]):
        dyaw = _wrap(b[2] - a[2])
        n = max(1, int(math.ceil(max(math.hypot(b[0] - a[0], b[1] - a[1]) / step_m, abs(dyaw) / step_rad))))
        for k in range(n + 1):
            u = k / n
            pose = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + dyaw * u)
            P = cart_corners(pose)
            for name, Q in obstacles.items():
                d = polygon_distance(P, Q)
                if d < best[name][0]:
                    best[name] = (d, pose)
    return best


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
