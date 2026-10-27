"""本棚: パタパタ・ブックキーパー v2 を横に長くしたもの（B4）の MJCF と、空きの見つけ方。Genesis に依存しない。

元の設計: workspace の 3d_modeling/pata_book_keeper_v2/pata_book_keeper_v2.scad（幅 142・奥行 120・指 22 枚）。
B4 の回答（2026-10-18）で、幅 400 mm（指 64 枚・6 mm おきはそのまま）・奥行 190 mm にした。指の形と回る範囲は .scad の値を写す。

  後ろの壁の上に、軸で回る指が並ぶ。休んでいる指は前下がり（−35°、キーで止まる）。本を小口から差し込むと、本の上の指だけが
  持ち上がり（上限 80°）、となりの指は下がったまま本の両側をはさむので、1 冊でも倒れない。ブックエンドは使わない。
  本は背表紙を手前・天を上にして置く。背表紙は同じ判型どうしでそろう（本を後ろの壁まで入れる。B4-2b）

棚の座標: x = 幅（左→右）、y = 奥行（手前 0 → 奥）、z = 上（床板の下面が 0）。

空きの見つけ方（free_spans）: 棚の前から、床板のすぐ上の高さを水平に見た距離（深さカメラの点）を x ごとに集め、
後ろの壁より手前に面がある x を「埋まっている」とする。埋まっていない x の連続を空きとする。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- .scad から写す値（mm → m） ----
BASE_T = 0.003
WT = 0.004
WALL_TOP = 0.100
EAR_T = 0.004
F_HUB_T = 0.0056
F_T = 0.004
F_HUB_R = 0.0065
F_L = 0.064
F_BAND = 0.0045
F_BULB = 0.0045
REST_DEG = -35.0
MAXA_DEG = 80.0
BOOK_D = F_HUB_R + 0.0008
CAM_M = 1 / math.tan(math.radians(MAXA_DEG))
CAM_C0 = -BOOK_D / math.sin(math.radians(MAXA_DEG)) + 0.001
PLA_DENSITY = 1240.0
WOOD_RGBA = "0.86 0.84 0.78 1"
FINGER_RGBA = "0.95 0.55 0.15 1"


@dataclass(frozen=True)
class KeeperSpec:
    width: float = 0.400
    depth: float = 0.190
    n_fingers: int = 64
    pitch: float = 0.006

    @property
    def piv_y(self) -> float:
        return self.depth - F_HUB_R - 0.0003

    @property
    def piv_z(self) -> float:
        return WALL_TOP + F_HUB_R + 0.0008

    @property
    def x0(self) -> float:
        """1 枚目の指のハブの左端（.scad の F_X0）。"""
        in_w = self.width - 2 * EAR_T
        return EAR_T + (in_w - ((self.n_fingers - 1) * self.pitch + F_HUB_T)) / 2

    def finger_x(self, i: int) -> float:
        """i 枚目の指の羽根の中心の x。"""
        return self.x0 + i * self.pitch + F_T / 2

    @property
    def back_face_y(self) -> float:
        """後ろの壁の手前の面の y（本の小口はここまで入る）。"""
        return self.depth - WT

    @property
    def usable_x(self) -> tuple[float, float]:
        """本を置ける x の範囲（左右の耳の内側）。"""
        return EAR_T, self.width - EAR_T

    def fingers_over(self, x_lo: float, x_hi: float) -> list[int]:
        """x_lo〜x_hi に置いた本の上にかかる指の番号。"""
        return [i for i in range(self.n_fingers)
                if self.finger_x(i) + F_T / 2 > x_lo and self.finger_x(i) - F_T / 2 < x_hi]

    def check(self) -> None:
        if self.x0 < EAR_T:
            raise ValueError(f"指 {self.n_fingers} 枚が幅 {self.width} m に入らない")


def _f(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".") if v else "0"


def keeper_mjcf(spec: KeeperSpec, env_bit: int = 1) -> str:
    """本棚の MJCF（根は固定。床板・後ろの壁・左右の耳・軸と、軸で回る指 n_fingers 枚）。
    当たりは env_bit だけ（爪とは当たらない。hand_design の ENV_BIT と同じ考え方）。"""
    spec.check()
    W, D = spec.width, spec.depth
    col = f'contype="{env_bit}" conaffinity="{env_bit}"'
    g = []
    g.append(f'<geom name="floor" type="box" size="{_f(W / 2)} {_f(D / 2)} {_f(BASE_T / 2)}" '
             f'pos="{_f(W / 2)} {_f(D / 2)} {_f(BASE_T / 2)}" {col} rgba="{WOOD_RGBA}"/>')
    g.append(f'<geom name="back" type="box" size="{_f(W / 2)} {_f(WT / 2)} {_f(WALL_TOP / 2)}" '
             f'pos="{_f(W / 2)} {_f(D - WT / 2)} {_f(WALL_TOP / 2)}" {col} rgba="{WOOD_RGBA}"/>')
    ear_len = 0.062
    ear_h = spec.piv_z + 0.0075
    for k, x in enumerate((EAR_T / 2, W - EAR_T / 2)):
        g.append(f'<geom name="ear{k}" type="box" size="{_f(EAR_T / 2)} {_f(ear_len / 2)} {_f(ear_h / 2)}" '
                 f'pos="{_f(x)} {_f(D - ear_len / 2)} {_f(ear_h / 2)}" {col} rgba="{WOOD_RGBA}"/>')
    g.append(f'<geom name="axle" type="cylinder" fromto="0 {_f(spec.piv_y)} {_f(spec.piv_z)} {_f(W)} {_f(spec.piv_y)} '
             f'{_f(spec.piv_z)}" size="0.0025" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>')
    # 指（羽根の座標: 手前 = -y、上 = +z。角度 0 で羽根の付け根の線が水平）。下辺 us(u) = CAM_C0 + CAM_M u が本に乗る面
    u1 = F_L - F_BULB
    v0 = CAM_C0 + F_BAND / 2
    v1 = CAM_C0 + CAM_M * u1 + F_BAND / 2
    bulb_v = CAM_C0 + CAM_M * u1 + F_BULB
    fingers = []
    for i in range(spec.n_fingers):
        x = spec.finger_x(i)
        fingers.append(
            f'<body name="finger{i}" pos="{_f(x)} {_f(spec.piv_y)} {_f(spec.piv_z)}">'
            f'<joint name="kf{i}" type="hinge" axis="-1 0 0" range="{REST_DEG} {MAXA_DEG}" damping="0.0005" '
            f'armature="0.00002"/>'
            f'<geom type="cylinder" fromto="{_f(-F_HUB_T / 2)} 0 0 {_f(F_HUB_T / 2)} 0 0" size="{_f(F_HUB_R)}" '
            f'density="{PLA_DENSITY}" {col} rgba="{FINGER_RGBA}"/>'
            f'<geom type="capsule" fromto="0 0 {_f(v0)} 0 {_f(-u1)} {_f(v1)}" size="{_f(F_BAND / 2)}" '
            f'density="{PLA_DENSITY}" {col} rgba="{FINGER_RGBA}"/>'
            f'<geom type="sphere" pos="0 {_f(-u1)} {_f(bulb_v)}" size="{_f(F_BULB)}" density="{PLA_DENSITY}" {col} '
            f'rgba="{FINGER_RGBA}"/>'
            f'</body>')
    return ('<mujoco model="pata_keeper_wide"><compiler angle="degree"/><worldbody>'
            '<body name="keeper" pos="0 0 0">' + "".join(g) + "".join(fingers) + "</body></worldbody></mujoco>")


# ---------- 空きを見つける ----------

def occupancy_from_points(pts_shelf: np.ndarray, spec: KeeperSpec, bin_m: float = 0.002,
                          z_band: tuple[float, float] = (0.012, 0.050), wall_margin: float = 0.030,
                          y_min: float = -0.05) -> tuple[np.ndarray, np.ndarray]:
    """棚の座標の点群（k, 3）から、x の区間ごとに「埋まっている」かを返す（区間の左端の配列, 埋まりの bool 配列）。
    床板のすぐ上（z_band）で、後ろの壁より wall_margin 以上手前に面がある区間を埋まっているとする。
    点が 1 つも無い区間は「見えない」ので、安全のため埋まっている扱いにする。y_min より手前の点は使わない。"""
    edges = np.arange(0.0, spec.width + 1e-9, bin_m)
    left = edges[:-1]
    p = np.asarray(pts_shelf, dtype=float).reshape(-1, 3)
    sel = (p[:, 2] >= z_band[0]) & (p[:, 2] <= z_band[1]) & (p[:, 0] >= 0) & (p[:, 0] < spec.width) & (p[:, 1] > y_min)
    p = p[sel]
    k = np.clip((p[:, 0] / bin_m).astype(int), 0, len(left) - 1)
    nearest = np.full(len(left), np.inf)
    np.minimum.at(nearest, k, p[:, 1])
    occupied = ~(np.isfinite(nearest) & (nearest > spec.back_face_y - wall_margin))  # 見えない区間は埋まっている扱い
    return left, occupied


def free_spans(left: np.ndarray, occupied: np.ndarray, bin_m: float, x_range: tuple[float, float]) -> list[tuple[float, float]]:
    """埋まっていない区間の連続を [x_lo, x_hi] のリストで返す（x_range の内側に切り詰める）。"""
    out = []
    start = None
    for x, occ in zip(np.append(left, left[-1] + bin_m), np.append(occupied, True)):
        if not occ and start is None:
            start = x
        elif occ and start is not None:
            lo, hi = max(start, x_range[0]), min(x, x_range[1])
            if hi > lo:
                out.append((float(lo), float(hi)))
            start = None
    return out


def spans_from_books(books_x: list[tuple[float, float]], spec: KeeperSpec) -> list[tuple[float, float]]:
    """置かれている本の x の範囲から空きを出す（深さカメラを描かない環境の代わり。結果には source を残す）。"""
    lo, hi = spec.usable_x
    occ = sorted((max(a, lo), min(b, hi)) for a, b in books_x if b > lo and a < hi)
    out, cur = [], lo
    for a, b in occ:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if hi > cur:
        out.append((cur, hi))
    return out


def choose_slot(spans: list[tuple[float, float]], thickness: float, palm_side: int, hand_clear: float,
                gap: float = 0.003) -> tuple[float, float] | None:
    """本を置く x の範囲を選ぶ。手のひらは本の palm_side（-1 = 左, +1 = 右）にあり、そちらに hand_clear の空きが要る
    （小口の側の壁の指が、手のひらから指先まで棚の奥に入るため）。本は手のひらと反対の側の隣（本か棚の端）に寄せる。
    条件を満たす空きのうち、寄せる側がいちばん外の端に近いものを選ぶ（左詰め／右詰め）。無ければ None。"""
    need = thickness + 2 * gap + hand_clear
    cands = [(a, b) for a, b in spans if b - a >= need]
    if not cands:
        return None
    if palm_side < 0:  # 手のひらが左 → 本は空きの右端に寄せる。右からつめる
        a, b = max(cands, key=lambda s: s[1])
        return (b - gap - thickness, b - gap)
    a, b = min(cands, key=lambda s: s[0])
    return (a + gap, a + gap + thickness)


# ---------- 平らに置く（B8 の回答 c: 締切を優先し、本棚には平らに積む） ----------

def finger_tip_rest(spec: KeeperSpec) -> tuple[float, float]:
    """休んでいる指（REST_DEG）の先のおもりの中心（棚の y, z）。"""
    u1 = F_L - F_BULB
    bulb_v = CAM_C0 + CAM_M * u1 + F_BULB
    a = math.radians(-REST_DEG)  # 軸は -x。REST_DEG（負）は +x まわりに -REST_DEG 回すのと同じ
    y = -u1 * math.cos(a) - bulb_v * math.sin(a)
    z = -u1 * math.sin(a) + bulb_v * math.cos(a)
    return spec.piv_y + y, spec.piv_z + z


def flat_fore_edge_max_y(spec: KeeperSpec, wall_t: float, clearance: float = 0.008) -> float:
    """平らに置く本の小口（奥の端）の y の上限。手の小口の側の壁（厚み wall_t）は本の底まで下りるので、
    休んでいる指の先（おもり）より手前で止める。"""
    y_tip, _ = finger_tip_rest(spec)
    return y_tip - F_BULB - wall_t - clearance


def flat_book_center_y(spec: KeeperSpec, book_w: float, wall_t: float, min_inside: float = 0.015) -> float | None:
    """平らに置く本（背を手前・小口を奥）の中心の y。小口を指の手前の上限まで奥へ寄せる。背は棚の手前の縁から
    はみ出してよいが、重心（中心）は縁から min_inside 以上内側に置く。置けなければ None。"""
    yc = flat_fore_edge_max_y(spec, wall_t) - book_w / 2
    return yc if yc >= min_inside else None


def choose_flat_slot(spans: list[tuple[float, float]], length: float, gap: float = 0.010) -> tuple[float, float] | None:
    """平らに置く本（天地 length を棚の x に沿わせる）の x の範囲。左から詰める。入らなければ None。"""
    for a, b in sorted(spans):
        if b - a >= length + 2 * gap:
            return (a + gap, a + gap + length)
    return None


def choose_stack(piles: list[dict], length: float, max_top_z: float, x_range: tuple[float, float],
                 gap: float = 0.005, support: float = 0.015) -> tuple[int, float] | None:
    """床に空きが無いとき、積む山と、積む本の中心の x を選ぶ（piles: {x_lo, x_hi, top_z}。棚の座標）。
    本は山の上の本の中央に置くが、棚の幅（x_range の内側 gap）からはみ出さないようにずらす。ずらした後も、積む本の重心が
    上の本の端から support 以上内側に載る山だけを候補にする。候補のうち上の本の x の長さがいちばん長い山（支えが広い）を選ぶ。
    山の上面が max_top_z（呼ぶ側で、積む本の厚みを引いておく）より高い山は選ばない。無ければ None。
    gap 10 mm・support 20 mm では、文庫の上の B5 が 1 mm しか余裕が無く、環境ごとの置き位置の差で積めなかった（20261023T022827）"""
    lo, hi = x_range[0] + gap + length / 2, x_range[1] - gap - length / 2
    if lo > hi:
        return None
    cands = []
    for i, p in enumerate(piles):
        xc = min(max((p["x_lo"] + p["x_hi"]) / 2, lo), hi)
        if p["top_z"] <= max_top_z and p["x_lo"] + support <= xc <= p["x_hi"] - support:
            cands.append((p["x_hi"] - p["x_lo"], -p["top_z"], i, xc))
    if not cands:
        return None
    _, _, i, xc = max(cands)
    return i, xc
