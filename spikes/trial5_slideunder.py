"""試行5: 本の下に滑り込ませる手で、本を下から支えて持ち上げる（手の形の比較）。

背景（2026-10-12 のユーザー判断）
  - 吸引で本を持ち上げない。どの本も、本の下に滑り込ませた手で下から支えて持ち上げる。吸引は補助だけ
    （本を押さえる・端を数 mm めくる・落下を防ぐ）。上限は表面ごと（滑らか 12 N・ざらざら 3 N、books.SURFACES）
  - 爪の長さを固定せず、手の形を複数試して最適な「手のひら」を探す（hand_design.HandDesign）
  - 物理的にありえることも成功の条件（plausibility.py）。試行4 は、本が 1 冊で長い辺で立った（剛体の箱だから
    起きる）ために不採用になった

1 つのシーンに「セル」（ロボット 1 台・取る場所・検品トレー・本）を横に並べ、手の形 × 吸引あり/なしを
同時に動かす（同じ本・同じ置き方・同じばらつき。環境ごとのばらつきはセルの間で同じ乱数）。

滑り込ませて持ち上げる手順（全セル・全環境で同じ。数値は下の定数）
  剛体の本は下の面と隙間が無い。くさびの先を押し当てても本を横に押すだけで、下には入らない
  （最初の実行: 全環境で爪の先が本の縁の 3.6 mm 外で止まり、本は横から挟まれて持ち上がっていた）。
  そこで「ちり取り」の要領で、爪の上へ本を押し出して隙間を作る（吸引は使わない）
  1. 本の位置と向きを測る（シミュレーションの真値。認識は別の試行で扱う）。指の開閉軸を本の横幅に合わせ、
     右の指（爪）をロボットから遠い側（+x）に置く。本を載せている面（下の本・台）の縁を測る
  2. 爪の上面を面より 0.5 mm 低く、爪の先を面の縁の 3 mm 外に置く（右の指の上下の関節で、左の指先は面の 2 mm 上）
  3. 左の指（両側の爪なら左の爪の先）で本の向こう側を押し、本を爪の上へ押し出す（本の縁が右の指の 2 mm 手前まで）
  4. 右の爪を 6 mm 上げて下から支える。両側の爪は本を約 6° 傾け（上げる量は横幅 × tan 6°）、左の爪を面に沿って左の縁の下へ押し込む
  5. 吸引あり: 爪が入ったあと、吸盤を上面に当てて吸着し、5 mm 引く（押さえ・運ぶ間の落下防止の補助）
  6. 左右から軽く挟み（推定の重さの 0.75 倍。上限は grip_plan）、吸盤（吸引ではなく押し付け）で上から軽く押さえる
  7. 持ち上げる（1.0 s）→ 検品トレーの上まで約 0.34 m 運ぶ（最小ジャーク軌道 1.2 s。加速度の最大は約 1.4 m/s²）。
     運ぶ間、手から本への力のうち本の下半分の接触の上向き成分を、本の重さで割った値（下から支える割合）を測る
  8. トレーの床の 1.5 mm 上まで下ろし、吸引を切る。吸盤で上から押さえたまま指を開き切って爪を抜き、
     吸盤を引いて、最初はゆっくり手を上げる
判定（環境ごと）: 爪が本の下に入った（両側の爪なら両方）・下から支える割合が 0.5 以上（＝横から挟んで
                持っていない）・持ち上がった・運ぶ間に落ちない・トレーの中に平らに置けた・放すとき本を 5 mm 以上
                引きずり上げない・下の本を乱していない。そのうえで、ありえない状態が無かった（plausibility.Monitor。
                違反は implausible として別に数える）

使い方: python spikes/trial5_slideunder.py --book book-10 --surface rough --situation stack --designs R25,B25 --n-envs 4
出力: /workspace/outputs/trial5_slideunder/<run_id>/（result.json・cells.csv・close_<cell>.mp4・montage_<cell>.png）
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import genesis as gs
from pai_book_intake.book_assets import make_book_assets
from pai_book_intake.books import DEFAULT_CATALOG, GRAVITY, SURFACES, grip_plan, load_catalog
from pai_book_intake.geometry import Q_DOWN, quat_mul, quat_to_mat, quat_z
from pai_book_intake.hand_design import CUP_TIP_RETRACTED_Z, FINGER_PLATE_T, PLUNGER_STROKE, V2, HandDesign
from pai_book_intake.plausibility import KINDS, Monitor, Rules
from pai_book_intake.runlog import classify, new_run, save_result
from pai_book_intake.tool import ARM, FINGERS, FORCE, LIFT, PLUNGER, Robot, World, build_robot_xml, home_for

TRIAL = "trial5_slideunder"
DT = 0.01
CELL_DX = 1.6  # セルの間隔（x）
PICK_C = (0.50, -0.14)  # 取る場所の中心（セルの座標）
TRAY_C = (0.50, 0.20)  # 検品トレーの中心（運ぶ距離 0.34 m）
TABLE_TOP = 0.10  # 平置きの台・トレーの台の上面
BOX_IN = (0.46, 0.34)  # 箱の内寸（x = 指の開閉の向き）
BOX_FLOOR = 0.02
BOX_WALL_H = 0.16
WALL_T = 0.01
TRAY_IN = (0.40, 0.30)
TRAY_RIM_H = 0.018
CARD = (0.72, 0.56, 0.38)
GRAY = (0.80, 0.80, 0.78)
TRAY_COLOR = (0.55, 0.57, 0.60)

MARGIN = 0.008  # 爪の先と本の縁の隙間（下ろすとき）
MARGIN_L = 0.005  # 左の指（左の爪の先）と本の縁の隙間（下ろすとき）
LIP_CLEAR = 0.0015  # 爪の下面と下の面の隙間（下ろすとき）
PLACE_CLEAR = 0.0005  # 置くとき、爪の下面とトレーの床の隙間（床に押し付けると、床との摩擦で指が爪を抜けなかった）
WITHDRAW_STEPS = 25  # 爪を抜く速さ（25 歩 = 0.25 s で爪の長さ + 12 mm）
TIP_GAP = 0.003  # 押し出すとき、爪の先を面の縁からこれだけ外に置く
BLADE_BELOW = 0.0005  # 押し出すとき、爪の上面を面よりこれだけ低く
PUSH_STEPS = 200
SLIDE_STOP = 0.002  # 押し出したあとの、本の縁と右の指の内面の隙間
RAISE_LIFT = -0.006  # 右の指を上げる量（lift 関節は手先方向が +）
RAISE_TILT_BOTH_DEG = 6.0  # 両側の爪: 右の爪で本を約 6° 傾け、左の側面を左の爪の上へ張り出させる（上げる量は横幅に比例。
# 一律 20 mm だと横幅 105 mm の本は 11° になり、滑らかな本は右の爪の上を滑って立ち上がった）
TABLE_LEN = 0.40  # 平置きの台の長さ（x）。反対側の縁は遠く、両側の爪の押し戻しは使えない
TABLE_EDGE_GAP = 0.01  # 平置き: 本の爪の側の縁から台の縁まで（ばらつきの中心。±1 cm）
SUPPORT_MIN_RATIO = 0.5  # 運ぶ間、本の重さの半分以上を下から支えていること
SQUEEZE = 0.004  # 挟むときの目標の食い込み（力は上限で決まる）
PEEL_PULL = 0.005
CARRY_BOTTOM_Z = BOX_WALL_H + 0.04  # 運ぶときの本の底の高さ
LIFT_STEPS, CARRY_STEPS, LOWER_STEPS = 100, 120, 100
CHECK_EVERY = 2  # ありえるかの判定の間隔（歩）

# 手の形の候補（試行5）。材料はどれも SUS304 の板（hand_design.py の冒頭）。
# 新しい形は指を 2 cm 長くした（0.16 m）: 厚い本（B5 45 mm）を爪に載せたとき、引き込んだ吸盤（gripper z 0.14）に
# 本の上面が当たらないように。指の開き（片側）は (182 + 爪の合計 + 隙間 13 + 下の本のはみ出し 20) / 2 mm 以上:
# B5（182 mm）の下の本が 2 cm はみ出していても、爪の先をその外に下ろせる
_B = dict(tip="bevel", bevel_deg=20.0, upturn_rad=0.0, finger_len_m=0.16, lip_t_m=0.002)
DESIGNS: dict[str, HandDesign] = {d.name: d for d in [
    V2,
    HandDesign(name="R10", lips="right", lip_len_m=0.010, finger_open_m=0.115, **_B),
    HandDesign(name="R25", lips="right", lip_len_m=0.025, finger_open_m=0.122, **_B),
    HandDesign(name="R40", lips="right", lip_len_m=0.040, finger_open_m=0.130, **_B),
    HandDesign(name="R60", lips="right", lip_len_m=0.060, finger_open_m=0.140, **{**_B, "lip_t_m": 0.0025}),
    HandDesign(name="R80", lips="right", lip_len_m=0.080, finger_open_m=0.150, **{**_B, "lip_t_m": 0.003}),
    HandDesign(name="B25", lips="both", lip_len_m=0.025, finger_open_m=0.135, **_B),
    HandDesign(name="B40", lips="both", lip_len_m=0.040, finger_open_m=0.150, **_B),
    HandDesign(name="SPAT", lips="right", lip_len_m=0.100, lip_w_m=0.12, finger_open_m=0.160,
               note="へら: 幅 12 cm・長さ 10 cm の板を本の下へ深く入れる", **{**_B, "lip_t_m": 0.0025, "bevel_deg": 15.0}),
    # 先端の形・厚み・幅・反りの違い（長さ 40 mm の片側で比べる）
    HandDesign(name="R40sq", lips="right", lip_len_m=0.040, finger_open_m=0.130, **{**_B, "tip": "square", "bevel_deg": 0.0}),
    HandDesign(name="R40b35", lips="right", lip_len_m=0.040, finger_open_m=0.130, **{**_B, "bevel_deg": 35.0}),
    HandDesign(name="R40b10", lips="right", lip_len_m=0.040, finger_open_m=0.130, **{**_B, "bevel_deg": 10.0}),
    HandDesign(name="R40up", lips="right", lip_len_m=0.040, finger_open_m=0.130, **{**_B, "upturn_rad": 0.10}),
    HandDesign(name="R40t4", lips="right", lip_len_m=0.040, finger_open_m=0.130, **{**_B, "lip_t_m": 0.004}),
    HandDesign(name="R40w10", lips="right", lip_len_m=0.040, lip_w_m=0.10, finger_open_m=0.130, **_B),
    HandDesign(name="R40rnd", lips="right", lip_len_m=0.040, finger_open_m=0.130, tip="round", upturn_rad=0.14,
               finger_len_m=0.16, lip_t_m=0.003, note="v2 と同じ丸い先端で長さ 40 mm"),
    HandDesign(name="B25t3", lips="both", lip_len_m=0.025, finger_open_m=0.135, **{**_B, "lip_t_m": 0.003}),
    HandDesign(name="B25sq", lips="both", lip_len_m=0.025, finger_open_m=0.135, **{**_B, "tip": "square", "bevel_deg": 0.0}),
    HandDesign(name="B15", lips="both", lip_len_m=0.015, finger_open_m=0.125, **_B),
]}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="book-10")
    ap.add_argument("--surface", choices=list(SURFACES), default=None, help="表面（省略時はカタログのまま）")
    ap.add_argument("--situation", choices=["flat", "stack"], default="stack",
                    help="flat: 台の上に 1 冊 / stack: 箱の中に 3 冊積んだ一番上（隙間なし）")
    ap.add_argument("--designs", default="R25,B25", help=f"カンマ区切り。候補: {','.join(DESIGNS)}")
    ap.add_argument("--suction", choices=["on", "off", "both"], default="both")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter-cm", type=float, default=1.0, help="本の位置のばらつき（±）")
    ap.add_argument("--jitter-deg", type=float, default=5.0, help="本の向きのばらつき（±）")
    ap.add_argument("--render", default="", help="動画と連続写真を撮るセル（カンマ区切り。例 R25:off）")
    # 試行8 と比べるための条件（2026-10-18 追加。省略時は試行5 のまま）
    ap.add_argument("--cone", choices=["pyramidal", "elliptic"], default="pyramidal",
                    help="摩擦円錐。pyramidal（試行5）は摩擦が設定の約半分しか効かない（friction_calib.py）")
    ap.add_argument("--edge-gap-cm", type=float, default=TABLE_EDGE_GAP * 100,
                    help="平置き: 本の爪の側の縁から台の縁まで（試行5 は 1 cm。台の縁を使わない比較は大きくする）")
    ap.add_argument("--table-friction", type=float, default=None, help="台・箱・トレーの摩擦（省略時は Genesis の既定）")
    return ap.parse_args()


@dataclasses.dataclass
class Cell:
    key: str  # 例 "R25:off"
    design: HandDesign
    suction: bool
    x0: float
    robot_ent: object = None
    robot: Robot | None = None
    books: list = dataclasses.field(default_factory=list)  # 下から上の順。最後が取る本
    walls: list = dataclasses.field(default_factory=list)
    supports: list = dataclasses.field(default_factory=list)
    fits: bool = True
    cams: dict = dataclasses.field(default_factory=dict)
    rec: dict = dataclasses.field(default_factory=dict)  # 測定の記録
    frames: list = dataclasses.field(default_factory=list)
    table_edge_x: float = 0.0  # 平置きの台の爪の側の縁（セルの x）
    measure: bool = False  # True の間、下から支える力を測る

    def at(self, x, y, z=0.0):
        return np.array([self.x0 + x, y, z])

    @property
    def target(self):
        return self.books[-1]


EDGE_GAP = [TABLE_EDGE_GAP]  # run() で --edge-gap-cm に置き換える
BOX_FRICTION = [None]  # run() で --table-friction に置き換える


def add_box(scene, size, pos, color):
    mat = gs.materials.Rigid(friction=BOX_FRICTION[0]) if BOX_FRICTION[0] is not None else None
    return scene.add_entity(gs.morphs.Box(size=tuple(float(v) for v in size), pos=tuple(float(v) for v in pos),
                                          fixed=True), material=mat, surface=gs.surfaces.Default(color=color))


def build_cell(scene, cell: Cell, spec, situation: str, out: Path, render: bool) -> None:
    """stack: 箱の中に同じ本を 3 冊。flat: 台の上に 1 冊。台の +x の縁（爪の側）は本の縁の TABLE_EDGE_GAP 外。"""
    px, py = PICK_C
    tx, ty = TRAY_C
    if situation == "stack":
        ix, iy = BOX_IN
        c = cell.at(px, py)
        cell.supports.append(add_box(scene, (ix + 2 * WALL_T, iy + 2 * WALL_T, BOX_FLOOR), c + [0, 0, BOX_FLOOR / 2], CARD))
        for size, off in [((WALL_T, iy + 2 * WALL_T, BOX_WALL_H), (-ix / 2 - WALL_T / 2, 0)),
                          ((WALL_T, iy + 2 * WALL_T, BOX_WALL_H), (ix / 2 + WALL_T / 2, 0)),
                          ((ix, WALL_T, BOX_WALL_H), (0, -iy / 2 - WALL_T / 2)),
                          ((ix, WALL_T, BOX_WALL_H), (0, iy / 2 + WALL_T / 2))]:
            cell.walls.append(add_box(scene, size, c + [off[0], off[1], BOX_WALL_H / 2], CARD))
        n_books = 3
    else:
        cell.table_edge_x = px + spec.width_m / 2 + EDGE_GAP[0]
        tl = TABLE_LEN
        cell.supports.append(add_box(scene, (tl, BOX_IN[1], TABLE_TOP), cell.at(cell.table_edge_x - tl / 2, py, TABLE_TOP / 2),
                                     GRAY))
        n_books = 1
    # 検品トレー（台の上に縁の低いトレー。トレーの床 = 台の上面）
    lx, ly = TRAY_IN
    cell.supports.append(add_box(scene, (lx + 0.06, ly + 0.06, TABLE_TOP), cell.at(tx, ty, TABLE_TOP / 2), GRAY))
    for size, off in [((WALL_T, ly + 2 * WALL_T, TRAY_RIM_H), (-lx / 2 - WALL_T / 2, 0)),
                      ((WALL_T, ly + 2 * WALL_T, TRAY_RIM_H), (lx / 2 + WALL_T / 2, 0)),
                      ((lx, WALL_T, TRAY_RIM_H), (0, -ly / 2 - WALL_T / 2)),
                      ((lx, WALL_T, TRAY_RIM_H), (0, ly / 2 + WALL_T / 2))]:
        cell.walls.append(add_box(scene, size, cell.at(tx + off[0], ty + off[1], TABLE_TOP + TRAY_RIM_H / 2), TRAY_COLOR))
    obj, tex = make_book_assets(spec, out)
    s = SURFACES[spec.surface]
    for i in range(n_books):
        cell.books.append(scene.add_entity(
            gs.morphs.Mesh(file=str(obj), pos=tuple(cell.at(px, py, 0.5 + 0.06 * i))),
            material=gs.materials.Rigid(rho=spec.density, friction=s["friction"]),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(tex)))))
    cell.robot_ent = scene.add_entity(gs.morphs.MJCF(file=str(build_robot_xml(out, design=cell.design)),
                                                     pos=(cell.x0, 0.0, 0.0)))
    if render:
        # 接写: 本の天地の向き（y）から、取る場所の本の縁と爪を見る。置く場所も同じ向きから
        zc = (BOX_FLOOR + 2 * spec.thickness_m) if situation == "stack" else TABLE_TOP
        cell.cams["close"] = scene.add_camera(res=(800, 600), pos=tuple(cell.at(px + 0.06, py - 0.38, zc + 0.28)),
                                              lookat=tuple(cell.at(px, py, zc)), fov=40, env_idx=0)
        cell.cams["place"] = scene.add_camera(res=(800, 600), pos=tuple(cell.at(tx - 0.05, ty + 0.55, TABLE_TOP + 0.25)),
                                              lookat=tuple(cell.at(tx, ty, TABLE_TOP + 0.02)), fov=40, env_idx=0)
        cell.cams["overview"] = scene.add_camera(res=(960, 540), pos=tuple(cell.at(1.45, -0.55, 0.95)),
                                                 lookat=tuple(cell.at(0.45, 0.03, 0.12)), fov=45, env_idx=0)


def place_books(cell: Cell, spec, situation: str, jit_xy, jit_yaw, face_down) -> None:
    """乱数（jit_*）は全セルで同じものを使う（同じ条件で手の形を比べる）。"""
    n = jit_xy.shape[0]
    t = spec.thickness_m
    z = np.full(n, BOX_FLOOR if situation == "stack" else TABLE_TOP)
    px, py = PICK_C
    for i, b in enumerate(cell.books):
        zc = z + t / 2 + 0.002
        pos = np.stack([cell.x0 + px + jit_xy[:, i, 0], py + jit_xy[:, i, 1], zc], -1)
        q = quat_z(jit_yaw[:, i])
        if face_down:
            q = quat_mul(q, np.broadcast_to(np.array([0.0, 1.0, 0.0, 0.0]), (n, 4)))
        b.set_pos(pos)
        b.set_quat(q)
        z = zc + t / 2


# ---------- 歩みを譲るモーション（全セルを同時に進める） ----------

def g_wait(k: int):
    yield k


def g_joints(r: Robot, idx, goal: np.ndarray, steps: int):
    start = r.target[:, idx].copy()
    for k in range(1, steps + 1):
        r.target[:, idx] = start + (goal - start) * (k / steps)
        yield 1


def min_jerk(a: float) -> float:
    return 10 * a ** 3 - 15 * a ** 4 + 6 * a ** 5


def g_cart(r: Robot, pos: np.ndarray, quat: np.ndarray, steps: int, n_way: int = 12, profile=lambda a: a):
    """手先を直線上の中継点に沿って動かす（中継点は profile で時間配分。最小ジャークなら加速度が滑らか）。
    中継点ごとに IK を解き、その間は関節空間で補間する。"""
    p0 = r.gripper.get_pos().cpu().numpy()
    q0 = r.gripper.get_quat().cpu().numpy()
    q1 = np.where((np.sum(q0 * quat, axis=1) < 0)[:, None], -quat, quat)
    per = max(1, steps // n_way)
    prev = r.target[:, :7].copy()
    for w in range(1, n_way + 1):
        a = profile(w / n_way)
        qw = q0 + (q1 - q0) * a
        qw /= np.linalg.norm(qw, axis=1, keepdims=True)
        goal = r.ik(p0 + (pos - p0) * a, qw)
        for k in range(1, per + 1):
            r.target[:, :7] = prev + (goal - prev) * (k / per)
            yield 1
        prev = goal


def g_correct(r: Robot, pos: np.ndarray, quat: np.ndarray, out: dict, iters: int = 2, steps: int = 30):
    """重力によるたわみを実測の誤差だけ目標をずらして打ち消す（tool.Robot.correct と同じ。歩みを譲る版）。"""
    aim = pos.copy()
    for _ in range(iters):
        yield 40
        err = pos - r.gripper.get_pos().cpu().numpy()
        norm = np.linalg.norm(err, axis=1, keepdims=True)
        err = np.where(norm > 0.02, err * (0.02 / np.maximum(norm, 1e-9)), err)
        aim = aim + err
        goal = r.ik(aim, quat)
        start = r.target[:, :7].copy()
        for k in range(1, steps + 1):
            r.target[:, :7] = start + (goal - start) * (k / steps)
            yield 1
    out["err"] = np.linalg.norm(pos - r.gripper.get_pos().cpu().numpy(), axis=1)
    out["aim"] = aim


def run_generators(world: World, gens: list) -> None:
    waits = [0] * len(gens)
    alive = [True] * len(gens)
    while any(alive):
        for i, g in enumerate(gens):
            while alive[i] and waits[i] == 0:
                try:
                    waits[i] = int(next(g))
                except StopIteration:
                    alive[i] = False
        if any(alive):
            world.step(1)
            waits = [max(0, w - 1) for w in waits]


# ---------- 力の上限 ----------

def set_forces(r: Robot, fingers: float | None = None, plunger: float | None = None) -> None:
    f = FORCE.copy()
    if fingers is not None:
        f[FINGERS] = fingers
    if plunger is not None:
        f[PLUNGER] = plunger
    r.e.set_dofs_force_range(-f, f, np.arange(len(f)))


def book_frame(body) -> tuple[np.ndarray, np.ndarray]:
    return body.get_pos().cpu().numpy(), quat_to_mat(body.get_quat().cpu().numpy())


def to_book(p_world, pos, mat):
    return np.einsum("nji,nj->ni", mat, p_world - pos)


def lip_links(cell: Cell) -> list[tuple[str, int, np.ndarray]]:
    """(左右, 爪の付いたリンクの番号, 爪の先端近くの点のリンク座標)。右は finger_r_tip、左は finger_l。"""
    d, e = cell.design, cell.robot_ent
    out = []
    for side, link, sign in (("r", "finger_r_tip", +1), ("l", "finger_l", -1)):
        if d.lips == "none" or (side == "l" and d.lips != "both"):
            continue
        out.append((side, e.get_link(link).idx, np.array(d.lip_probe_local(sign))))
    return out


def lip_points(cell: Cell) -> list[tuple[str, np.ndarray]]:
    """爪の先端近くの点（世界座標）。"""
    solver = cell.robot_ent._solver
    out = []
    for side, idx, loc in lip_links(cell):
        p = solver.get_links_pos(links_idx=[idx]).cpu().numpy().reshape(-1, 3)
        m = quat_to_mat(solver.get_links_quat(links_idx=[idx]).cpu().numpy().reshape(-1, 4))
        out.append((side, p + m @ loc))
    return out


def lip_side_quat(e: np.ndarray) -> np.ndarray:
    """真下向きで、右の指（爪）が水平方向 e の側に来る姿勢（gripper の +y = -e）。"""
    gy = -e
    psi = np.arctan2(gy[:, 0], -gy[:, 1])
    return quat_mul(quat_z(psi), np.broadcast_to(Q_DOWN, psi.shape + (4,)))


def edge_along(e: np.ndarray, center: np.ndarray, axis: np.ndarray, half: float, p0: np.ndarray, blade_half: float):
    """本の中心 p0 から e の向きに測った、ある面（中心 center、法線 ±axis、半幅 half）の縁の位置。
    面が e に対して傾いていると、爪の幅（blade_half）の範囲で縁がずれるので、最も外側を返す。"""
    s = np.sign(np.einsum("ni,ni->n", axis, e))
    nrm = axis * s[:, None]
    fc = center + nrm * half
    cosd = np.clip(np.abs(np.einsum("ni,ni->n", nrm[:, :2], e[:, :2])), 1e-3, 1.0)
    tand = np.sqrt(1 - cosd ** 2) / cosd
    return np.einsum("ni,ni->n", (fc - p0)[:, :2], e[:, :2]) + tand * blade_half


def slide_under_cycle(cell: Cell, spec, n: int, snap, situation: str):
    """1 つのセルの手順（全環境）。測定は cell.rec に入れる。

    隙間を作る方法（剛体の本は下の面と隙間が無く、くさびの先は本の側面を横に押すだけになる。試行5 の最初の実行で
    すべての環境で爪が本の縁の 3.6 mm 外で止まり、本は横から挟まれて持ち上がっていた＝下から支えていない）:
      爪を、本を載せている面（下の本・台）の縁の外に、上面を面より 0.5 mm 低くして置き、
      左の指（両側の爪なら左の爪の先）で本の向こう側を押して、本を爪の上へ押し出す（ちり取りの要領）。
      吸引は使わない。吸引ありのセルでは、押し出したあと吸盤で上から押さえる・端を引く・運ぶ間の落下防止に使う"""
    r, d, rec = cell.robot, cell.design, cell.rec
    book = cell.target
    w, h, t = spec.size_m
    plan = grip_plan(spec)
    w_est = spec.weight_estimated_kg * GRAVITY
    # 押し出す・爪を押し込む力: 推定の重さの 3 倍（8 N 以上・上限は grip_plan）。本を下の本の上で滑らせるのに要るのは
    # 摩擦係数 × 重さ（≦ 0.9 × 重さ）。上限（滑らか 60 N）のまま押すと、軽い本では爪の先が本の側面に 2〜4.5 mm めり込み、
    # 本が跳ねた（試行5 の決勝の最初の実行）
    f_push = float(min(plan["grip_limit_n"], max(8.0, 3.0 * w_est)))
    f_side = float(min(plan["grip_limit_n"], max(3.0, 0.75 * w_est)))  # 挟む力: 推定の重さの 0.75 倍（3 N 以上・上限まで）。重さは下から支える
    f_top = float(np.clip(0.5 * w_est, 2.0, 10.0))  # 吸盤で上から押さえる力（吸引ではなく押し付け）
    rec.update(f_push_n=f_push, f_side_n=f_side, f_top_n=f_top)
    t_start = r.world.sim_time

    p0, m0 = book_frame(book)
    rec["p0"], rec["up0"] = p0.copy(), m0[:, 2, 2].copy()
    xb = m0[:, :, 0].copy()
    xb[:, 2] = 0.0
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    e = xb * np.sign(xb[:, 0])[:, None]  # 爪の側（ロボットから遠い +x 側の本の縁）
    quat = lip_side_quat(e)
    gy = -e
    z_s = p0[:, 2] - t / 2  # 本の底 = 載せている面の高さ
    # 載せている面の、爪の側の縁（本の中心から e の向きに測る）
    if situation == "stack":
        pl, ml = book_frame(cell.books[-2])
        u_s = edge_along(e, pl, ml[:, :, 0], w / 2, p0, d.lip_w_m / 2)
    else:
        ex = np.clip(e[:, 0], 1e-3, None)
        u_s = (cell.x0 + cell.table_edge_x - p0[:, 0]) / ex + np.sqrt(np.clip(1 - e[:, 0] ** 2, 0, 1)) / ex * d.lip_w_m / 2
    # 載せている面の、反対側（左）の縁（e の向きに測る。負の値）。両側の爪の「押し戻し」に使う
    if situation == "stack":
        u_sl = -edge_along(-e, pl, ml[:, :, 0], w / 2, p0, d.lip_w_m / 2)
    else:
        ex = np.clip(e[:, 0], 1e-3, None)
        u_sl = (cell.x0 + cell.table_edge_x - TABLE_LEN - p0[:, 0]) / ex
    u_top = np.full(n, w / 2)  # 取る本の、爪の側の縁
    L_r, L_l, fo = d.lip_len_r, d.lip_len_l, d.finger_open_m
    u_tip = u_s + TIP_GAP  # 押し出すときの爪の先
    u_rf = u_tip + L_r  # そのときの右の指の内面
    u_rf0 = np.maximum(u_top, u_s) + MARGIN + L_r  # 下ろすときの右の指の内面（爪の先が本の縁と面の縁の外）
    u_lf0 = np.full(n, -(w / 2 + L_l + MARGIN_L))  # 下ろすときの左の指の内面
    push = np.clip(u_rf - SLIDE_STOP - w / 2, 0.0, None)  # 本を押し出す量
    u_lf1 = -w / 2 + push - L_l + 0.002  # 押し終えたときの左の指の内面（2 mm 押し付ける）
    # 手のひら（吸盤）の位置: 押し出した後の本の中心より 2 cm 向こう側（爪と反対）に吸盤を当てたい。
    # 置くとき吸盤で押さえる力を、爪ではなく床に載る側へかけるため（爪の側で押さえると、爪を抜くとき本が爪につられて動いた）
    want = push - 0.02
    lo = np.maximum(u_rf0 - fo, u_lf1)
    hi_ = np.minimum(u_lf0 + fo, u_rf)
    fits = (u_rf0 - u_lf0 <= 2 * fo) & (lo <= hi_)
    if situation == "stack":  # 右の指の外面が箱の壁に当たらないか
        wall_u = (cell.x0 + PICK_C[0] + BOX_IN[0] / 2 - p0[:, 0]) / np.clip(e[:, 0], 1e-3, None)
        fits &= u_rf0 + FINGER_PLATE_T + 0.003 < wall_u
    u_p = np.clip(want, lo, np.maximum(lo, hi_))
    rec.update(fits_env=fits, push_mm=push * 1000, support_edge_mm=u_s * 1000, palm_u_mm=u_p * 1000)
    q_r0, q_l0 = np.clip(u_rf0 - u_p, 0, fo), np.clip(u_p - u_lf0, 0, fo)
    # 高さ: 左のいちばん下（左の爪の下面か左の指先）を面の 2 mm 上、右の爪の上面を面の 0.5 mm 下に
    # （右の指の上下の関節で差をつける）
    left_low = d.fingertip_z + (d.lip_t_m if d.lips == "both" else 0.0)
    g_z = z_s + 0.002 + left_low
    right_top = d.lip_lowest_z - (d.lip_t_m if d.lips != "none" else 0.0)
    lift_down = np.clip(np.full(n, left_low + 0.002 + BLADE_BELOW - right_top), -d.lift_stroke_m, d.lift_stroke_m)
    rec["lift_down_mm"] = lift_down * 1000
    palm = p0 + u_p[:, None] * e
    palm[:, 2] = g_z
    rec["cup_gap_mm"] = (g_z - (z_s + t) - CUP_TIP_RETRACTED_Z) * 1000  # 引き込んだ吸盤と本の上面の隙間

    set_forces(r, fingers=f_push, plunger=f_top)
    r.target[:, FINGERS] = np.stack([q_l0, q_r0], -1)
    r.target[:, PLUNGER] = 0.0
    r.target[:, LIFT] = lift_down
    above = palm.copy()
    above[:, 2] += 0.12
    yield from g_cart(r, above, quat, 150)
    yield from g_cart(r, palm, quat, 100, n_way=6)
    cor: dict = {}
    yield from g_correct(r, palm, quat, cor)
    rec["pos_err_mm"] = cor["err"] * 1000
    aim = cor["aim"]
    snap(cell, "1 blade beside the edge")

    # 2. 爪の先を面の縁の 3 mm 外まで寄せる（本が面の縁より出ていれば、その下へ入る）
    yield from g_joints(r, FINGERS[1], np.clip(u_rf - u_p, 0, fo), 50)
    # 3. 左で本の向こう側を押して、本を爪の上へ押し出す
    pb_before = book.get_pos().cpu().numpy()
    yield from g_joints(r, FINGERS[0], np.clip(u_p - u_lf1, 0, fo), PUSH_STEPS)
    yield 20
    pb, mb = book_frame(book)
    rec["push_actual_mm"] = np.einsum("ni,ni->n", (pb - pb_before)[:, :2], e[:, :2]) * 1000
    snap(cell, "2 pushed onto blade")

    # 5. 右の爪を上げて下から支える。両側の爪は、傾いた本の左の側面へ左の爪を押し込む
    if d.lips == "both":
        # 両側の爪の左: (1) 左の面の縁が手の届く範囲にあれば「押し戻し」: 左の爪の先を左の面の縁の 3 mm 外、上面を面の
        # 0.5 mm 下に置き、右の指を閉じて（右の爪は本の下に入ったまま一緒に動く）本を左の爪の上へ押し戻す。表面によらない。
        # (2) 届かなければ（広い台）、右の爪で本を約 6° 傾け、左の爪を面に沿って左の縁へ押し込む（くさび。滑らかな本は入らない）
        pb, _ = book_frame(book)
        u_book_l = np.einsum("ni,ni->n", (pb - p0)[:, :2], e[:, :2]) - w / 2  # 押し出した後の本の左の縁
        u_lf2 = u_sl - TIP_GAP - L_l  # 左の爪の先を左の面の縁の外に置いたときの左の指の内面
        back = np.clip(u_book_l - (u_lf2 + SLIDE_STOP), 0.0, None)  # 本を戻す量（左の縁が左の指の内面の 2 mm 手前まで）
        q_l2 = u_p - u_lf2
        q_r2 = r.target[:, FINGERS[1]] - back
        mirror = (q_l2 <= fo) & (q_r2 >= 0.0) & (back < 0.15)
        rec["mirror_push"] = mirror
        rec["push_back_mm"] = np.where(mirror, back, 0.0) * 1000
        if mirror.any():
            # 手を (2.5 mm + 爪の厚み) 下げて、左の爪の上面を面の 0.5 mm 下へ。右の爪は関節で上げ、下面を面の 0.5 mm 上に
            # （戻すとき右の爪は下の本の上を通るので、面より低いと下の本の縁に当たる）
            drop = 0.0025 + d.lip_t_m
            yield from g_joints(r, FINGERS[0], np.where(mirror, np.clip(q_l2, 0, fo), r.target[:, FINGERS[0]]), 40)
            lower = aim.copy()
            lower[:, 2] -= np.where(mirror, drop, 0.0)
            lift_keep = r.target[:, LIFT] - np.where(mirror, drop + 0.001 + d.lip_t_m, 0.0)
            yield from g_joints(r, LIFT, lift_keep, 20)
            yield from g_cart(r, lower, quat, 30, n_way=3)
            yield from g_joints(r, FINGERS[1], np.where(mirror, np.clip(q_r2, 0, fo), r.target[:, FINGERS[1]]), PUSH_STEPS)
            yield 20
        if (~mirror).any():
            raise_w = float(np.clip(w * math.tan(math.radians(RAISE_TILT_BOTH_DEG)), 0.008, 0.020))
            yield from g_joints(r, LIFT, np.where(mirror, r.target[:, LIFT], np.clip(r.target[:, LIFT] - raise_w, -d.lift_stroke_m, d.lift_stroke_m)), 60)
            lower = aim.copy()
            lower[:, 2] -= 0.0025  # 左の爪の下面を面の 0.5 mm 下へ（へらのように面に軽く押し付ける）
            yield from g_cart(r, lower, quat, 30, n_way=3)
            q_now = r.target[:, FINGERS[0]].copy()
            yield from g_joints(r, FINGERS[0], np.where(mirror, q_now, np.clip(q_now - L_l, 0, fo)), 150)
            yield 10
        yield from g_cart(r, aim, quat, 30, n_way=3)
        # 両方の爪が入ったら、右の爪を 6 mm 上げて下から支える（左は爪の上面で支える）
        yield from g_joints(r, LIFT, np.clip(r.target[:, LIFT] + RAISE_LIFT, -d.lift_stroke_m, d.lift_stroke_m), 40)
    else:
        yield from g_joints(r, LIFT, np.clip(lift_down + RAISE_LIFT, -d.lift_stroke_m, d.lift_stroke_m), 60)
    yield 10
    pb, mb = book_frame(book)
    depth = {}
    under = {}
    for side, p in lip_points(cell):
        loc = to_book(p, pb, mb)
        depth[side] = (w / 2 - np.abs(loc[:, 0])) * 1000  # 本の縁から爪の先の点までの深さ（mm。負なら外）
        under[side] = (np.abs(loc[:, 0]) < w / 2 - 0.002) & (loc[:, 2] < -t / 2 + 0.001)
    rec["insert_depth_mm"] = depth
    rec["inserted"] = under.get("r", np.zeros(n, bool))
    rec["inserted_l"] = under.get("l", np.ones(n, bool))
    snap(cell, "3 blade under, raised")

    # 5b. 吸引あり: 爪が入ったあと、吸盤を上面に当てて吸着し、少し引く（押さえ・落下防止の補助）。
    # 爪を入れる前に吸着すると、両側の爪で本が押さえられて左の爪が入らなかった（試行5 の決勝の最初の実行で 1/16）
    top_gz = np.einsum("ni,ni->n", pb - r.gripper.get_pos().cpu().numpy(), r.approach_axis()) - t / 2
    if cell.suction:
        r.suction.attach(book, SURFACES[spec.surface]["suction_max_n"])
        q_cup = np.clip(top_gz - CUP_TIP_RETRACTED_Z + 0.003, 0.0, PLUNGER_STROKE)
        yield from g_joints(r, PLUNGER, q_cup, 40)
        r.suction.on[:] = True
        yield 15
        yield from g_joints(r, PLUNGER, np.clip(q_cup - PEEL_PULL, 0.0, PLUNGER_STROKE), 40)
        yield 10

    # 6. 左右から軽く挟み、上から吸盤で押さえる
    set_forces(r, fingers=f_side, plunger=f_top)
    g_now = r.gripper.get_pos().cpu().numpy()
    s_book = np.einsum("ni,ni->n", pb - g_now, gy)  # 本の中心の、手のひらからの位置（gripper の y）
    q_sq = np.stack([s_book + w / 2 - SQUEEZE, -s_book + w / 2 - SQUEEZE], -1)
    yield from g_joints(r, FINGERS, np.clip(q_sq, 0, fo), 40)
    top_gz = np.einsum("ni,ni->n", pb - g_now, r.approach_axis()) - t / 2
    yield from g_joints(r, PLUNGER, np.clip(top_gz - CUP_TIP_RETRACTED_Z + 0.004, 0.0, PLUNGER_STROKE), 40)
    yield 20

    # 7. 持ち上げる → 運ぶ（運ぶ間、下から支えている力の割合を測る）
    g_now = r.gripper.get_pos().cpu().numpy()
    pb = book.get_pos().cpu().numpy()
    up = g_now.copy()
    up[:, 2] = CARRY_BOTTOM_Z + (g_now[:, 2] - (pb[:, 2] - t / 2))
    yield from g_cart(r, up, quat, LIFT_STEPS, n_way=10, profile=min_jerk)
    yield 20
    pb = book.get_pos().cpu().numpy()
    lifted = (pb[:, 2] - p0[:, 2] > 0.05) & touching_robot(cell)
    rec["lifted"] = lifted
    rec["lift_gain_mm"] = (pb[:, 2] - p0[:, 2]) * 1000
    g_now = r.gripper.get_pos().cpu().numpy()
    off = pb - g_now
    snap(cell, "4 lifted")
    dest = cell.at(*TRAY_C)
    goal = g_now.copy()
    goal[:, :2] = dest[:2] - off[:, :2]
    rec["carry_m"] = np.linalg.norm(goal[:, :2] - g_now[:, :2], axis=1)
    cell.measure = True
    yield from g_cart(r, goal, quat, CARRY_STEPS, n_way=24, profile=min_jerk)
    yield 20
    cell.measure = False
    pb = book.get_pos().cpu().numpy()
    g_now = r.gripper.get_pos().cpu().numpy()
    slip = np.linalg.norm((pb - g_now) - off, axis=1)
    rec["carried"] = lifted & (slip < 0.02) & touching_robot(cell)
    rec["slip_in_carry_mm"] = slip * 1000
    snap(cell, "5 carried over tray")

    # 8. 下ろして、上から押さえたまま爪を抜く
    yield from g_joints(r, LIFT, np.zeros(n), 30)
    g_now = r.gripper.get_pos().cpu().numpy()
    place = g_now.copy()
    place[:, 2] = TABLE_TOP + max(d.lip_lowest_z, left_low) + PLACE_CLEAR
    yield from g_cart(r, place, quat, LOWER_STEPS, n_way=8, profile=min_jerk)
    yield 30
    if cell.suction:
        r.suction.on[:] = False
    yield 10
    snap(cell, "6 lowered onto tray")
    # 爪を抜く力は指の出せる最大（60 N）。本を挟む力ではなく、爪と本の底の摩擦（約 0.9 ×（重さ + 押さえ））に
    # 打ち勝つための力（ざらざらの本の把持の上限 25 N では、B5 で指が止まって爪が抜けなかった）
    set_forces(r, fingers=float(FORCE[FINGERS].max()), plunger=f_top)
    z_rest = book.get_pos().cpu().numpy()[:, 2]
    q_before = r.e.get_qpos().cpu().numpy()[:, FINGERS]
    rec["q_fingers_before_withdraw_mm"] = q_before * 1000
    # 爪の抜き方: 左の指を開き切る → 右の指を開き切る（両側の爪は左右同じ速さで開き、引きずる力を打ち消す）。
    # それでも爪の先が本の下に残れば、吸盤を引いてから手全体を爪の側へ、残りの深さ + 8 mm だけずらす
    # （左の指と本の隙間の範囲で）。抜ける量の合計は「2 × 指の開き − 本の横幅」程度で、長い爪ほど足りなくなる
    q_now = r.target[:, FINGERS].copy()
    if d.lips == "both":
        step_open = np.minimum(fo - q_now[:, 0], fo - q_now[:, 1])
        yield from g_joints(r, FINGERS, q_now + step_open[:, None], WITHDRAW_STEPS)
    else:
        yield from g_joints(r, FINGERS[0], np.full(n, fo), 20)
    yield from g_joints(r, FINGERS, np.full((n, 2), fo), WITHDRAW_STEPS)
    yield 10
    rec["q_fingers_after_withdraw_mm"] = r.e.get_qpos().cpu().numpy()[:, FINGERS] * 1000
    pb, mb = book_frame(book)
    still = np.zeros(n)
    for side, p in lip_points(cell):
        loc = to_book(p, pb, mb)
        if side == "r":
            still = np.where(loc[:, 2] < 0, w / 2 - np.abs(loc[:, 0]) + 0.004, 0.0)  # 先端までの残りの深さ（探る点は先端の 4 mm 手前）
    yield from g_joints(r, PLUNGER, np.zeros(n), 30)
    left_room = np.clip(fo - q_before[:, 0] - 0.003, 0.0, None)  # 左の指の内面と本の左の側面の隙間（開き切ったとき）
    shift = np.where(still > 0, np.minimum(still + 0.008, left_room), 0.0)
    rec["release_arm_shift_mm"] = shift * 1000
    rec["withdraw_short_mm"] = np.clip(still + 0.008 - left_room, 0.0, None) * np.where(still > 0, 1.0, 0.0) * 1000
    if (shift > 0).any():
        g_now = r.gripper.get_pos().cpu().numpy()
        yield from g_cart(r, g_now + shift[:, None] * e, quat, 40, n_way=4)
        yield 10
    rise = np.zeros(n)
    touch_names: set = set()  # 手を上げる間に本に触れていたリンク（環境 0。引きずり上げの原因を調べるため）
    leave = r.gripper.get_pos().cpu().numpy()
    for k in range(1, 9):  # 最初はゆっくり上げ、本が手につられて持ち上がらないかを見る
        lk = leave.copy()
        lk[:, 2] += 0.12 * k / 8
        yield from g_cart(r, lk, quat, 25 if k <= 2 else 10, n_way=1)
        rise = np.maximum(rise, book.get_pos().cpu().numpy()[:, 2] - z_rest)
        ci = cell.robot_ent.get_contacts(with_entity=book)
        names = {lk.idx: lk.name for lk in cell.robot_ent.links}
        for la_, lb_, v_ in zip(ci["link_a"][0].tolist(), ci["link_b"][0].tolist(), ci["valid_mask"][0].tolist()):
            if v_:
                touch_names.add(names.get(la_, names.get(lb_, "?")))
    rec["release_rise_mm"] = rise * 1000
    rec["release_touch_links_env0"] = sorted(touch_names)
    rec["cycle_s"] = r.world.sim_time - t_start
    yield 80
    snap(cell, "7 released, settled")


def touching_robot(cell: Cell) -> np.ndarray:
    ci = cell.robot_ent.get_contacts(with_entity=cell.target)
    return (ci["valid_mask"].cpu().numpy()).any(axis=1)


class Probe:
    """毎歩の後に呼ぶ（World.monitors）。全セルの本の接触・姿勢を見て、ありえるかを判定する。"""

    def __init__(self, scene, world: World, cells: list[Cell], n: int):
        self.scene, self.world, self.cells, self.n = scene, world, cells, n
        self.active = False
        self.k = 0
        solver = scene.rigid_solver
        n_links = solver.n_links
        self.cat = np.zeros(n_links + 1, int)  # 0 なし / 1 ロボット / 2 壁 / 3 台・床・本
        self.hand = np.zeros(n_links + 1, bool)  # 力・めり込みを数える手のリンク
        self.book_links = []
        self.book_cell = []
        self.hand_links = []
        for ci, c in enumerate(cells):
            e = c.robot_ent
            self.cat[e.link_start:e.link_end] = 1
            for name in ("gripper", "finger_l", "finger_r", "finger_r_tip", "plunger"):
                self.hand[e.get_link(name).idx] = True
                self.hand_links.append(e.get_link(name).idx)
            for wl in c.walls:
                self.cat[wl.link_start:wl.link_end] = 2
            for sp in c.supports:
                self.cat[sp.link_start:sp.link_end] = 3
            for b in c.books:
                self.cat[b.link_start:b.link_end] = 3
                self.book_links.append(b.base_link.idx)
                self.book_cell.append(ci)
        self.cat[n_links] = 0  # 番兵（-1 の詰め物を n_links に写す）
        self.book_links = np.array(self.book_links)
        self.book_cell = np.array(self.book_cell)
        self.n_links = n_links
        self.mons = [Monitor(n, len(c.books), Rules()) for c in cells]
        self.lips = [(ci, idx, loc) for ci, c in enumerate(cells) for _, idx, loc in lip_links(c)]
        self.lip_idx = np.array([idx for _, idx, _ in self.lips], int)
        self.support = [[] for _ in cells]  # 運ぶ間の「下から支える力 / 重さ」（環境ごと、測るたび）
        self.mass = [c.rec["mass"] for c in cells]
        self.total_force = [np.zeros(n) for _ in cells]

    def __call__(self):
        self.k += 1
        if not self.active or self.k % CHECK_EVERY:
            return
        solver = self.scene.rigid_solver
        cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        la = cd["link_a"].cpu().numpy()
        lb = cd["link_b"].cpu().numpy()
        if la.ndim == 1:
            la, lb = la[None], lb[None]
        pen = cd["penetration"].cpu().numpy().reshape(la.shape)
        fvec = cd["force"].cpu().numpy().reshape(la.shape + (3,))  # geom B に働く力
        cpos = cd["position"].cpu().numpy().reshape(la.shape + (3,))
        force = np.linalg.norm(fvec, axis=-1)
        valid = (la >= 0) & (lb >= 0)
        la = np.where(valid, la, self.n_links)
        lb = np.where(valid, lb, self.n_links)
        ca, cb = self.cat[la], self.cat[lb]
        ha, hb = self.hand[la], self.hand[lb]
        bl = self.book_links
        is_a = la[:, :, None] == bl[None, None, :]  # (n, c, B)
        is_b = lb[:, :, None] == bl[None, None, :]
        other_cat = np.where(is_a, cb[:, :, None], 0) + np.where(is_b, ca[:, :, None], 0)
        other_hand = (is_a & hb[:, :, None]) | (is_b & ha[:, :, None])
        touch_robot = (other_cat == 1).any(axis=1)
        touch_wall = (other_cat == 2).any(axis=1)
        touch_sup = (other_cat == 3).any(axis=1)
        pen_hand = np.where(other_hand, pen[:, :, None], 0.0).max(axis=1, initial=0.0)
        f_hand = np.where(other_hand, force[:, :, None], 0.0).sum(axis=1)  # 手から本への力の合計（全リンク）
        other_link = np.where(is_a, lb[:, :, None], la[:, :, None])
        f_link = np.zeros_like(f_hand)  # 手のリンクごとの力の合計の最大（傷みの目安）
        for hl in self.hand_links:
            f_link = np.maximum(f_link, np.where(other_hand & (other_link == hl), force[:, :, None], 0.0).sum(axis=1))
        for ci in range(len(self.cells)):
            sel = np.nonzero(self.book_cell == ci)[0]
            self.total_force[ci] = np.maximum(self.total_force[ci], f_hand[:, sel].max(axis=1))
        pos = solver.get_links_pos(links_idx=bl).cpu().numpy().reshape(self.n, len(bl), 3)
        quat = solver.get_links_quat(links_idx=bl).cpu().numpy().reshape(self.n, len(bl), 4)
        up_z = quat_to_mat(quat)[:, :, 2, 2]
        if len(self.lip_idx):
            lp = solver.get_links_pos(links_idx=self.lip_idx).cpu().numpy().reshape(self.n, len(self.lip_idx), 3)
            lm = quat_to_mat(solver.get_links_quat(links_idx=self.lip_idx).cpu().numpy().reshape(self.n, len(self.lip_idx), 4))
        t = self.world.sim_time
        dt = CHECK_EVERY * DT
        for ci, c in enumerate(self.cells):
            sel = np.nonzero(self.book_cell == ci)[0]
            inside = np.zeros((self.n, len(sel)), bool)
            # 爪の先端の点が取る本の箱の中にあるか（すり抜けたときは接触として出ないため）
            j = len(sel) - 1
            bm = quat_to_mat(quat[:, sel[j]])
            hw, hh, ht = (v / 2 for v in c.rec["size"])
            for k, (cj, _, loc) in enumerate(self.lips):
                if cj != ci:
                    continue
                p = lp[:, k] + np.einsum("nij,j->ni", lm[:, k], loc)
                lc = to_book(p, pos[:, sel[j]], bm)
                inside[:, j] |= ((np.abs(lc[:, 0]) < hw - 0.002) & (np.abs(lc[:, 1]) < hh - 0.002)
                                 & (np.abs(lc[:, 2]) < ht - 0.0015))
            if c.measure:
                # 下から支える力: 手と本の接触のうち、本の下半分（本の座標 z < -t/4）にあるものの、本に働く力の上向き成分
                bl_i = bl[sel[j]]
                on_b = (lb == bl_i) & ha
                on_a = (la == bl_i) & hb
                fz = np.where(on_b, fvec[..., 2], 0.0) - np.where(on_a, fvec[..., 2], 0.0)
                zb = np.einsum("nci,ni->nc", cpos - pos[:, None, sel[j]], bm[:, :, 2])
                below = zb < -c.rec["size"][2] / 4
                self.support[ci].append(np.where((on_a | on_b) & below, fz, 0.0).sum(axis=1)
                                        / (self.mass[ci] * GRAVITY))
            self.mons[ci].update(t, dt, up_z[:, sel], touch_robot[:, sel], touch_wall[:, sel], touch_sup[:, sel],
                                 penetration_m=pen_hand[:, sel], inside=inside, hand_force_n=f_link[:, sel])


def montage(frames, path: Path) -> None:
    if not frames:
        return
    h, w = frames[0][1].shape[:2]
    cols = 4
    rows = math.ceil(len(frames) / cols)
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for k, (label, f) in enumerate(frames):
        x, y = (k % cols) * w, (k // cols) * h
        canvas.paste(Image.fromarray(f), (x, y))
        dr.rectangle([x, y, x + 7 * len(label) + 12, y + 20], fill=(255, 255, 255))
        dr.text((x + 6, y + 4), label, fill=(0, 0, 0))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    run_id, out, meta = new_run(TRIAL, dict(vars(args)), extra={"genesis": gs.__version__})
    result: dict = {}
    error = None
    ctx: dict = {"recording": []}
    try:
        result = run(args, out, ctx)
    except BaseException as e:  # 状態 failed として記録してから、そのまま上げ直す
        error = e
        result = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        raise
    finally:
        for cam in ctx["recording"]:
            cam.stop_recording()
        result["status"] = classify(bool(result.get("acceptance_ok", False)), error)
        save_result(out, meta, result)
        print(f"run_id={run_id} status={result['status']} out={out}")


def run(args, out: Path, ctx: dict) -> dict:
    catalog = load_catalog(DEFAULT_CATALOG)
    spec = catalog.by_id(args.book)
    if args.surface is not None:
        spec = dataclasses.replace(spec, surface=args.surface)
    n = args.n_envs
    names = [s for s in args.designs.split(",") if s]
    modes = {"on": [True], "off": [False], "both": [True, False]}[args.suction]
    render = set(s for s in args.render.split(",") if s)
    rng = np.random.default_rng(args.seed)
    EDGE_GAP[0] = args.edge_gap_cm / 100
    BOX_FRICTION[0] = args.table_friction
    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=4),
                     rigid_options=gs.options.RigidOptions(enable_self_collision=False,
                                                           friction_cone=getattr(gs.friction_cone, args.cone)),
                     vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    cells: list[Cell] = []
    for name in names:
        for on in modes:
            key = f"{name}:{'on' if on else 'off'}"
            c = Cell(key, DESIGNS[name], on, len(cells) * CELL_DX)
            build_cell(scene, c, spec, args.situation, out, key in render)
            cells.append(c)
    t_build = time.time()
    scene.build(n_envs=n)
    build_s = time.time() - t_build
    world = World(scene, n, DT)
    for c in cells:
        c.robot = Robot(world, c.robot_ent, c.key, home=home_for(c.design), base_xy=(c.x0, 0.0))
        c.robot.setup()
        c.fits = c.design.fits(spec.width_m, MARGIN)
        c.rec["size"] = spec.size_m
        c.rec["mass"] = spec.weight_kg
    n_books = len(cells[0].books)
    jit_xy = np.zeros((n, n_books, 2))
    jc, jd = args.jitter_cm / 100, math.radians(args.jitter_deg)
    jit_xy[..., 0] = rng.uniform(-jc, jc, (n, n_books))
    jit_xy[..., 1] = rng.uniform(-jc, jc, (n, n_books))
    jit_yaw = rng.uniform(-jd, jd, (n, n_books))
    for c in cells:
        place_books(c, spec, args.situation, jit_xy, jit_yaw, face_down=False)
    world.step(100)
    probe = Probe(scene, world, cells, n)
    world.monitors.append(probe)
    for c in cells:
        for name, cam in c.cams.items():
            if name in ("close", "place"):
                cam.start_recording(save_to_filename=str(out / f"{name}_{c.key.replace(':', '_')}.mp4"), fps=25)
                ctx["recording"].append(cam)

    def snap(cell: Cell, label: str) -> None:
        cell.rec.setdefault("phase_t", {})[label] = world.sim_time
        if "close" in cell.cams:
            cam = cell.cams["place"] if label[0] in "567" else cell.cams["close"]
            rgb, *_ = cam.render()
            cell.frames.append((label, np.asarray(rgb)))

    p_lower0 = [[b.get_pos().cpu().numpy() for b in c.books[:-1]] for c in cells]
    probe.active = True
    t_sim0 = world.sim_time
    t_wall0 = time.time()
    run_generators(world, [slide_under_cycle(c, spec, n, snap, args.situation) if c.fits else g_wait(1) for c in cells])
    wall_s = time.time() - t_wall0
    probe.active = False

    rows = []
    cell_res = {}
    lx, ly = TRAY_IN
    t = spec.thickness_m
    for ci, c in enumerate(cells):
        mon = probe.mons[ci]
        rec = c.rec
        if not c.fits:
            mode_list = ["no_fit"] * n
            success_raw = np.zeros(n, bool)
            success = success_raw
            placed = success_raw
        else:
            pe, me = book_frame(c.target)
            up_end = me[:, 2, 2]
            corners = np.array([[sx * spec.width_m / 2, sy * spec.height_m / 2] for sx in (-1, 1) for sy in (-1, 1)])
            cw = pe[:, None, :2] + np.einsum("nij,kj->nki", me[:, :2, :2], corners)
            tc = c.at(*TRAY_C)[:2]
            inside = ((np.abs(cw[:, :, 0] - tc[0]) <= lx / 2) & (np.abs(cw[:, :, 1] - tc[1]) <= ly / 2)).all(axis=1)
            flat = (np.abs(up_end) > 0.97) & (np.sign(up_end) == np.sign(rec["up0"]))
            on_floor = np.abs(pe[:, 2] - (TABLE_TOP + t / 2)) < 0.004
            released = ~touching_robot(c)
            placed = inside & flat & on_floor & released
            lower_ok = np.ones(n, bool)
            for b, p_start in zip(c.books[:-1], p_lower0[ci]):
                pb, mb = book_frame(b)
                lower_ok &= (np.abs(mb[:, 2, 2]) > 0.97) & (np.linalg.norm(pb - p_start, axis=1) < 0.03)
            sup = np.array(probe.support[ci]) if probe.support[ci] else np.zeros((1, n))
            sup_med = np.median(sup, axis=0)
            from_below = rec["inserted"] & rec["inserted_l"] & (sup_med >= SUPPORT_MIN_RATIO)
            rec["support_ratio_median"] = sup_med
            rec["from_below"] = from_below
            fits_env = rec["fits_env"]
            clean_release = rec["release_rise_mm"] <= 5.0  # 放すとき本を 5 mm 以上引きずり上げない
            success_raw = fits_env & rec["lifted"] & rec["carried"] & placed & lower_ok & from_below & clean_release
            success = success_raw & mon.ok
            mode_list = []
            for i in range(n):
                if success[i]:
                    mode_list.append("ok")
                elif success_raw[i]:
                    kinds = [k for k in KINDS if mon.violated[k][i]]
                    mode_list.append("implausible:" + "+".join(kinds))
                elif not fits_env[i]:
                    mode_list.append("no_fit")
                elif not (rec["inserted"][i] and rec["inserted_l"][i]):
                    mode_list.append("not_inserted" + ("" if not rec["lifted"][i] else "(pinch_lift)"))
                elif not rec["lifted"][i]:
                    mode_list.append("lift_failed")
                elif not rec["carried"][i]:
                    mode_list.append("dropped_in_carry")
                elif not inside[i]:
                    mode_list.append("place_outside")
                elif rec["withdraw_short_mm"][i] > 0:
                    mode_list.append("cannot_withdraw")
                elif rec["release_rise_mm"][i] > 5.0:
                    mode_list.append("dragged_on_release")
                elif not (flat[i] and on_floor[i]):
                    mode_list.append("place_not_flat")
                elif not released[i]:
                    mode_list.append("not_released")
                elif not lower_ok[i]:
                    mode_list.append("stack_disturbed")
                elif not from_below[i]:
                    mode_list.append("held_by_pinch(support<0.5)")
                else:
                    mode_list.append("other")
            rec.update(placed=placed, place_inside=inside, place_flat=flat, place_on_floor=on_floor,
                       lower_ok=lower_ok, end_pos=pe, end_up_z=up_end)
        pe_env = mon.per_env()
        cell_res[c.key] = {
            "design": c.design.name, "suction_assist": c.suction, "fits": c.fits,
            "success_n": int(success.sum()), "success_raw_n": int(success_raw.sum()),
            "implausible_n": int((success_raw & ~mon.ok).sum()), "violations_any_n": int((~mon.ok).sum()),
            "violations": {k: int(mon.violated[k].sum()) for k in KINDS},
            "failure_modes": {m: mode_list.count(m) for m in sorted(set(mode_list))},
            "peak_hand_force_n": {"median": float(np.median(mon.peak_hand_force_n)), "max": float(mon.peak_hand_force_n.max())},
            "max_penetration_mm": float(mon.max_penetration_m.max() * 1000),
            "peak_total_hand_force_n": {"median": float(np.median(probe.total_force[ci])), "max": float(probe.total_force[ci].max())},
            "cycle_s": float(rec.get("cycle_s", float("nan"))),
            "forces_n": {k: rec.get(k) for k in ("f_push_n", "f_side_n", "f_top_n")},
            "per_env": {"mode": mode_list, **{k: v for k, v in rec.items() if k not in ("size", "p0", "off_before_carry")},
                        **pe_env},
        }
        rows.append({
            "run_book": spec.book_id, "surface": spec.surface, "situation": args.situation, "cell": c.key,
            "design": c.design.name, "suction": "on" if c.suction else "off", "n": n,
            "success": int(success.sum()), "success_raw": int(success_raw.sum()),
            "implausible": int((success_raw & ~mon.ok).sum()),
            **{f"viol_{k}": int(mon.violated[k].sum()) for k in KINDS},
            "peak_force_med_n": round(float(np.median(mon.peak_hand_force_n)), 1),
            "peak_force_max_n": round(float(mon.peak_hand_force_n.max()), 1),
            "max_pen_mm": round(float(mon.max_penetration_m.max() * 1000), 2),
            "cycle_s": round(float(rec.get("cycle_s", float("nan"))), 2),
            "support_med": round(float(np.median(rec["support_ratio_median"])), 2) if "support_ratio_median" in rec else "",
            "modes": ";".join(f"{m}={mode_list.count(m)}" for m in sorted(set(mode_list))),
        })
        if c.frames:
            montage(c.frames, out / f"montage_{c.key.replace(':', '_')}.png")
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return {
        "criterion": "5_slideunder",
        "acceptance": "持ち上がった・運ぶ間に落ちない・トレーの中に平ら・下の本を乱さない、かつありえない状態が無い",
        "acceptance_ok": any(v["success_n"] >= math.ceil(0.8 * n) for v in cell_res.values()),
        "n_envs": n,
        "book": {"book_id": spec.book_id, "format": spec.format_name, "size_mm": [round(v * 1000, 1) for v in spec.size_m],
                 "weight_g": round(spec.weight_kg * 1000, 1), "weight_est_g": round(spec.weight_estimated_kg * 1000, 1),
                 "surface": spec.surface},
        "situation": args.situation,
        "physics": {"friction_cone": args.cone, "edge_gap_m": EDGE_GAP[0], "table_friction": BOX_FRICTION[0]},
        "plausibility_rules": Rules().describe(),
        "designs": {c.design.name: c.design.params() for c in cells},
        "cells": cell_res,
        "carry": {"distance_m_median": float(np.median(next((c.rec["carry_m"] for c in cells if "carry_m" in c.rec),
                                                              np.array([np.nan])))),
                  "duration_s": CARRY_STEPS * DT, "profile": "min_jerk",
                  "peak_acc_m_s2_planned": round(5.7735 * 0.34 / (CARRY_STEPS * DT) ** 2, 2)},
        "wall_build_s": build_s, "wall_run_s": wall_s, "sim_s": world.sim_time - t_sim0,
    }


if __name__ == "__main__":
    main()
