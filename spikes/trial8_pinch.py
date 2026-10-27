"""試行8: 背の側で挟む手（PinchHand）で、本を浮かせて爪を下に入れ、板と爪で挟んで運ぶ。

背景（2026-10-17・18 のユーザー判断。Obsidian Projects/PhysicalAI最終課題/02_確認事項 A0〜A5・B5・B6）
  - 背の側の指: 外側 2 本のゴムの面＋真ん中 2 本の爪（内側へ出す量＝爪の長さ）。小口の側の指: ゴムを貼った壁
  - 小口の側を下から支えたり、こじったりしない（コミックは小口が弱い）。爪は背の側からだけ入れる
  - 板で上から押さえ、爪と挟む。(i) 爪の真上（背の側）／(ii) 本の中央。両方を比べる
  - 吸盤は滑り防止の押さえだけ（この試行では使わない）
  - 台の縁を使わない（平置きは広い台の真ん中に置く）。置き先は仕切りの無い平らな査定台

手順（全セル・全環境で同じ。数値は下の定数）
  1. 本の位置と向きを測る（シミュの真値）。本の +x の縁を背とし、右の指（ゴムと爪）を背の側に置く
  2. 指先（ゴム・壁・爪の下面）を置く面の 1 mm 上まで下ろす
  3. 背の側の指を背の面に寄せて止め（位置の基準）、小口の側の壁で挟む（力 = 推定の重さ ÷ (2μ) × 2、8〜40 N。A3）
  4. 手を 5 mm 上げる。同時に爪の台を 5 mm 下げて、爪だけを置く面の 1 mm 上に残す（A2）
  5. 爪を内側へ出す（本の中心を 10 mm 越える所まで。比べる相手は 15 mm だけ）
  6. 爪の台を上げて、爪の上面を本の底に当てる（0.5 mm 押す）
  7. 板を下ろして本を上から押さえ、爪と挟む（力 = 重さ × 0.75、8 N 以上。A3）。挟む力を重さ × 0.75 に緩める
  8. 持ち上げて査定台の上へ運ぶ（試行5 と同じ 0.34 m・最小ジャーク 1.2 s）。運ぶ間、本の底面にかかる上向きの力を測る
  9. 爪の下面が台の 0.5 mm 上になるまで下ろす。板で押さえたまま爪を引き抜く（背の側のゴムが、本が爪につられて
     背の側へ動くのを止める）。板を上げ、指を開き、ゆっくり手を上げる

判定（環境ごと。A5）: 爪が両方とも本の下に入った・下から支える割合が 0.5 以上・持ち上がり運ぶ間に落ちない・
  浮かせたときの傾きが 5° 以内・爪を出す間の本のずれ（挟む向き）が 3 mm 以内・小口の側の底面（小口から幅の 1/3）に
  手が触れない・査定台の上に平ら・爪を抜くときの本のずれが 5 mm 以内（B5）・放すとき 5 mm 以上引きずり上げない・
  下の本を乱さない。そのうえで、ありえない状態が無い（plausibility.Monitor。違反は implausible として別に数える）

シミュの限界（結果に書く）: 本は剛体の箱（帯・表紙のしなりは表さない）。ゴムは硬い面で摩擦だけ（食い込みは表さない）。
  摩擦は 2 面の大きい方（ざらざらの本 0.9 は、ゴム 0.7 より大きいので 0.9 になる）。爪は本とだけ当たる（hand_design）

使い方: python spikes/trial8_pinch.py --book book-10 --surface rough --situation flat --cells P10:i:center,P10:ii:center --n-envs 4
出力: /workspace/outputs/trial8_pinch/<run_id>/（result.json・cells.csv・close_<cell>.mp4・montage_<cell>.png）
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

import genesis as gs
from pai_book_intake.book_assets import make_book_assets
from pai_book_intake.books import DEFAULT_CATALOG, GRAVITY, SURFACES, grip_plan, load_catalog
from pai_book_intake.geometry import quat_mul, quat_to_mat, quat_z
from pai_book_intake.hand_design import CLAW_RECESS, ENV_BIT, PLATE_BASE_Z, PinchHand
from pai_book_intake.plausibility import KINDS, Monitor, Rules
from pai_book_intake.runlog import classify, new_run, save_result
from pai_book_intake.tool import HOME, KP, KV, Robot, World, build_robot_xml
from trial5_slideunder import (book_frame, g_cart, g_correct, g_joints, g_wait, lip_side_quat, min_jerk, montage,
                               run_generators, to_book)

TRIAL = "trial8_pinch"
DT = 0.01
CELL_DX = 1.6
PICK_C = (0.50, -0.14)
PLACE_C = (0.50, 0.20)  # 査定台（仕切りの無い平らな台）の中心。運ぶ距離 0.34 m（試行5 と同じ）
TABLE_TOP = 0.10
FLAT_TABLE = (0.60, 0.46)  # 平置きの台（本の縁から台の縁まで 20 cm 以上。台の縁は使わない）
PLACE_TABLE = (0.40, 0.30)
BOX_IN = (0.46, 0.34)
BOX_FLOOR = 0.02
BOX_WALL_H = 0.16
WALL_T = 0.01
CARD = (0.72, 0.56, 0.38)
GRAY = (0.80, 0.80, 0.78)

MARGIN_SPINE = 0.012  # 下ろすとき、背とゴムの面の隙間（B6: 背の側 25 mm の内訳）
MARGIN_FORE = 0.012  # 下ろすとき、小口と壁の隙間
TIP_CLEAR = 0.002  # 下ろしたとき、指先（爪の下面）と置く面の隙間。1 mm では、補正後も手の位置が 1 mm ずれた環境で
# 指が下の本に乗り、下の本を動かした
LIFT_GAP = 0.005  # ゴムで浮かせる高さ（A2: 爪 3 mm ＋ 隙間 2 mm）
LIFT_GAP_STEPS = 60
MAX_LIFT = 0.010  # 中心で浮かせる高さの上限
SPINE_ANCHOR_N = 60.0  # 背の側の指は位置の基準（背の面の位置で止め、この力まで保つ）。挟む力は小口の側の壁だけで出す。
# 左右を同じ力で押し合わせると、本と指がひとかたまりで横へ流れた（爪を出す間 +4.5 mm、抜く間 −16 mm）
SPINE_PRESS = 0.001  # 背の側の指の目標を、背の面よりこれだけ内側にする
SQUEEZE = 0.010  # 挟むときの目標の食い込み（力は上限で決まる）。4 mm では片方の指が力の上限まで届かなかった（elliptic で 9 N）
CLAW_PAST_CENTER = 0.010  # 爪を本の中心からさらに出す量（A2-2）
CLAW_SHORT = 0.015  # 比べる相手: 15 mm だけ出す
CLAW_STEPS = 150
CLAW_PRELOAD = 0.002  # 爪の上面を本の底に押し当てる量（爪の台を上げて、本を爪に載せる）。1 mm では山の上の本で
# ゴムの摩擦が重さの半分近くを受けた（正味の支え 0.45〜0.65）
CLAW_FLOOR_CLEAR = 0.0005  # 爪の下面と、下の面（下の本の上面・台）の最小のすき間
CLAW_UNDER_CLEAR = 0.0015  # 爪を出す前に、爪の上面を本の最も低い角よりこれだけ下げる（本が 0.44° 傾き、
# 爪の台も 0.8 mm 遅れて上がったため、奥で爪が本の底をこすって本を 3.6 mm 押した）
SIDE_HOLD_RATIO = 0.2  # 爪に載せた後のゴムの挟む力 = 推定の重さ × これ（2 N 以上）。重さ × 0.75 では摩擦だけで本を
# 持て（最大 1.5 倍）、爪に重さがかからなかった（下から支える割合 0〜0.17）
PLATE_MIN_N = 2.0  # 板で押さえる力の下限。8 N では 250 g 以下の本で板が爪に勝ち、本がゴムの摩擦で押し下げられて
# 下からの支えが負になった（試行8 本試験 20261020T0134〜、book-03/05/07 が全部 0/16）
WITHDRAW_PLATE_N = 2.0  # 爪を抜くときの板の力（軽く当てるだけ。強いと爪と本の摩擦が増え、本を引きずった）
PLATE_PRESS = 0.003  # 板の目標の食い込み（力は上限で決まる）
PLACE_CLEAR = 0.0005  # 置くとき、爪の下面と台の隙間
WITHDRAW_GAP = 0.0005  # 爪を抜く前の、爪の上面と本の底のすき間
WITHDRAW_ITERS = 8
SET_DOWN_PRESS = -0.0005  # 爪を抜いた後に本を台へ下ろすとき、本の底を台の上面よりこれだけ下へ指令する（確実に台に載せる）
WITHDRAW_STEPS = 60
CARRY_BOTTOM_Z = BOX_WALL_H + 0.04
LIFT_STEPS, CARRY_STEPS, LOWER_STEPS = 100, 120, 100
CHECK_EVERY = 2
SUPPORT_MIN_RATIO = 0.5
LIFT_TILT_MAX_DEG = 5.0  # A5
CLAW_SHIFT_MAX_MM = 3.0  # A5
WITHDRAW_SHIFT_MAX_MM = 5.0  # B5
RELEASE_RISE_MAX_MM = 5.0

DESIGNS: dict[str, PinchHand] = {d.name: d for d in [
    PinchHand(name="P10", rubber_mu=1.0, note="ゴム μ1.0（基準）"),
    PinchHand(name="P07", rubber_mu=0.7, note="ゴム μ0.7"),
    PinchHand(name="P13", rubber_mu=1.3, note="ゴム μ1.3"),
    PinchHand(name="P20", rubber_mu=2.0, note="ゴム μ2.0（摩擦が効いているかを確かめる）"),
]}
PLATE_MODES = ("i", "ii")  # i: 爪の真上（背の側） / ii: 本の中央
CLAW_MODES = ("center", "short")
DEFAULT_CELLS = "P10:i:center,P10:ii:center,P10:i:short,P07:i:center,P13:i:center"

# 関節の利得（名前で引く）。腕は tool.py と同じ
GAINS = {  # name: (kp, kv, force)
    "finger_l_joint": (8000, 100, 60), "finger_r_joint": (8000, 100, 60), "finger_r_lift": (8000, 100, 60),
    "claw_joint": (3000, 100, 40), "plunger_joint": (3000, 150, 40),
    "plate_y_joint": (3000, 100, 40), "plate_z_joint": (3000, 100, 40),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="book-10")
    ap.add_argument("--surface", choices=list(SURFACES), default=None, help="表面（省略時はカタログのまま）")
    ap.add_argument("--situation", choices=["flat", "stack"], default="flat",
                    help="flat: 広い台の真ん中に 1 冊 / stack: 箱の中に 3 冊積んだ一番上")
    ap.add_argument("--cells", default=DEFAULT_CELLS, help="手:板:爪 をカンマ区切り。例 P10:i:center")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter-cm", type=float, default=1.0)
    ap.add_argument("--jitter-deg", type=float, default=5.0)
    ap.add_argument("--render", default="", help="動画と連続写真を撮るセル（カンマ区切り）")
    ap.add_argument("--noslip", type=int, default=0,
                    help="Genesis の noslip の反復回数（0 で無効）。既定の摩擦（pyramidal・impratio 1）は力をかけ続けると"
                         "ゆっくり滑る（ゴムで 15 N 挟んで手を 14 mm 上げても本は 2 mm しか浮かなかった）。Genesis の説明が"
                         "把持には 5 を勧めているが、elliptic の方が摩擦が設定どおりに効く（friction_calib.py）")
    ap.add_argument("--cone", choices=["pyramidal", "elliptic"], default="elliptic",
                    help="摩擦円錐。既定の elliptic は摩擦が設定どおりに効く（friction_calib.py: μ1.0 の板で挟むと μ_eff 1.0）。"
                         "pyramidal（Genesis の既定・試行5）は μ_eff 0.5 で、力をかけ続けるとゆっくり滑る。noslip とは併用できない")
    ap.add_argument("--book-prim", action="store_true", help="本を Box（表紙の絵なし）で作る。接触の計算の違いを調べるため")
    ap.add_argument("--impratio", type=float, default=None, help="摩擦と垂直の拘束の硬さの比（省略時は Genesis の既定）")
    ap.add_argument("--timeconst", type=float, default=0.01, help="拘束の時定数 s（0 で最も硬い 2×dt）")
    return ap.parse_args()


@dataclasses.dataclass
class Cell:
    key: str
    design: PinchHand
    plate_mode: str
    claw_mode: str
    x0: float
    robot_ent: object = None
    robot: "PinchRobot | None" = None
    books: list = dataclasses.field(default_factory=list)
    walls: list = dataclasses.field(default_factory=list)
    supports: list = dataclasses.field(default_factory=list)
    cams: dict = dataclasses.field(default_factory=dict)
    rec: dict = dataclasses.field(default_factory=dict)
    frames: list = dataclasses.field(default_factory=list)
    measure: bool = False

    def at(self, x, y, z=0.0):
        return np.array([self.x0 + x, y, z])

    @property
    def target(self):
        return self.books[-1]


class PinchRobot(Robot):
    """Robot の dof の番号・利得を、PinchHand の関節の名前から決める。"""

    def __init__(self, world: World, entity, name: str, design: PinchHand, base_xy):
        self.idx = {n: int(entity.get_joint(n).dofs_idx_local[0]) for n in GAINS}
        n_dofs = entity.n_dofs
        home = np.zeros(n_dofs)
        home[:7] = HOME[:7]
        home[self.idx["finger_l_joint"]] = design.finger_open_m
        home[self.idx["finger_r_joint"]] = design.finger_open_m
        super().__init__(world, entity, name, home=home, base_xy=base_xy)
        self.kp, self.kv, self.force = np.zeros(n_dofs), np.zeros(n_dofs), np.zeros(n_dofs)
        self.kp[:7], self.kv[:7] = KP[:7], KV[:7]
        self.force[:7] = [87, 87, 87, 87, 12, 12, 12]
        for n, (kp, kv, f) in GAINS.items():
            i = self.idx[n]
            self.kp[i], self.kv[i], self.force[i] = kp, kv, f

    def setup(self, grip_limit_n=None) -> None:
        dofs = np.arange(len(self.home))
        self.e.set_dofs_kp(self.kp, dofs)
        self.e.set_dofs_kv(self.kv, dofs)
        self.e.set_dofs_force_range(-self.force, self.force, dofs)
        self.e.set_qpos(self.target)
        self.e.control_dofs_position(self.target)

    def set_forces(self, **named: float) -> None:
        f = self.force.copy()
        for n, v in named.items():
            f[self.idx[n]] = v
        self.e.set_dofs_force_range(-f, f, np.arange(len(f)))

    def i(self, name: str) -> int:
        return self.idx[name]

    def q(self, name: str) -> np.ndarray:
        return self.e.get_qpos().cpu().numpy()[:, self.idx[name]]


TABLE_FRICTION = 0.3  # 台・箱の摩擦。既定（1.0）のままだと本との接触が 2 面の大きい方の 1.0 になり、本の表面の違いが消える


def add_box(scene, size, pos, color):
    """台・箱（爪と当たらない。hand_design の ENV_BIT）。"""
    return scene.add_entity(gs.morphs.Box(size=tuple(float(v) for v in size), pos=tuple(float(v) for v in pos),
                                          fixed=True, contype=ENV_BIT, conaffinity=ENV_BIT),
                            material=gs.materials.Rigid(friction=TABLE_FRICTION),
                            surface=gs.surfaces.Default(color=color))


def build_cell(scene, cell: Cell, spec, situation: str, out: Path, render: bool, book_prim: bool = False) -> None:
    px, py = PICK_C
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
        fx, fy = FLAT_TABLE
        cell.supports.append(add_box(scene, (fx, fy, TABLE_TOP), cell.at(px, py - 0.08, TABLE_TOP / 2), GRAY))
        n_books = 1
    lx, ly = PLACE_TABLE
    cell.supports.append(add_box(scene, (lx, ly, TABLE_TOP), cell.at(*PLACE_C, TABLE_TOP / 2), GRAY))
    obj, tex = make_book_assets(spec, out)
    s = SURFACES[spec.surface]
    for i in range(n_books):
        if book_prim:
            cell.books.append(scene.add_entity(
                gs.morphs.Box(size=spec.size_m, pos=tuple(cell.at(px, py, 0.5 + 0.06 * i))),
                material=gs.materials.Rigid(rho=spec.density, friction=s["friction"]),
                surface=gs.surfaces.Default(color=s["color"])))
            continue
        cell.books.append(scene.add_entity(
            gs.morphs.Mesh(file=str(obj), pos=tuple(cell.at(px, py, 0.5 + 0.06 * i))),
            material=gs.materials.Rigid(rho=spec.density, friction=s["friction"]),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(tex)))))
    cell.robot_ent = scene.add_entity(gs.morphs.MJCF(file=str(build_robot_xml(out, design=cell.design)),
                                                     pos=(cell.x0, 0.0, 0.0)))
    if render:
        zc = (BOX_FLOOR + 2 * spec.thickness_m) if situation == "stack" else TABLE_TOP
        # 接写: 本の天地の向き（y）から、背の側の指と爪を見る
        cell.cams["close"] = scene.add_camera(res=(800, 600), pos=tuple(cell.at(px + 0.10, py - 0.36, zc + 0.20)),
                                              lookat=tuple(cell.at(px + 0.03, py, zc)), fov=40, env_idx=0)
        cell.cams["place"] = scene.add_camera(res=(800, 600), pos=tuple(cell.at(PLACE_C[0] + 0.05, PLACE_C[1] + 0.50, TABLE_TOP + 0.22)),
                                              lookat=tuple(cell.at(*PLACE_C, TABLE_TOP + 0.02)), fov=40, env_idx=0)


def place_books(cell: Cell, spec, situation: str, jit_xy, jit_yaw) -> None:
    n = jit_xy.shape[0]
    t = spec.thickness_m
    z = np.full(n, BOX_FLOOR if situation == "stack" else TABLE_TOP)
    px, py = PICK_C
    for i, b in enumerate(cell.books):
        zc = z + t / 2 + 0.002
        pos = np.stack([cell.x0 + px + jit_xy[:, i, 0], py + jit_xy[:, i, 1], zc], -1)
        b.set_pos(pos)
        b.set_quat(quat_z(jit_yaw[:, i]))  # 本の +x（背）がロボットから遠い +x を向く（±ばらつき）
        z = zc + t / 2


def claw_links(cell: Cell) -> list[tuple[int, np.ndarray]]:
    d, e = cell.design, cell.robot_ent
    idx = e.get_link("claw").idx
    return [(idx, np.array(d.claw_probe_local(s))) for s in (-1, +1)]


def claw_points(cell: Cell) -> list[np.ndarray]:
    solver = cell.robot_ent._solver
    out = []
    for idx, loc in claw_links(cell):
        p = solver.get_links_pos(links_idx=[idx]).cpu().numpy().reshape(-1, 3)
        m = quat_to_mat(solver.get_links_quat(links_idx=[idx]).cpu().numpy().reshape(-1, 4))
        out.append(p + m @ loc)
    return out


def book_contacts_env0(cell: Cell, book, pb: np.ndarray, mb: np.ndarray) -> list:
    """環境 0 で本に触れている接触（相手のリンク名・本の座標での位置 mm・本に働く力 N）。原因を調べるための記録。"""
    solver = book._solver
    cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
    la, lb = cd["link_a"].cpu().numpy(), cd["link_b"].cpu().numpy()
    if la.ndim == 1:
        la, lb = la[None], lb[None]
    pos = cd["position"].cpu().numpy().reshape(la.shape + (3,))
    fv = cd["force"].cpu().numpy().reshape(la.shape + (3,))
    names = {}
    for ent in solver.entities:
        for lk in ent.links:
            names[lk.idx] = lk.name if ent is not book else "BOOK"
    bi = book.base_link.idx
    out = []
    for k in range(la.shape[1]):
        a, b = int(la[0, k]), int(lb[0, k])
        if a < 0 or b < 0 or bi not in (a, b):
            continue
        other = b if a == bi else a
        f_on_book = fv[0, k] if b == bi else -fv[0, k]
        loc = mb.T @ (pos[0, k] - pb)
        out.append({"other": names.get(other, str(other)), "loc_mm": np.round(loc * 1000, 1).tolist(),
                    "f_world_n": np.round(f_on_book, 2).tolist()})
    return out


def claw_top_z(r: "PinchRobot", d: PinchHand) -> np.ndarray:
    """爪の上面の世界 z（手は真下向き。claw リンクの原点は指の付け根の高さ）。"""
    return r.e.get_link("claw").get_pos().cpu().numpy()[:, 2] - (d.finger_len_m - d.claw_t_m)


def book_lowest_z(book, size) -> np.ndarray:
    """本の箱の 8 つの角のうち最も低い z（傾いていると底の片側が下がる）。"""
    pb, mb = book_frame(book)
    w, h, t = size
    corners = np.array([[sx * w / 2, sy * h / 2, sz * t / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (pb[:, None, 2] + np.einsum("nj,kj->nk", mb[:, 2, :], corners)).min(axis=1)


def claw_path_points(spec, d: PinchHand, reach: float) -> np.ndarray:
    """爪が通る範囲（取る本の座標）: 背（+x）から reach だけ内側まで × 爪 2 本の x の範囲（本の y）。格子の点 (k, 2)。"""
    w = spec.width_m
    xs = np.linspace(w / 2, w / 2 - reach, 8)
    ys = np.array([-d.claw_x_out, -d.claw_x_in, d.claw_x_in, d.claw_x_out])
    return np.array([[x, y] for x in xs for y in ys])


def finger_footprint_points(spec, d: PinchHand) -> np.ndarray:
    """指が下りる範囲（取る本の座標）: 背の側の指と、引っ込めた爪の後ろ（背から 13 cm 外まで）・小口の側の壁。"""
    w = spec.width_m
    ys = np.linspace(-d.span_m / 2, d.span_m / 2, 5)
    xs = list(np.linspace(w / 2, w / 2 + MARGIN_SPINE + 0.012 + d.claw_body_len_m, 8)) +         list(np.linspace(-w / 2, -(w / 2 + MARGIN_FORE + d.wall_t_m), 3))
    return np.array([[x, y] for x in xs for y in ys])


def claw_rear_points(spec, d: PinchHand) -> np.ndarray:
    """引っ込めた爪の後ろ（背の外、指の外へはみ出す部分）の点（取る本の座標）。"""
    w = spec.width_m
    xs = np.linspace(w / 2, w / 2 + MARGIN_SPINE + 0.012 + d.claw_body_len_m, 8)
    ys = np.array([-d.claw_x_out, -d.claw_x_in, d.claw_x_in, d.claw_x_out])
    return np.array([[x, y] for x in xs for y in ys])


def bottom_z_over(book, size, pts: np.ndarray) -> np.ndarray:
    """取る本の底面の、本の座標の点 pts (k, 2) での世界 z の最小（環境ごと）。"""
    pb, mb = book_frame(book)
    loc = np.concatenate([pts, np.full((len(pts), 1), -size[2] / 2)], axis=1)
    return (pb[:, None, 2] + np.einsum("nj,kj->nk", mb[:, 2, :], loc)).min(axis=1)


def top_z_under(lower, size, book, pts: np.ndarray) -> np.ndarray:
    """取る本の座標の点 pts の真下で、下の本の上面の世界 z の最大（環境ごと）。"""
    pb, mb = book_frame(book)
    loc = np.concatenate([pts, np.full((len(pts), 1), -size[2] / 2)], axis=1)
    wp = pb[:, None, :] + np.einsum("nij,kj->nki", mb, loc)  # 世界の点 (n, k, 3)
    pl, ml = book_frame(lower)
    nrm = ml[:, :, 2]  # 下の本の厚みの軸（世界）
    c = pl + nrm * size[2] / 2 * np.sign(nrm[:, 2:3])
    nz = np.where(np.abs(nrm[:, 2]) < 1e-6, 1e-6, nrm[:, 2])
    z = c[:, None, 2] - (nrm[:, None, 0] * (wp[..., 0] - c[:, None, 0]) + nrm[:, None, 1] * (wp[..., 1] - c[:, None, 1])) / nz[:, None]
    return z.max(axis=1)


def lower_disp(cell: Cell, rec: dict, label: str) -> None:
    """下の本（取る本以外）の、測り始めからの動き（mm、環境ごとの最大）を記録する。"""
    if len(cell.books) < 2:
        return
    if "_lower0" not in rec:
        rec["_lower0"] = [b.get_pos().cpu().numpy().copy() for b in cell.books[:-1]]
    d = [np.linalg.norm(b.get_pos().cpu().numpy() - p0, axis=1) for b, p0 in zip(cell.books[:-1], rec["_lower0"])]
    rec[f"lower_disp_mm_{label}"] = np.max(np.stack(d), axis=0) * 1000


def hand_env_contacts_env0(cell: Cell) -> list:
    """環境 0 で、手（爪・指・板）が本以外（台・箱・床）に触れている接触（リンク名・相手・力 N）。"""
    solver = cell.robot_ent._solver
    cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
    la, lb = cd["link_a"].cpu().numpy(), cd["link_b"].cpu().numpy()
    fv = cd["force"].cpu().numpy()
    if la.ndim == 2:
        la, lb, fv = la[0], lb[0], fv.reshape(la.shape + (3,))[0]
    mine = {lk.idx: lk.name for lk in cell.robot_ent.links}
    env = {}
    for sp in cell.supports + cell.walls:
        for lk in sp.links:
            env[lk.idx] = "support"
    out = []
    for a, b, f in zip(la, lb, fv):
        if a in mine and b in env:
            out.append({"link": mine[a], "other": env[b], "f": np.round(f, 2).tolist()})
        elif b in mine and a in env:
            out.append({"link": mine[b], "other": env[a], "f": np.round(-f, 2).tolist()})
    return out


def touching_robot(cell: Cell) -> np.ndarray:
    ci = cell.robot_ent.get_contacts(with_entity=cell.target)
    return (ci["valid_mask"].cpu().numpy()).any(axis=1)


def g_raise_keep_claws(r: PinchRobot, aim: np.ndarray, quat: np.ndarray, gap: float, steps: int, book, z_s: np.ndarray,
                       t: float, out: dict, iters: int = 8, gap_fn=None, max_rise: float = 0.010):
    """本の底が置く面から gap 浮くまで手を上げる。爪の台は、手が実際に上がった量だけ毎歩下げる（爪は面の上に残る）。

    腕は位置制御なので、本の重さ（B5 で約 15 N）が手に加わると腕がたわみ、命令した 5 mm がほとんど出ない
    （最初の実行で 0.3 mm）。そこで本の底の高さを測り、足りない分だけ目標を足す（最大 iters 回・1 回 1 cm まで）"""
    li = r.i("finger_r_lift")
    gz0 = r.gripper.get_pos().cpu().numpy()[:, 2]
    lz0 = {nm: r.e.get_link(nm).get_pos().cpu().numpy()[0, 2] for nm in ("gripper", "finger_l", "finger_r", "claw")}
    l0 = r.target[:, li].copy()
    tgt = aim.copy()
    need = np.full(len(z_s), gap)
    hist = []
    for it in range(iters):
        tgt[:, 2] += np.clip(need, 0.0, 0.01)
        goal = r.ik(tgt, quat)
        start = r.target[:, :7].copy()
        n_st = steps if it == 0 else steps // 2
        for k in range(1, n_st + 30 + 1):
            if k <= n_st:
                r.target[:, :7] = start + (goal - start) * (k / n_st)
            r.target[:, li] = l0 + (r.gripper.get_pos().cpu().numpy()[:, 2] - gz0)
            yield 1
        bottom = book.get_pos().cpu().numpy()[:, 2] - t / 2
        now = gap_fn() if gap_fn is not None else bottom - z_s  # 爪が通る範囲のいちばん狭いすき間（与えられれば）
        need = np.minimum(gap - now, max_rise - (bottom - z_s))
        hist.append([round(float(v), 2) for v in now * 1000])
        lz = {nm: r.e.get_link(nm).get_pos().cpu().numpy()[0, 2] for nm in ("gripper", "finger_l", "finger_r", "claw")}
        out.setdefault("raise_link_rise_mm_env0", []).append({k: round(float(v - lz0[k]) * 1000, 2) for k, v in lz.items()})
        if (need < 0.0005).all():
            break
    out["raise_iters"] = it + 1
    out["raise_gap_hist_mm"] = hist
    out["hand_rise_mm"] = (r.gripper.get_pos().cpu().numpy()[:, 2] - gz0) * 1000
    out["raise_cmd_mm"] = (tgt[:, 2] - aim[:, 2]) * 1000


def pinch_cycle(cell: Cell, spec, n: int, snap, situation: str):
    r, d, rec = cell.robot, cell.design, cell.rec
    book = cell.target
    w, h, t = spec.size_m
    plan = grip_plan(spec)
    w_est = spec.weight_estimated_kg * GRAVITY
    f_sq = float(np.clip(w_est / (2 * d.rubber_mu) * 2.0, 8.0, 40.0))  # A3
    f_side = float(min(plan["grip_limit_n"], max(3.0, 0.75 * w_est)))
    f_plate = float(max(PLATE_MIN_N, 0.75 * w_est))
    rec.update(f_squeeze_n=f_sq, f_side_n=f_side, f_plate_n=f_plate)
    t_start = r.world.sim_time
    FL, FR, LI, CL, PY, PZ = (r.i(k) for k in ("finger_l_joint", "finger_r_joint", "finger_r_lift", "claw_joint",
                                                "plate_y_joint", "plate_z_joint"))
    fo = d.finger_open_m

    p0, m0 = book_frame(book)
    rec["p0"], rec["up0"] = p0.copy(), m0[:, 2, 2].copy()
    e = m0[:, :, 0].copy()  # 背の向き（本の +x）
    e[:, 2] = 0.0
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    quat = lip_side_quat(e)  # 右の指（ゴムと爪）が背の側
    z_s = p0[:, 2] - t / 2  # 本の底 = 載せている面
    q_r0 = np.full(n, w / 2 + MARGIN_SPINE)
    q_l0 = np.full(n, w / 2 + MARGIN_FORE)
    fits = (q_r0 <= fo) & (q_l0 <= fo)
    if situation == "stack":  # 指の外面が箱の壁に当たらないか（背の側・小口の側）
        cx = cell.x0 + PICK_C[0]
        u_wall_far = (cx + BOX_IN[0] / 2 - p0[:, 0]) / np.clip(e[:, 0], 1e-3, None)
        u_wall_near = (p0[:, 0] - (cx - BOX_IN[0] / 2)) / np.clip(e[:, 0], 1e-3, None)
        fits &= (q_r0 + 0.012 + 0.003 < u_wall_far) & (q_l0 + d.wall_t_m + 0.003 < u_wall_near)
    rec["fits_env"] = fits
    palm = p0.copy()
    # 指先の高さ: 取る本の底と、指（と引っ込めた爪の後ろ）が下りる範囲での下の本の上面の、高い方の TIP_CLEAR 上。
    # 取る本の底だけで決めると、背の側にはみ出した下の本（傾いて高い）に指が乗り、下の本を 4〜7 mm 動かした
    fp = finger_footprint_points(spec, d)
    under_f = np.max(np.stack([top_z_under(b, spec.size_m, book, fp) for b in cell.books[:-1]]), axis=0) if len(cell.books) > 1 else z_s
    rec["finger_floor_above_bottom_mm"] = (under_f - z_s) * 1000
    palm[:, 2] = np.maximum(z_s, under_f) + TIP_CLEAR + d.fingertip_z
    r.set_forces(finger_l_joint=f_sq, finger_r_joint=SPINE_ANCHOR_N)
    r.target[:, FL], r.target[:, FR] = q_l0, q_r0
    r.target[:, LI] = 0.0
    r.target[:, CL] = 0.0
    r.target[:, PY] = 0.0
    r.target[:, PZ] = 0.0
    r.target[:, r.i("plunger_joint")] = 0.0
    above = palm.copy()
    above[:, 2] += 0.12
    yield from g_cart(r, above, quat, 150)
    yield from g_cart(r, palm, quat, 100, n_way=6)
    cor: dict = {}
    yield from g_correct(r, palm, quat, cor, iters=4)
    rec["pos_err_mm"] = cor["err"] * 1000
    aim = cor["aim"]
    snap(cell, "1 lowered beside")
    lower_disp(cell, rec, "0_lowered")

    # 3. 挟む
    g_now = r.gripper.get_pos().cpu().numpy()
    pb = book.get_pos().cpu().numpy()
    s_book = np.einsum("ni,ni->n", pb - g_now, -e)  # 本の中心の gripper y
    yield from g_joints(r, FR, np.clip(w / 2 - s_book - SPINE_PRESS, 0, fo), 40)  # 背の面まで寄せる（本は押さない）
    yield from g_joints(r, FL, np.clip(s_book + w / 2 - SQUEEZE, 0, fo), 50)  # 小口の側の壁で挟む
    yield 30
    cf = r.e.get_dofs_control_force().cpu().numpy()
    rec["squeeze_force_l_n"], rec["squeeze_force_r_n"] = cf[:, FL], cf[:, FR]
    rec["squeeze_q_mm"] = r.e.get_qpos().cpu().numpy()[:, [FL, FR]] * 1000
    rec["squeeze_target_mm"] = r.target[:, [FL, FR]] * 1000
    snap(cell, "2 squeezed")
    lower_disp(cell, rec, "1_squeezed")
    if len(cell.books) > 1:
        rec["lower_contacts_squeeze_env0"] = book_contacts_env0(cell, cell.books[-2], *[x[0] for x in book_frame(cell.books[-2])])

    # 4. 浮かせる（爪は面の上に残す）
    pb0, mb0 = book_frame(book)
    # 浮かせる高さは、爪が通る範囲のいちばん狭い所で LIFT_GAP（中心で測ると、山の上で 1.1° 傾いた本では
    # 爪の側のすき間が 4 mm 前後しかなく、爪が下の本をこすった）。中心の浮きは MAX_LIFT まで
    reach = w / 2 + CLAW_PAST_CENTER if cell.claw_mode == "center" else CLAW_SHORT
    path = claw_path_points(spec, d, reach)
    path_floor = np.concatenate([path, claw_rear_points(spec, d)])  # 爪の後ろ（指の外へはみ出す部分）も下の本に当てない
    floor_fn = ((lambda: np.max(np.stack([top_z_under(b, spec.size_m, book, path_floor) for b in cell.books[:-1]]), axis=0))
                if situation == "stack" else (lambda: np.full(n, TABLE_TOP)))
    yield from g_raise_keep_claws(r, aim, quat, LIFT_GAP, LIFT_GAP_STEPS, book, z_s, t, rec,
                                  gap_fn=lambda: bottom_z_over(book, spec.size_m, path) - floor_fn(), max_rise=MAX_LIFT)
    rec["path_gap_mm"] = (bottom_z_over(book, spec.size_m, path) - floor_fn()) * 1000
    pb1, mb1 = book_frame(book)
    rec["contacts_after_raise_env0"] = book_contacts_env0(cell, book, pb1[0], mb1[0])
    rec["lift_gap_mm"] = ((pb1[:, 2] - t / 2) - z_s) * 1000
    rec["lift_tilt_deg"] = np.degrees(np.arccos(np.clip(np.abs(mb1[:, 2, 2]), 0, 1)))
    snap(cell, "3 lifted by rubber")
    lower_disp(cell, rec, "2_raised")

    # 5. 爪の高さを合わせてから、爪を出す。爪が通る範囲で、爪の上面を本の底より CLAW_UNDER_CLEAR 下げる。ただし爪の下面は
    #    下の面（下の本の上面・台）の CLAW_FLOOR_CLEAR 上より下げない（シミュの爪は台とは当たらないが、下の本とは当たる）。
    #    本の最も低い角で合わせると、山の上で 1.1° 傾いた本では下げすぎて、爪が下の本をこすった
    low = bottom_z_over(book, spec.size_m, path)
    floor = floor_fn()
    down = claw_top_z(r, d) - (low - CLAW_UNDER_CLEAR)
    floor_room = (claw_top_z(r, d) - d.claw_t_m) - (floor + CLAW_FLOOR_CLEAR)
    down = np.minimum(down, floor_room)
    rec["claw_room_mm"] = (low - floor - d.claw_t_m) * 1000  # 爪の厚みを除いた、本の底と下の面のすき間
    rec["claw_clear_before_extend_mm"] = (low - claw_top_z(r, d)) * 1000
    yield from g_joints(r, LI, r.target[:, LI] + down, 20)
    yield 10
    rec["claw_clear_at_extend_mm"] = (bottom_z_over(book, spec.size_m, path) - claw_top_z(r, d)) * 1000
    rec["claw_floor_clear_mm"] = (claw_top_z(r, d) - d.claw_t_m - floor) * 1000
    ext = reach + CLAW_RECESS + SQUEEZE
    ext = min(ext, d.claw_stroke_m)
    rec["claw_ext_mm"] = ext * 1000
    yield from g_joints(r, CL, np.full(n, ext / 2), CLAW_STEPS // 2)
    rec["extend_contacts_env0"] = book_contacts_env0(cell, book, *[a[0] for a in book_frame(book)])
    yield from g_joints(r, CL, np.full(n, ext), CLAW_STEPS // 2)
    yield 10
    pb2, mb2 = book_frame(book)
    rec["claw_shift_mm"] = np.abs(np.einsum("ni,ni->n", (pb2 - pb1)[:, :2], e[:, :2])) * 1000
    rec["claw_shift_signed_mm"] = np.einsum("ni,ni->n", (pb2 - pb1)[:, :2], e[:, :2]) * 1000  # + は背の向き
    under = np.ones(n, bool)
    depth = []
    for p in claw_points(cell):
        loc = to_book(p, pb2, mb2)
        under &= (np.abs(loc[:, 0]) < w / 2 - 0.002) & (loc[:, 2] < -t / 2 + 0.0005) & (np.abs(loc[:, 1]) < h / 2)
        depth.append((w / 2 - loc[:, 0]) * 1000)  # 背（+x）から爪の先の点までの深さ
    rec["inserted"] = under
    rec["claw_depth_mm"] = np.min(np.stack(depth), axis=0)
    snap(cell, "4 claws under")
    lower_disp(cell, rec, "3_claws")

    # 6. 爪の台を上げて、爪の上面を本の底に当てる
    #    先にゴムの挟む力を緩める（強いままだと、摩擦で本が持たれて爪に重さがかからない）
    f_hold = float(max(2.0, SIDE_HOLD_RATIO * w_est))
    rec["f_hold_n"] = f_hold
    r.set_forces(finger_l_joint=f_hold, finger_r_joint=SPINE_ANCHOR_N)
    up_by = np.clip(bottom_z_over(book, spec.size_m, path) - claw_top_z(r, d) + CLAW_PRELOAD, 0.0, 0.012)
    yield from g_joints(r, LI, r.target[:, LI] - up_by, 40)
    yield 15
    rec["claw_gap_after_raise_mm"] = (bottom_z_over(book, spec.size_m, path) - claw_top_z(r, d)) * 1000
    rec["contacts_after_preload_env0"] = book_contacts_env0(cell, book, *[a[0] for a in book_frame(book)])

    # 7. 板を下ろして挟む
    q_r_now = r.q("finger_r_joint")
    g_now = r.gripper.get_pos().cpu().numpy()
    pb, _ = book_frame(book)
    s_book = np.einsum("ni,ni->n", pb - g_now, -e)
    y_plate = -q_r_now + 0.002 + d.plate_len_m / 2 if cell.plate_mode == "i" else s_book
    rec["plate_y_mm"] = y_plate * 1000
    yield from g_joints(r, PY, y_plate, 30)
    top_gz = np.einsum("ni,ni->n", pb - g_now, r.approach_axis()) - t / 2
    r.set_forces(plate_z_joint=f_plate, finger_l_joint=f_hold, finger_r_joint=SPINE_ANCHOR_N)
    yield from g_joints(r, PZ, np.clip(top_gz - (PLATE_BASE_Z + d.plate_t_m) + PLATE_PRESS, 0, d.plate_stroke_m), 60)
    yield 30
    snap(cell, "5 plate pressed")
    rec["hand_contacts_after_plate_env0"] = hand_env_contacts_env0(cell)
    rec["contacts_after_plate_env0"] = book_contacts_env0(cell, book, *[a[0] for a in book_frame(book)])
    lower_disp(cell, rec, "4_plate")

    # 8. 持ち上げて運ぶ
    g_now = r.gripper.get_pos().cpu().numpy()
    pb = book.get_pos().cpu().numpy()
    up = g_now.copy()
    up[:, 2] = CARRY_BOTTOM_Z + (g_now[:, 2] - (pb[:, 2] - t / 2))
    yield from g_cart(r, up, quat, LIFT_STEPS, n_way=10, profile=min_jerk)
    yield 20
    pb = book.get_pos().cpu().numpy()
    lifted = (pb[:, 2] - p0[:, 2] > 0.05) & touching_robot(cell)
    rec["lifted"] = lifted
    g_now = r.gripper.get_pos().cpu().numpy()
    off = pb - g_now
    snap(cell, "6 lifted")
    rec["contacts_after_lift_env0"] = book_contacts_env0(cell, book, *[a[0] for a in book_frame(book)])
    lower_disp(cell, rec, "5_lifted")
    dest = cell.at(*PLACE_C)
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
    snap(cell, "7 over place table")

    # 9. 下ろして、板で押さえたまま爪を抜く
    lift_q = r.q("finger_r_lift")
    place = g_now.copy()
    place[:, 2] = TABLE_TOP + PLACE_CLEAR + d.fingertip_z + lift_q
    yield from g_cart(r, place, quat, LOWER_STEPS, n_way=8, profile=min_jerk)
    yield 30
    snap(cell, "8 lowered")
    pw0, _ = book_frame(book)
    # 背の側のゴムで本を受け止める（爪につられて本が背の側へ動かないように）。板は軽く当てるだけ
    r.set_forces(plate_z_joint=WITHDRAW_PLATE_N, finger_l_joint=f_sq, finger_r_joint=SPINE_ANCHOR_N)
    yield 20
    # 爪を本の底から離してから抜く。爪の上面は押し上げ（CLAW_PRELOAD）で本の底に 1.3 mm めり込んでいて、背の側の底の角に
    # 引っかかって爪が 1 mm も戻らなかった（試行8 本試験 20261019T043223・ざらざらの B5。爪の台を下げるだけでは、
    # 台に当たらない範囲が 0.3 mm しかなかった）。そこでゴムで本を持ったまま手を少し上げ、同じだけ爪の台を下げる。
    # 爪の上面が本の底より WITHDRAW_GAP 下になるまで繰り返す（腕のたわみが変わるので測りながら）。
    # 置いたときには、腕のたわみで爪の下面が台の上面より 0.3 mm 下にあった（実物なら爪が台に載り、本は爪の上）
    rise = np.zeros(n)
    hist = []
    for _ in range(WITHDRAW_ITERS):
        gap = bottom_z_over(book, spec.size_m, path) - claw_top_z(r, d)
        need = np.clip(WITHDRAW_GAP - gap, 0.0, 0.002)
        if not (need > 0.0001).any():
            break
        rise += need
        g_up = place.copy()  # 指令の位置から上げる（実際の位置は腕のたわみで 1〜2 mm 低く、そこから上げると逆に下がった）
        g_up[:, 2] += rise
        yield from g_cart(r, g_up, quat, 15, n_way=2)
        # 爪の台を下げる。ただし爪の下面は台の上面より下げない（シミュの爪は台をすり抜けるが、実物は台に当たる）
        room = claw_top_z(r, d) - d.claw_t_m - TABLE_TOP
        yield from g_joints(r, LI, r.q("finger_r_lift") + np.clip(room - 0.0002, 0.0, need + 0.001), 15)
        yield 5
        hist.append([float(x) for x in (gap[0] * 1000, r.q("finger_r_lift")[0] * 1000, claw_top_z(r, d)[0] * 1000,
                                        bottom_z_over(book, spec.size_m, path)[0] * 1000, r.gripper.get_pos().cpu().numpy()[0, 2] * 1000,
                                        r.target[0, LI] * 1000)])
    rec["withdraw_hist_env0"] = hist
    rec["withdraw_hand_rise_mm"] = rise * 1000
    rec["withdraw_claw_gap_mm"] = (bottom_z_over(book, spec.size_m, path) - claw_top_z(r, d)) * 1000
    yield from g_joints(r, CL, np.zeros(n), WITHDRAW_STEPS)
    yield 20
    rec["claw_q_after_withdraw_mm"] = r.q("claw_joint") * 1000
    pw1, _ = book_frame(book)
    rec["withdraw_shift_mm"] = np.linalg.norm((pw1 - pw0)[:, :2], axis=1) * 1000
    rec["withdraw_shift_signed_mm"] = np.einsum("ni,ni->n", (pw1 - pw0)[:, :2], e[:, :2]) * 1000  # + は背の向き
    rec["withdraw_contacts_env0"] = book_contacts_env0(cell, book, *[a[0] for a in book_frame(book)])
    pb, mb = book_frame(book)
    still = np.zeros(n, bool)
    for p in claw_points(cell):
        loc = to_book(p, pb, mb)
        still |= (np.abs(loc[:, 0]) < w / 2) & (loc[:, 2] < 0)
    rec["claw_left_under"] = still
    # 爪を抜いた後、本は台から（手を上げた分＋爪の厚み）浮いている。ゴムで持ったまま台に下ろしてから放す
    # （下ろさずに放すと B5 が 5 mm 落ちた）。指の先は本の底より上にあるので台には当たらない。
    #    腕のたわみで 1 回では 2.6 mm 残ったので、測りながら 3 回まで（手を上げる側と同じ）
    g_dn = place.copy()
    g_dn[:, 2] += rise
    for _ in range(3):
        gap = book_lowest_z(book, spec.size_m) - TABLE_TOP
        if not (gap > 0.0003).any():
            break
        g_dn[:, 2] -= np.clip(gap + SET_DOWN_PRESS, 0.0, None)
        yield from g_cart(r, g_dn, quat, 30, n_way=3)
        yield 15
    rec["set_down_gap_mm"] = (book_lowest_z(book, spec.size_m) - TABLE_TOP) * 1000
    yield from g_joints(r, PZ, np.zeros(n), 30)
    yield from g_joints(r, [FL, FR], np.full((n, 2), fo), 30)
    yield 10
    z_rest = book.get_pos().cpu().numpy()[:, 2]
    rise = np.zeros(n)
    leave = r.gripper.get_pos().cpu().numpy()
    for k in range(1, 9):
        lk = leave.copy()
        lk[:, 2] += 0.12 * k / 8
        yield from g_cart(r, lk, quat, 25 if k <= 2 else 10, n_way=1)
        rise = np.maximum(rise, book.get_pos().cpu().numpy()[:, 2] - z_rest)
    rec["release_rise_mm"] = rise * 1000
    rec["cycle_s"] = r.world.sim_time - t_start
    yield 80
    snap(cell, "9 released")


class Probe:
    """毎歩の後に呼ぶ。本の接触・姿勢を見て、ありえるかを判定し、運ぶ間の下からの支えと、小口の側の底面への接触を測る。"""

    HAND = ("gripper", "finger_l", "finger_r", "finger_r_tip", "claw", "plunger", "plate_carriage", "plate")

    def __init__(self, scene, world: World, cells: list[Cell], n: int):
        self.scene, self.world, self.cells, self.n = scene, world, cells, n
        self.active = False
        self.k = 0
        solver = scene.rigid_solver
        n_links = solver.n_links
        self.cat = np.zeros(n_links + 1, int)  # 0 なし / 1 ロボット / 2 壁 / 3 台・床・本
        self.hand = np.zeros(n_links + 1, bool)
        self.book_links, self.book_cell, self.hand_links = [], [], []
        for ci, c in enumerate(cells):
            e = c.robot_ent
            self.cat[e.link_start:e.link_end] = 1
            for name in self.HAND:
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
        self.cat[n_links] = 0
        self.book_links = np.array(self.book_links)
        self.book_cell = np.array(self.book_cell)
        self.n_links = n_links
        self.mons = [Monitor(n, len(c.books), Rules()) for c in cells]
        self.claws = [(ci, idx, loc) for ci, c in enumerate(cells) for idx, loc in claw_links(c)]
        self.claw_idx = np.array([idx for _, idx, _ in self.claws], int)
        self.support = [[] for _ in cells]  # 正味の支え（判定に使う）
        self.support_raw = [[] for _ in cells]  # 爪からの上向きの力だけ（参考）
        self.claw_cx = [[] for _ in cells]
        self.claw_link = [c.robot_ent.get_link("claw").idx for c in cells]
        self.plate_link = [c.robot_ent.get_link("plate").idx for c in cells]
        self.fore_touch = [np.zeros(n, bool) for _ in cells]  # 小口の側の底面（小口から幅の 1/3）に手が触れた
        self.mass = [c.rec["mass"] for c in cells]

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
        fvec = cd["force"].cpu().numpy().reshape(la.shape + (3,))
        cpos = cd["position"].cpu().numpy().reshape(la.shape + (3,))
        force = np.linalg.norm(fvec, axis=-1)
        valid = (la >= 0) & (lb >= 0)
        la = np.where(valid, la, self.n_links)
        lb = np.where(valid, lb, self.n_links)
        ca, cb = self.cat[la], self.cat[lb]
        ha, hb = self.hand[la], self.hand[lb]
        bl = self.book_links
        is_a = la[:, :, None] == bl[None, None, :]
        is_b = lb[:, :, None] == bl[None, None, :]
        other_cat = np.where(is_a, cb[:, :, None], 0) + np.where(is_b, ca[:, :, None], 0)
        other_hand = (is_a & hb[:, :, None]) | (is_b & ha[:, :, None])
        touch_robot = (other_cat == 1).any(axis=1)
        touch_wall = (other_cat == 2).any(axis=1)
        touch_sup = (other_cat == 3).any(axis=1)
        pen_hand = np.where(other_hand, pen[:, :, None], 0.0).max(axis=1, initial=0.0)
        other_link = np.where(is_a, lb[:, :, None], la[:, :, None])
        f_link = np.zeros(other_hand.shape[::2], float)
        for hl in self.hand_links:
            f_link = np.maximum(f_link, np.where(other_hand & (other_link == hl), force[:, :, None], 0.0).sum(axis=1))
        pos = solver.get_links_pos(links_idx=bl).cpu().numpy().reshape(self.n, len(bl), 3)
        quat = solver.get_links_quat(links_idx=bl).cpu().numpy().reshape(self.n, len(bl), 4)
        up_z = quat_to_mat(quat)[:, :, 2, 2]
        lp = solver.get_links_pos(links_idx=self.claw_idx).cpu().numpy().reshape(self.n, len(self.claw_idx), 3)
        lm = quat_to_mat(solver.get_links_quat(links_idx=self.claw_idx).cpu().numpy().reshape(self.n, len(self.claw_idx), 4))
        t = self.world.sim_time
        dt = CHECK_EVERY * DT
        for ci, c in enumerate(self.cells):
            sel = np.nonzero(self.book_cell == ci)[0]
            j = len(sel) - 1
            bm = quat_to_mat(quat[:, sel[j]])
            hw, hh, ht = (v / 2 for v in c.rec["size"])
            inside = np.zeros((self.n, len(sel)), bool)
            for k, (cj, _, loc) in enumerate(self.claws):
                if cj != ci:
                    continue
                p = lp[:, k] + np.einsum("nij,j->ni", lm[:, k], loc)
                lc = to_book(p, pos[:, sel[j]], bm)
                inside[:, j] |= ((np.abs(lc[:, 0]) < hw - 0.002) & (np.abs(lc[:, 1]) < hh - 0.002)
                                 & (np.abs(lc[:, 2]) < ht - 0.0015))
            # 手と取る本の接触の、本の座標での位置
            bl_i = bl[sel[j]]
            on_b = (lb == bl_i) & ha
            on_a = (la == bl_i) & hb
            loc_c = np.einsum("nci,nij->ncj", cpos - pos[:, None, sel[j]], bm)  # 本の座標（列ベクトル m の転置で回す）
            bottom = (loc_c[..., 2] < -ht + 0.003) & (np.abs(loc_c[..., 0]) < hw - 0.003)  # 底面（側面を除く）
            fore = bottom & (loc_c[..., 0] < -hw + 2 * hw / 3)  # 小口（-x）から幅の 1/3
            self.fore_touch[ci] |= ((on_a | on_b) & fore).any(axis=1)
            if c.measure:
                fz = np.where(on_b, fvec[..., 2], 0.0) - np.where(on_a, fvec[..., 2], 0.0)
                # 正味の支え = (爪からの上向きの力 + 板からの下向きの力) / 重さ = 1 − 横（ゴム・壁）の摩擦が受けた割合。
                # 爪の力だけだと、板が上から押す力まで「下からの支え」に数えてしまう（爪 15 mm で 2.1 倍になった）。
                # 面の位置で分けると、本の縁ちょうどの接触が漏れたので、部品（リンク）で分ける
                other = np.where(on_b, la, lb)
                wgt = self.mass[ci] * GRAVITY
                is_claw = (on_a | on_b) & (other == self.claw_link[ci])
                is_plate = (on_a | on_b) & (other == self.plate_link[ci])
                self.support_raw[ci].append(np.where(is_claw, fz, 0.0).sum(axis=1) / wgt)
                self.support[ci].append(np.where(is_claw | is_plate, fz, 0.0).sum(axis=1) / wgt)
                # 爪の力の中心（本の座標 x。重心 = 0、背 = +w/2）。重心から遠いほど、実物の本は支えの外側が垂れる
                fc = np.where(is_claw, np.clip(fz, 0.0, None), 0.0)
                self.claw_cx[ci].append(np.where(fc.sum(axis=1) > 1e-6, (fc * loc_c[..., 0]).sum(axis=1) / np.maximum(fc.sum(axis=1), 1e-9), np.nan))
            self.mons[ci].update(t, dt, up_z[:, sel], touch_robot[:, sel], touch_wall[:, sel], touch_sup[:, sel],
                                 penetration_m=pen_hand[:, sel], inside=inside, hand_force_n=f_link[:, sel])


def main() -> None:
    args = parse_args()
    run_id, out, meta = new_run(TRIAL, dict(vars(args)), extra={"genesis": gs.__version__})
    result: dict = {}
    error = None
    ctx: dict = {"recording": []}
    try:
        result = run(args, out, ctx)
    except BaseException as e:
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
    render = set(s for s in args.render.split(",") if s)
    rng = np.random.default_rng(args.seed)
    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=4),
                     rigid_options=gs.options.RigidOptions(enable_self_collision=False, noslip_iterations=args.noslip,
                                                           friction_cone=getattr(gs.friction_cone, args.cone),
                                                           impratio=args.impratio,
                                                           constraint_timeconst=args.timeconst or None),
                     vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane(contype=ENV_BIT, conaffinity=ENV_BIT))
    cells: list[Cell] = []
    for key in [s for s in args.cells.split(",") if s]:
        dn, pm, cm = key.split(":")
        if pm not in PLATE_MODES or cm not in CLAW_MODES:
            raise ValueError(f"セルの指定が不正: {key}（板 {PLATE_MODES}・爪 {CLAW_MODES}）")
        c = Cell(key, DESIGNS[dn], pm, cm, len(cells) * CELL_DX)
        build_cell(scene, c, spec, args.situation, out, key in render, args.book_prim)
        cells.append(c)
    t_build = time.time()
    scene.build(n_envs=n)
    build_s = time.time() - t_build
    world = World(scene, n, DT)
    for c in cells:
        c.robot = PinchRobot(world, c.robot_ent, c.key, c.design, base_xy=(c.x0, 0.0))
        c.robot.setup()
        c.rec["size"] = spec.size_m
        c.rec["mass"] = spec.weight_kg
    n_books = len(cells[0].books)
    jit_xy = np.zeros((n, n_books, 2))
    jc, jd = args.jitter_cm / 100, math.radians(args.jitter_deg)
    jit_xy[..., 0] = rng.uniform(-jc, jc, (n, n_books))
    jit_xy[..., 1] = rng.uniform(-jc, jc, (n, n_books))
    jit_yaw = rng.uniform(-jd, jd, (n, n_books))
    for c in cells:
        place_books(c, spec, args.situation, jit_xy, jit_yaw)
    world.step(100)
    probe = Probe(scene, world, cells, n)
    world.monitors.append(probe)
    for c in cells:
        for name, cam in c.cams.items():
            cam.start_recording(save_to_filename=str(out / f"{name}_{c.key.replace(':', '_')}.mp4"), fps=25)
            ctx["recording"].append(cam)

    def snap(cell: Cell, label: str) -> None:
        cell.rec.setdefault("phase_t", {})[label] = world.sim_time
        if "close" in cell.cams:
            cam = cell.cams["place"] if label[0] in "789" else cell.cams["close"]
            rgb, *_ = cam.render()
            cell.frames.append((label, np.asarray(rgb)))

    p_lower0 = [[b.get_pos().cpu().numpy() for b in c.books[:-1]] for c in cells]
    probe.active = True
    t_sim0 = world.sim_time
    t_wall0 = time.time()
    run_generators(world, [pinch_cycle(c, spec, n, snap, args.situation) for c in cells])
    wall_s = time.time() - t_wall0
    probe.active = False

    rows, cell_res = [], {}
    lx, ly = PLACE_TABLE
    w, h, t = spec.size_m
    for ci, c in enumerate(cells):
        mon, rec = probe.mons[ci], c.rec
        pe, me = book_frame(c.target)
        up_end = me[:, 2, 2]
        corners = np.array([[sx * w / 2, sy * h / 2] for sx in (-1, 1) for sy in (-1, 1)])
        cw = pe[:, None, :2] + np.einsum("nij,kj->nki", me[:, :2, :2], corners)
        tc = c.at(*PLACE_C)[:2]
        on_table = ((np.abs(cw[:, :, 0] - tc[0]) <= lx / 2) & (np.abs(cw[:, :, 1] - tc[1]) <= ly / 2)).all(axis=1)
        flat = (np.abs(up_end) > 0.97) & (np.sign(up_end) == np.sign(rec["up0"]))
        on_floor = np.abs(pe[:, 2] - (TABLE_TOP + t / 2)) < 0.004
        released = ~touching_robot(c)
        placed = on_table & flat & on_floor & released
        lower_ok = np.ones(n, bool)
        for b, p_start in zip(c.books[:-1], p_lower0[ci]):
            pb, mb = book_frame(b)
            lower_ok &= (np.abs(mb[:, 2, 2]) > 0.97) & (np.linalg.norm(pb - p_start, axis=1) < 0.03)
        sup = np.array(probe.support[ci]) if probe.support[ci] else np.zeros((1, n))
        sup_med = np.median(sup, axis=0)
        sup_raw = np.array(probe.support_raw[ci]) if probe.support_raw[ci] else np.zeros((1, n))
        rec["support_claw_only_median"] = np.median(sup_raw, axis=0)
        rec["support_series_env0"] = [round(float(v), 2) for v in sup[::5, 0]]  # 運ぶ間の正味の支え（環境 0、10 歩ごと）
        rec["claw_only_series_env0"] = [round(float(v), 2) for v in sup_raw[::5, 0]]
        cx = np.array(probe.claw_cx[ci]) if probe.claw_cx[ci] else np.full((1, n), np.nan)
        rec["claw_support_from_com_mm"] = np.nanmedian(cx, axis=0) * 1000  # + は背の側
        from_below = rec["inserted"] & (sup_med >= SUPPORT_MIN_RATIO)
        lift_tilt_ok = rec["lift_tilt_deg"] <= LIFT_TILT_MAX_DEG
        claw_shift_ok = rec["claw_shift_mm"] <= CLAW_SHIFT_MAX_MM
        no_fore = ~probe.fore_touch[ci]
        withdraw_ok = (rec["withdraw_shift_mm"] <= WITHDRAW_SHIFT_MAX_MM) & ~rec["claw_left_under"]
        clean_release = rec["release_rise_mm"] <= RELEASE_RISE_MAX_MM
        checks = {  # 判定の順（失敗の分類も、この順で最初に落ちたもの）
            "no_fit": rec["fits_env"], "lift_tilt>5deg": lift_tilt_ok, "not_inserted": rec["inserted"],
            "claw_shift>3mm": claw_shift_ok, "lift_failed": rec["lifted"], "dropped_in_carry": rec["carried"],
            "held_by_pinch(support<0.5)": from_below, "fore_edge_touched": no_fore, "withdraw_shift>5mm": withdraw_ok,
            "place_off_table": on_table, "place_not_flat": flat & on_floor, "not_released": released,
            "dragged_on_release": clean_release, "stack_disturbed": lower_ok,
        }
        success_raw = np.ones(n, bool)
        for v in checks.values():
            success_raw &= v
        success = success_raw & mon.ok
        modes = []
        for i in range(n):
            if success[i]:
                modes.append("ok")
            elif success_raw[i]:
                modes.append("implausible:" + "+".join(k for k in KINDS if mon.violated[k][i]))
            else:
                modes.append(next(k for k, v in checks.items() if not v[i]))
        rec.update(placed=placed, support_ratio_median=sup_med, from_below=from_below, fore_touch=probe.fore_touch[ci],
                   lower_ok=lower_ok, end_pos=pe, end_up_z=up_end)
        cell_res[c.key] = {
            "design": c.design.name, "rubber_mu": c.design.rubber_mu, "plate_mode": c.plate_mode, "claw_mode": c.claw_mode,
            "success_n": int(success.sum()), "success_raw_n": int(success_raw.sum()),
            "implausible_n": int((success_raw & ~mon.ok).sum()),
            "violations": {k: int(mon.violated[k].sum()) for k in KINDS},
            "failure_modes": {m: modes.count(m) for m in sorted(set(modes))},
            "checks_pass_n": {k: int(v.sum()) for k, v in checks.items()},
            "max_penetration_mm": float(mon.max_penetration_m.max() * 1000),
            "peak_hand_force_n": {"median": float(np.median(mon.peak_hand_force_n)), "max": float(mon.peak_hand_force_n.max())},
            "cycle_s": float(rec.get("cycle_s", float("nan"))),
            "forces_n": {k: rec.get(k) for k in ("f_squeeze_n", "f_side_n", "f_plate_n")},
            "per_env": {"mode": modes, **{k: v for k, v in rec.items() if k not in ("size", "p0", "_lower0")}, **mon.per_env()},
        }
        rows.append({
            "run_book": spec.book_id, "surface": spec.surface, "situation": args.situation, "cell": c.key,
            "design": c.design.name, "mu": c.design.rubber_mu, "plate": c.plate_mode, "claw": c.claw_mode, "n": n,
            "success": int(success.sum()), "success_raw": int(success_raw.sum()),
            "implausible": int((success_raw & ~mon.ok).sum()),
            **{f"viol_{k}": int(mon.violated[k].sum()) for k in KINDS},
            "lift_gap_mm_med": round(float(np.median(rec["lift_gap_mm"])), 2),
            "lift_tilt_deg_max": round(float(rec["lift_tilt_deg"].max()), 2),
            "claw_shift_mm_max": round(float(rec["claw_shift_mm"].max()), 2),
            "support_med": round(float(np.median(sup_med)), 2),
            "withdraw_shift_mm_max": round(float(rec["withdraw_shift_mm"].max()), 2),
            "release_rise_mm_max": round(float(rec["release_rise_mm"].max()), 2),
            "max_pen_mm": round(float(mon.max_penetration_m.max() * 1000), 2),
            "lower_disp_mm_max": round(float(rec.get("lower_disp_mm_5_lifted", np.zeros(1)).max()), 2),
            "claw_support_from_com_mm_med": round(float(np.nanmedian(rec["claw_support_from_com_mm"])), 1),
            "cycle_s": round(float(rec.get("cycle_s", float("nan"))), 2),
            "modes": ";".join(f"{m}={modes.count(m)}" for m in sorted(set(modes))),
        })
        if c.frames:
            montage(c.frames, out / f"montage_{c.key.replace(':', '_')}.png")
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return {
        "criterion": "8_pinch",
        "acceptance": "A5 の判定（爪が入る・下から支える・傾き・ずれ・小口に触れない・置ける・抜くときのずれ）とありえる状態",
        "acceptance_ok": any(v["success_n"] >= math.ceil(0.8 * n) for v in cell_res.values()),
        "n_envs": n,
        "book": {"book_id": spec.book_id, "format": spec.format_name, "size_mm": [round(v * 1000, 1) for v in spec.size_m],
                 "weight_g": round(spec.weight_kg * 1000, 1), "weight_est_g": round(spec.weight_estimated_kg * 1000, 1),
                 "surface": spec.surface, "book_friction": SURFACES[spec.surface]["friction"]},
        "situation": args.situation,
        "physics": {"noslip_iterations": args.noslip, "friction_cone": args.cone, "impratio": args.impratio, "timeconst": args.timeconst, "table_friction": TABLE_FRICTION, "dt": DT, "substeps": 4},
        "plausibility_rules": Rules().describe(),
        "designs": {c.design.name: c.design.params() for c in cells},
        "cells": cell_res,
        "wall_build_s": build_s, "wall_run_s": wall_s, "sim_s": world.sim_time - t_sim0,
    }


if __name__ == "__main__":
    main()
