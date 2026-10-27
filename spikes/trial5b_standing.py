"""試行5b: 箱の中に立っている本（隣の本・箱の壁に寄りかかっている）を、試行5 の手（B15・R25）で取り出し、
検品トレーに平らに置けるか。

前提（2026-10-12〜13 のユーザー判断。試行5 と同じ）
  - 本を吸引で持ち上げない。吸引は補助だけ（押さえる・端を数 mm めくる）
  - 本は (1) 面の上に平ら (2) 手に持たれている (3) 両側（隣の本・箱の壁）に支えられて立っている、のどれか。
    支えなしで辺・角に載っている瞬間があれば失敗（plausibility.py。edge_rest は一瞬でも違反）
  - 成功は「トレーに平ら・運ぶ間の下からの支えが重さの 0.5 倍以上・ありえない状態が 0 件」
  - 摩擦・接触の値は変えない（試行5 と同じ）。力の値も試行5 と同じ（新しく要る値は下の LAY_* に分けて書く）

箱の中の並び（このファイルで作るシーン。本はすべて同じ本）
  - 箱の内寸 0.56 m（x）× 0.42 m（y）・壁の高さ 0.16 m（layout.py と同じ高さ。x は B5 の山と寝かせる場所が入るよう広げた）
  - ロボットは箱の -y 側から +y を向く（本を倒す向き -x がロボットから見て横になり、倒す間の手の届く距離が変わらない。
    ロボットの正面へ倒す配置では、奥の壁の近くと台座の近くの両方で IK が収束しなかった）。検品トレーは箱の -x の外
  - 奥の壁（+x）に 3 冊を立てかける（scene.py の standing と同じ作り: 傾き 12° ± 3°・本の間 2 mm・
    表紙は -x）。取るのは一番手前の 1 冊（後ろに隣の本が 2 冊）
  - pile:  手前の本の下の手前の辺から d だけ手前に、平積みの山（同じ本を k 冊、表紙を下）。山の高さ ≧ 0.15 × 本の高さ、
           寝かせる本が山の奥の縁に当たる位置が本の下の端から 0.30 × 本の高さ（重心が縁の手前 → 山の上へ倒れる）
  - floor: 手前は何も無い床（未検証）

寝かせる手順（方法 L。本を手から離さずに、下の辺を軸に手前へ倒す）
  1. 指の開閉軸を x にし、右の指の爪（下面が平らなステンレス）を本の天の面に載せ、右の指の上下の関節の力の上限
     （LAY_PRESS = 推定の重さ × 1.5、3〜15 N）で押し付ける。手は本の軸に沿わせる（鉛直から -25°〜+60°）
  2. 本の姿勢（シミュレーションの真値）を毎回測り、いま床（または山の縁）に接している辺を軸に 2° 回した位置へ手を動かす
     （0.15 s ごと）。爪の摩擦で本の上の端を手前へ引き、倒れていく本を支える。吸盤は使わない
  3. pile: 本は山の奥の縁に載り、手で支えながら山の上へ平らに倒れる（シーソー）
  4. 爪を上げて手を離す（本は平ら）。そのあと試行5 の「山の一番上」と同じちりとり方式で取る（pile_pick_cycle）。
     floor は台の縁が無いので floor_pick_cycle（吸引ありで端を数 mm めくる）を書いたが、まだ動かしていない

判定（環境ごと）: 寝かせられた（平ら・山/床の上）→ 試行5 と同じ判定（爪が入った・下からの支え ≧ 0.5・持ち上がった・
運ぶ間に落ちない・トレーに平ら・引きずらない）→ 山の本・残りの立った本を乱していない → ありえない状態が 0 件。
立っている本どうしの寄りかかりは「壁に立てかけた本」と同じに数える（Probe5b）

使い方: python spikes/trial5b_standing.py --book book-10 --surface rough --situation pile --designs B15,R25 --n-envs 4
出力: /workspace/outputs/trial5b_standing/<run_id>/（result.json・cells.csv・side_<cell>.mp4・montage_<cell>.png）
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
import trial5_slideunder as T5
from pai_book_intake.book_assets import make_book_assets
from pai_book_intake.books import DEFAULT_CATALOG, GRAVITY, SURFACES, grip_plan, load_catalog
from pai_book_intake.geometry import Q_DOWN, mat_to_quat, quat_mul, quat_to_mat, quat_z
from pai_book_intake.hand_design import CUP_TIP_RETRACTED_Z, FINGER_PLATE_T, PLUNGER_STROKE
from pai_book_intake.plausibility import KINDS, Rules
from pai_book_intake.runlog import classify, new_run, save_result
from pai_book_intake.tool import FINGERS, FORCE, LIFT, PLUNGER, Robot, World, build_robot_xml, home_for
from trial5_slideunder import (BLADE_BELOW, CARD, CELL_DX, CHECK_EVERY, DESIGNS, DT, GRAY, LIFT_STEPS, LIP_CLEAR,
                               LOWER_STEPS, MARGIN, MARGIN_L, PEEL_PULL, PLACE_CLEAR, PUSH_STEPS, RAISE_LIFT,
                               RAISE_TILT_BOTH_DEG, SLIDE_STOP, SQUEEZE, SUPPORT_MIN_RATIO, TABLE_TOP, TIP_GAP,
                               TRAY_COLOR, TRAY_IN, TRAY_RIM_H, WALL_T, WITHDRAW_STEPS, add_box,
                               book_frame, edge_along, g_cart, g_correct, g_joints, g_wait, lip_points, lip_side_quat,
                               min_jerk, montage, run_generators, set_forces, to_book, touching_robot)

TRIAL = "trial5b_standing"
BOX_C = (0.41, -0.12)  # 箱の中心（セルの座標）。本は +x の壁に立てかけ、-x へ倒す
BOX_IN = (0.56, 0.42)
BOX_FLOOR = 0.02
BOX_WALL_H = 0.16
ROBOT_XY = (BOX_C[0], BOX_C[1] - 0.55)  # ロボットの台座（セルの座標）。箱の -y 側から +y を向く（本を倒す向き x はロボットから見て横）
ROBOT_YAW_DEG = 90.0
TRAY_C = (-0.12, -0.40)  # 検品トレーの中心（箱の -x の外、ロボットの左前。台座から約 0.6 m）
CARRY_BOTTOM_Z = BOX_WALL_H + 0.04  # 試行5 と同じ
CARRY_STEPS = 170  # 運ぶ時間 1.7 s（試行5 は 0.34 m を 1.2 s。ここは約 0.65 m なので、最小ジャーク軌道の加速度の最大を
# 試行5 の約 1.4 m/s² 以下（5.77 × 0.65 / 1.7² ≈ 1.3 m/s²）に保つ）
N_NEIGHBOURS = 2  # 取る本の後ろに立っている本の数
LEAN_DEG, LEAN_JITTER_DEG = 12.0, 3.0  # scene.PileConfig と同じ
STAND_GAP = 0.002
STAND_JITTER_Y = 0.01
PILE_H_RATIO = 0.15  # 山の高さ ≧ 本の高さ × これ（高いほど本が山の縁に載るときの傾きが大きく手を立てたまま扱えるが、
# シーソーで本の下の端が後ろへ大きく振れて、後ろの立った本を押す。0.22 では後ろの本を起こして 64° で止まった）
PILE_SEESAW_RATIO = 0.30  # 山の縁に載る位置（本の下の端から、本の高さ × これ）。0.5 未満なら重心が縁の手前で、山の上へ倒れる
PILE_D_MIN_RATIO = 0.15
PILE_JITTER_XY, PILE_JITTER_DEG = 0.005, 3.0

# 寝かせる手順の値（新しい値。ユーザーの確認が要る）
LAY_PRESS_RATIO, LAY_PRESS_MIN_N, LAY_PRESS_MAX_N = 1.5, 3.0, 15.0  # 爪を天の面に押し付ける力 = 推定の重さ × 1.5（3〜15 N）。
# 強すぎると本を軸の向きに押して山の縁の上を滑らせ、重心が縁の奥に来てシーソーしない（10 N で 4/4 が 76° で止まった）
LAY_STEP_DEG = 2.0  # 1 回に回す角度
LAY_STEPS_PER = 15  # 1 回の時間（15 歩 = 0.15 s）→ 約 13°/s
LAY_MAX_ITERS = 90
LAY_BETA_MIN_DEG, LAY_BETA_MAX_DEG = -25.0, 60.0  # 手の鉛直からの傾き（+ は奥へ）。爪の下面を天の面に沿わせる
SAFE_Z = 0.60  # 手を運ぶときの高さ（立てた本・箱の壁より十分高い）
LAY_LIFT_PLAN = 0.022  # 右の指の上下の関節がこの位置で爪の下面が天の面に触れるよう手を置く（目標は下いっぱい＝力の上限で押す）
LAY_Q_R, LAY_Q_L = 0.02, 0.025  # 寝かせる間の指の開き（右 = 爪を天に当てる指・左 = 右の爪より 22 mm 上で後ろの本の天に触れない位置。
# 手を本の傾きに合わせると、後ろへ行くほど手の面が下がる。0.07 では左の指が後ろの本の天を叩いた）
LAY_FLAT_DEG = 3.0

# 本の向き: 立てた本 = 横幅 -y・天地 +z・表紙（+z の面）が -x（ロボットの側）。寝かせた本 = 天が -x・表紙が下
R_STAND = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
R_LAID = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
Q_STAND = mat_to_quat(R_STAND)
Q_LAID = mat_to_quat(R_LAID)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="book-10")
    ap.add_argument("--surface", choices=list(SURFACES), default=None, help="表面（省略時はカタログのまま）")
    ap.add_argument("--situation", choices=["pile", "floor"], default="pile",
                    help="pile: 立った本の手前に平積みの山 / floor: 手前は何も無い床")
    ap.add_argument("--designs", default="B15,R25", help=f"カンマ区切り。候補: {','.join(DESIGNS)}")
    ap.add_argument("--suction", choices=["on", "off", "both"], default="both")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", default="", help="動画と連続写真を撮るセル（カンマ区切り。例 B15:off）")
    return ap.parse_args()


@dataclasses.dataclass
class Cell5b(T5.Cell):
    """books の並び: 立っている隣の本（奥から）→ 山（下から。pile のとき）→ 取る本（最後）。
    pile のとき books[-2] が山の一番上になり、試行5 の「下の本」と同じ位置に来る。"""
    situation: str = "pile"
    n_stand: int = 0
    n_pile: int = 0

    @property
    def neighbours(self):
        return self.books[:self.n_stand]

    @property
    def pile(self):
        return self.books[self.n_stand:self.n_stand + self.n_pile]


def quat_y(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    z = np.zeros_like(a)
    return np.stack([np.cos(a / 2), z, np.sin(a / 2), z], axis=-1)


def rot_y(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def box_corners(w, h, t) -> np.ndarray:
    return np.array([[sx * w / 2, sy * h / 2, sz * t / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def pile_geometry(spec) -> dict:
    """山の冊数 k・高さ・山の奥の縁と手前の本の下の辺の距離 d。"""
    w, h, t = spec.size_m
    k = max(1, math.ceil(PILE_H_RATIO * h / t))
    hp = k * t
    s_e = PILE_SEESAW_RATIO * h
    d = math.sqrt(max(s_e ** 2 - hp ** 2, (PILE_D_MIN_RATIO * h) ** 2))
    return {"k": k, "height_m": hp, "d_m": d, "contact_from_bottom_m": math.hypot(d, hp),
            "landing_tilt_from_horizontal_deg": math.degrees(math.atan2(hp, d))}


def standing_layout(spec, n_stand: int, lean: np.ndarray, x_wall: float, y_c: np.ndarray):
    """奥の壁（内面 x_wall）から順に、平行に立てかけた本の (中心, 向き)（環境ごと）。scene._standing_poses の x 版。"""
    w, h, t = spec.size_m
    n = lean.shape[0]
    q = quat_mul(quat_y(lean), np.broadcast_to(Q_STAND, (n, 4)))
    rm = quat_to_mat(q)
    cw = np.einsum("nij,kj->nki", rm, box_corners(w, h, t))
    z = BOX_FLOOR + 0.001 - cw[:, :, 2].min(axis=1)
    nb = -rm[:, :, 2]  # 裏表紙（+x の側）の法線
    out, prev = [], None
    for i in range(n_stand):
        if prev is None:
            x = x_wall - 0.002 - cw[:, :, 0].max(axis=1)
        else:
            x = (prev - STAND_GAP - t / 2 - nb[:, 2] * z) / nb[:, 0]
        c = np.stack([x, y_c[:, i], z], -1)
        prev = np.einsum("ni,ni->n", nb, c) - t / 2  # この本の表紙の面（nb·p の値）
        out.append((c, q))
    return out


def build_cell(scene, cell: Cell5b, spec, situation: str, out: Path, render: bool) -> None:
    bx, by = BOX_C
    ix, iy = BOX_IN
    c = cell.at(bx, by)
    cell.supports.append(add_box(scene, (ix + 2 * WALL_T, iy + 2 * WALL_T, BOX_FLOOR), c + [0, 0, BOX_FLOOR / 2], CARD))
    for size, off in [((WALL_T, iy + 2 * WALL_T, BOX_WALL_H), (-ix / 2 - WALL_T / 2, 0)),
                      ((WALL_T, iy + 2 * WALL_T, BOX_WALL_H), (ix / 2 + WALL_T / 2, 0)),
                      ((ix, WALL_T, BOX_WALL_H), (0, -iy / 2 - WALL_T / 2)),
                      ((ix, WALL_T, BOX_WALL_H), (0, iy / 2 + WALL_T / 2))]:
        cell.walls.append(add_box(scene, size, c + [off[0], off[1], BOX_WALL_H / 2], CARD))
    tx, ty = TRAY_C
    lx, ly = TRAY_IN
    cell.supports.append(add_box(scene, (lx + 0.06, ly + 0.06, TABLE_TOP), cell.at(tx, ty, TABLE_TOP / 2), GRAY))
    for size, off in [((WALL_T, ly + 2 * WALL_T, TRAY_RIM_H), (-lx / 2 - WALL_T / 2, 0)),
                      ((WALL_T, ly + 2 * WALL_T, TRAY_RIM_H), (lx / 2 + WALL_T / 2, 0)),
                      ((lx, WALL_T, TRAY_RIM_H), (0, -ly / 2 - WALL_T / 2)),
                      ((lx, WALL_T, TRAY_RIM_H), (0, ly / 2 + WALL_T / 2))]:
        cell.walls.append(add_box(scene, size, cell.at(tx + off[0], ty + off[1], TABLE_TOP + TRAY_RIM_H / 2), TRAY_COLOR))
    cell.situation = situation
    cell.n_stand = N_NEIGHBOURS
    cell.n_pile = pile_geometry(spec)["k"] if situation == "pile" else 0
    obj, tex = make_book_assets(spec, out)
    s = SURFACES[spec.surface]
    for i in range(cell.n_stand + cell.n_pile + 1):
        cell.books.append(scene.add_entity(
            gs.morphs.Mesh(file=str(obj), pos=tuple(cell.at(bx, by, 0.6 + 0.06 * i))),
            material=gs.materials.Rigid(rho=spec.density, friction=s["friction"]),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(tex)))))
    cell.robot_ent = scene.add_entity(gs.morphs.MJCF(file=str(build_robot_xml(out, design=cell.design)),
                                                     pos=(cell.x0 + ROBOT_XY[0], ROBOT_XY[1], 0.0),
                                                     euler=(0.0, 0.0, ROBOT_YAW_DEG)))
    if render:
        # 横から（+y。ロボットの反対側）: 本を倒す動きは x-z 面にあるので横から見る。壁越しに床が見える高さ
        cell.cams["side"] = scene.add_camera(res=(800, 600), pos=tuple(cell.at(bx + 0.04, by + 0.80, 0.62)),
                                             lookat=tuple(cell.at(bx + 0.08, by, 0.10)), fov=40, env_idx=0)
        # 奥（+x）から: 爪は ±y の縁に入るので、本の天地の向きから縁と爪を見る（手前からは腕が隠す）
        cell.cams["front"] = scene.add_camera(res=(800, 600), pos=tuple(cell.at(bx + 0.55, by + 0.04, 0.55)),
                                              lookat=tuple(cell.at(bx, by, 0.10)), fov=40, env_idx=0)
        cell.cams["place"] = scene.add_camera(res=(800, 600), pos=tuple(cell.at(tx - 0.05, ty + 0.60, TABLE_TOP + 0.30)),
                                              lookat=tuple(cell.at(tx, ty, TABLE_TOP + 0.02)), fov=40, env_idx=0)


def place_books(cell: Cell5b, spec, lean, stand_y, pile_jit) -> None:
    """乱数は全セルで同じものを使う。立った本 → 山 → 取る本（一番手前の立った本）。"""
    w, h, t = spec.size_m
    n = lean.shape[0]
    bx, by = BOX_C
    x_wall = cell.x0 + bx + BOX_IN[0] / 2
    poses = standing_layout(spec, cell.n_stand + 1, lean, x_wall, by + stand_y)
    for b, (c, q) in zip(cell.neighbours, poses[:-1]):
        b.set_pos(c)
        b.set_quat(q)
    c_f, q_f = poses[-1]
    cell.target.set_pos(c_f)
    cell.target.set_quat(q_f)
    hinge_x = (c_f + np.einsum("nij,j->ni", quat_to_mat(q_f), np.array([0.0, -h / 2, t / 2])))[:, 0]
    cell.rec["hinge_x_planned"] = hinge_x - cell.x0
    if cell.n_pile:
        g = pile_geometry(spec)
        x_edge = hinge_x - g["d_m"]  # 山の奥の縁
        for i, b in enumerate(cell.pile):
            pos = np.stack([x_edge - h / 2 + pile_jit[:, 0], by + pile_jit[:, 1],
                            np.full(n, BOX_FLOOR + t / 2 + i * (t + 0.001) + 0.001)], -1)
            b.set_pos(pos)
            b.set_quat(quat_mul(quat_z(pile_jit[:, 2]), np.broadcast_to(Q_LAID, (n, 4))))
        cell.rec["pile"] = g


def forward_tilt(mb: np.ndarray) -> np.ndarray:
    """本の軸（天の向き = 本の y）の、鉛直から手前（-x）への傾き（rad）。奥へ寄りかかっていれば負。平らなら 90°。"""
    u = mb[:, :, 1]
    return np.arctan2(-u[:, 0], u[:, 2])


def lip_contact_z(spec, d, phi=None):
    """右の爪の下面の中心を当てる位置（天の面の上、本の座標 z。表紙が +t/2）。
    立っている間: 爪の先は裏表紙の 3 mm 手前まで（隙間 2 mm の後ろの本の天に掛かると、後ろの本を押して手前の本を
    押し倒した）。根元は表紙の 1 mm 以上手前（長い爪は根元が表紙より手前に出る）。
    倒れていく間（30°〜60°）: 根元を表紙（寝かせると下面）の 1 mm 内側へずらす（寝かせ終わりに、表紙より手前に出た
    爪の根元が山の一番上の本に食い込んだ）。爪の先は裏表紙（寝かせると上面）の上の何も無いところへ出る。"""
    t, L = spec.thickness_m, d.lip_len_r
    s_early = min(-0.001, t - 0.003 - L)  # 表紙から奥へ測った爪の根元の位置
    s_late = 0.001
    a = 0.0 if phi is None else np.clip((np.degrees(phi) - 30.0) / 30.0, 0.0, 1.0)
    s_root = (1 - a) * s_early + a * s_late
    return t / 2 - (s_root + L / 2)


def gz_of(beta: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(beta), np.zeros_like(beta), -np.cos(beta)], -1)


def quat_of(beta: np.ndarray) -> np.ndarray:
    """指の開閉軸を x（右の指 = 爪がロボットの側 -x、爪の先は +x を向く）にした下向きの手を、y 軸まわりに傾けて
    手先を (sin β, 0, -cos β) に向ける。"""
    sh = beta.shape + (4,)
    base = quat_mul(np.broadcast_to(quat_z(math.pi / 2), sh), np.broadcast_to(Q_DOWN, sh))
    return quat_mul(quat_y(-beta), base)


def lay_hand_pose(p_contact: np.ndarray, beta: np.ndarray, d, q_r: float):
    """右の爪の下面の中心を p_contact に置く手の位置と向き（右の指の上下の関節が LAY_LIFT_PLAN のとき）。"""
    q = quat_of(beta)
    m = quat_to_mat(q)
    off = np.array([0.0, -q_r + d.lip_len_r / 2, d.lip_bottom_z + LAY_LIFT_PLAN])
    return p_contact - np.einsum("nij,j->ni", m, off), q


def set_forces_lay(r: Robot, lift_n: float) -> None:
    f = FORCE.copy()
    f[LIFT] = lift_n
    r.e.set_dofs_force_range(-f, f, np.arange(len(f)))


def pile_edge_point(cell: Cell5b, spec) -> np.ndarray:
    """山の一番上の本の、上面の奥（+x）の縁の中点。"""
    w, h, t = spec.size_m
    pp, mp = book_frame(cell.pile[-1])
    cw = pp[:, None, :] + np.einsum("nij,kj->nki", mp, box_corners(w, h, t))
    top = cw[:, :, 2] > pp[:, None, 2]
    xs = np.where(top, cw[:, :, 0], -np.inf)
    far = xs >= xs.max(axis=1, keepdims=True) - 0.02  # 上面の奥の 2 つの角（向きのばらつき 3° で x の差は 1 cm 未満）
    sel = top & far
    return (cw * sel[:, :, None]).sum(axis=1) / np.maximum(sel.sum(axis=1), 1)[:, None]


def support_top_z(cell: Cell5b, spec) -> np.ndarray:
    """寝かせた本が載る面の高さ（山の一番上の本の上面、または箱の床）。"""
    if cell.n_pile:
        return book_frame(cell.pile[-1])[0][:, 2] + spec.thickness_m / 2
    return np.full(cell.robot.n, BOX_FLOOR)


def lay_down_cycle(cell: Cell5b, spec, n: int, snap):
    """立っている本を、下の辺を軸に手前へ倒して平らに寝かせる（本を手から離さない）。測定は cell.rec に入れる。
    天の面に右の爪の下面（平ら・ステンレス）を押し付け、その摩擦で本の上の端を手前へ引き、倒れていく本を支える。
    押し付ける力は右の指の上下の関節の力の上限（LAY_PRESS）。吸盤は使わない（引き込んだまま）。"""
    r, d, rec = cell.robot, cell.design, cell.rec
    book = cell.target
    w, h, t = spec.size_m
    w_est = spec.weight_estimated_kg * GRAVITY
    f_press = float(np.clip(LAY_PRESS_RATIO * w_est, LAY_PRESS_MIN_N, LAY_PRESS_MAX_N))
    rec["f_press_n"] = f_press
    t0 = r.world.sim_time
    fo = d.finger_open_m
    q_r, q_l = LAY_Q_R, min(fo, LAY_Q_L)
    zc = lip_contact_z(spec, d)
    rec["lip_contact_z_mm"] = zc * 1000
    loc = np.array([0.0, h / 2, zc])

    def loc_at(ph):  # 傾き ph（環境ごと）での当てる位置（本の座標）
        z = lip_contact_z(spec, d, ph)
        return np.stack([np.zeros_like(z), np.full_like(z, h / 2), z], -1)
    set_forces_lay(r, f_press)
    r.target[:, FINGERS] = np.array([q_l, q_r])
    r.target[:, LIFT] = -d.lift_stroke_m  # 近づく間は爪を上げておく
    r.target[:, PLUNGER] = 0.0
    bmin, bmax = math.radians(LAY_BETA_MIN_DEG), math.radians(LAY_BETA_MAX_DEG)

    # 1. 爪を天の面の上（本の軸の延長）へ → 当てる
    pb, mb = book_frame(book)
    phi = forward_tilt(mb)
    rec["phi0_deg"] = np.degrees(phi)
    contact = pb + np.einsum("nij,j->ni", mb, loc)
    beta = np.clip(phi, bmin, bmax)
    gz = gz_of(beta)
    pos, quat = lay_hand_pose(contact, beta, d, q_r)
    # 箱の外の高いところ（0.6 m）を通って本の上へ行き、そこで手首を回してから下ろす
    g0 = r.gripper.get_pos().cpu().numpy()
    q0 = r.gripper.get_quat().cpu().numpy()
    up0 = g0.copy()
    up0[:, 2] = np.maximum(up0[:, 2], SAFE_Z)
    yield from g_cart(r, up0, q0, 60, n_way=4)
    pre = pos - 0.06 * gz
    high = pre.copy()
    high[:, 2] = np.maximum(pre[:, 2] + 0.05, SAFE_Z)
    yield from g_cart(r, high, quat, 150, n_way=15)
    yield from g_cart(r, pre, quat, 80, n_way=8)
    yield from g_cart(r, pos - 0.03 * gz, quat, 40, n_way=4)
    yield 20
    rec["above_head_err_mm"] = np.linalg.norm(r.gripper.get_pos().cpu().numpy() - (pos - 0.03 * gz), axis=1) * 1000
    rec["book_moved_before_touch_mm"] = np.linalg.norm(book_frame(book)[0] - pb, axis=1) * 1000
    snap(cell, "L1 lip above head")
    yield from g_cart(r, pos, quat, 40, n_way=4)
    yield from g_joints(r, LIFT, np.full(n, d.lift_stroke_m), 30)  # 下いっぱいを目標に → 力の上限で押す
    yield 20
    f_touch, _ = r.touching(book)
    rec["lip_on_head"] = f_touch
    rec["ik_err_max_mm"] = r.ik_err_max * 1000
    snap(cell, "L1 lip pressed on head")
    pos_cmd, quat_cmd = pos, quat

    # 2〜3. 下の辺（山に載ったら山の縁）を軸に回した位置へ、爪を押し付けたまま手を動かす
    done = np.zeros(n, bool)
    on_pile_seen = np.zeros(n, bool)
    onp = np.zeros(n, bool)
    shots = [5.0, 35.0, 60.0]
    trace: list = []
    phi_max = phi.copy()
    for it in range(LAY_MAX_ITERS):
        pb, mb = book_frame(book)
        phi = forward_tilt(mb)
        phi_max = np.maximum(phi_max, phi)
        cw = pb[:, None, :] + np.einsum("nij,kj->nki", mb, box_corners(w, h, t))
        low = cw[:, :, 2].min(axis=1)
        flat = np.abs(mb[:, 2, 2]) > math.cos(math.radians(LAY_FLAT_DEG))
        rest = np.abs(low - support_top_z(cell, spec)) < 0.005
        done |= flat & rest
        if done.all():
            break
        pf = pb + np.einsum("nij,j->ni", mb, np.array([0.0, -h / 2, t / 2]))  # 手前の下の辺
        pk = pb + np.einsum("nij,j->ni", mb, np.array([0.0, -h / 2, -t / 2]))  # 後ろの下の辺
        pivot = np.where((pk[:, 2] < pf[:, 2])[:, None], pk, pf)
        if cell.n_pile:
            pe = pile_edge_point(cell, spec)
            nf = mb[:, :, 2]
            dist = np.einsum("ni,ni->n", pe - (pb + nf * t / 2), nf)
            s = np.einsum("ni,ni->n", pe - pb, mb[:, :, 1])
            onp = (np.abs(dist) < 0.005) & (np.abs(s) < h / 2)
            on_pile_seen |= onp
            pivot = np.where(onp[:, None], pe, pivot)
        dphi = np.clip(math.radians(90.0 + 0.5) - phi, 0.0, math.radians(LAY_STEP_DEG))
        ry = rot_y(-dphi)
        c_now = pb + np.einsum("nij,nj->ni", mb, loc_at(phi + dphi))
        c_next = pivot + np.einsum("nij,nj->ni", ry, c_now - pivot)
        beta = np.clip(phi + dphi, bmin, bmax)
        pos, q = lay_hand_pose(c_next, beta, d, q_r)
        pos_cmd = np.where(done[:, None], pos_cmd, pos)
        quat_cmd = np.where(done[:, None], quat_cmd, q)
        goal = r.ik(pos_cmd, quat_cmd)
        start = r.target[:, :7].copy()
        for k in range(1, LAY_STEPS_PER + 1):
            r.target[:, :7] = start + (goal - start) * (k / LAY_STEPS_PER)
            yield 1
        # 記録（環境ごと）: 傾き・爪が触れているか・軸の種類・右の爪の上下の関節の位置
        f_touch, _ = r.touching(book)
        ci_ = cell.robot_ent.get_contacts(with_entity=book)
        names = {lk.idx: lk.name for lk in cell.robot_ent.links}
        touch_links0 = sorted({names.get(a_, names.get(b_, "?")) for a_, b_, v_ in zip(
            ci_["link_a"][0].tolist(), ci_["link_b"][0].tolist(), ci_["valid_mask"][0].tolist()) if v_})
        rec.setdefault("lay_touch_links_env0", []).append(touch_links0)
        hand_err = np.linalg.norm(r.gripper.get_pos().cpu().numpy() - pos_cmd, axis=1) * 1000
        trace.append({"phi_deg": np.degrees(phi), "touch": f_touch, "touch_any": touching_robot(cell), "pivot_pile": onp.copy(),
                      "hand_err_mm": hand_err, "beta_deg": np.degrees(beta),
                      "nb_phi_deg": np.stack([np.degrees(forward_tilt(book_frame(b)[1])) for b in cell.neighbours], 1),
                      "pivot_back": pk[:, 2] < pf[:, 2], "lift_mm": r.e.get_qpos().cpu().numpy()[:, LIFT] * 1000})
        if shots and math.degrees(phi[0]) >= shots[0]:
            snap(cell, f"L2 tilted {math.degrees(phi[0]):.0f} deg")
            shots.pop(0)
    rec["lay_iters"] = it + 1
    rec["ik_held_steps"] = getattr(r, "ik_held", np.zeros(n, int)).copy()
    rec["lay_trace"] = {k: np.stack([s[k] for s in trace], 1) for k in trace[0]} if trace else {}
    rec["lay_done"] = done
    rec["on_pile_edge_seen"] = on_pile_seen
    rec["phi_max_deg"] = np.degrees(phi_max)
    rec["cup_on_head"] = rec["lip_on_head"]
    snap(cell, "L3 flat, lip still on")

    # 4. 爪を上げて、手を本の軸の向きに離してから上げる
    yield from g_joints(r, LIFT, np.full(n, -d.lift_stroke_m), 30)
    gz_c = quat_to_mat(quat_cmd)[:, :, 2]
    away = r.gripper.get_pos().cpu().numpy() - 0.04 * gz_c
    away[:, 2] += 0.08
    yield from g_cart(r, away, quat_cmd, 60, n_way=4)
    set_forces(r)
    yield 40
    pb, mb = book_frame(book)
    cw = pb[:, None, :] + np.einsum("nij,kj->nki", mb, box_corners(w, h, t))
    rec["laid_flat"] = np.abs(mb[:, 2, 2]) > 0.97
    rec["laid_gap_mm"] = (cw[:, :, 2].min(axis=1) - support_top_z(cell, spec)) * 1000
    rec["laid_on_support"] = np.abs(rec["laid_gap_mm"]) < 5.0
    if cell.n_pile:
        pp, _ = book_frame(cell.pile[-1])
        rec["laid_offset_x_mm"] = (pb[:, 0] - pp[:, 0]) * 1000  # 寝かせた本の中心と山の一番上の本の中心の差（+ は奥）
    rec["lay_s"] = r.world.sim_time - t0
    snap(cell, "L4 laid flat, hand away")


def ik_strong(r: Robot, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Robot.ik と同じ。反復の上限だけ 20 → 100（寝かせる間は手首を大きく回すので、20 回では収束せず
    腕が別の解へ飛んで本を叩いた）。"""
    q, err = r.e.inverse_kinematics(link=r.gripper, pos=pos, quat=quat, dofs_idx_local=np.arange(7),
                                    init_qpos=r.e.get_qpos(), return_error=True, max_solver_iters=100)
    e = err.cpu().numpy().reshape(r.n, -1)
    r.ik_err_max = np.maximum(r.ik_err_max, np.linalg.norm(e[:, :3], axis=1))
    return q.cpu().numpy()[:, :7]


def ik_safe(r: Robot, pos: np.ndarray, quat: np.ndarray, tol: float = 0.005, hold: float = 0.02) -> np.ndarray:
    """ik_strong に、収束しなかった環境の救済を足したもの。いまの姿勢から解いて誤差が tol を超えた環境は、
    いまの目標（指令中の関節角）から解き直して良いほうを使う。それでも hold を超える環境は動かさない
    （遠い別の解へ腕が飛ぶと、本や箱を叩く。最初の版で 384 mm 飛んで立った本を 3 冊とも倒した）。"""
    def solve(init):
        q, err = r.e.inverse_kinematics(link=r.gripper, pos=pos, quat=quat, dofs_idx_local=np.arange(7),
                                        init_qpos=init, return_error=True, max_solver_iters=100)
        return q.cpu().numpy()[:, :7], np.linalg.norm(err.cpu().numpy().reshape(r.n, -1)[:, :3], axis=1)
    q1, e1 = solve(r.e.get_qpos())
    bad = e1 > tol
    if bad.any():
        init = r.e.get_qpos().cpu().numpy().copy()
        init[:, :7] = r.target[:, :7]
        q2, e2 = solve(init)
        better = bad & (e2 < e1)
        q1 = np.where(better[:, None], q2, q1)
        e1 = np.where(better, e2, e1)
    held = e1 > hold
    r.ik_held = getattr(r, "ik_held", np.zeros(r.n, int)) + held
    r.ik_err_max = np.maximum(r.ik_err_max, np.where(held, 0.0, e1))
    return np.where(held[:, None], r.target[:, :7], q1)


def standing_cycle(cell: Cell5b, spec, n: int, snap):
    t0 = cell.robot.world.sim_time
    r = cell.robot
    r.ik = lambda pos, quat: ik_safe(r, pos, quat)  # 寝かせる間だけ（取る・運ぶは試行5 と同じ IK）
    yield from lay_down_cycle(cell, spec, n, snap)
    del r.ik
    if cell.situation == "pile":
        yield from pile_pick_cycle(cell, spec, n, snap)
    else:
        yield from floor_pick_cycle(cell, spec, n, snap)
    cell.rec["total_cycle_s"] = cell.robot.world.sim_time - t0
    cell.rec["ik_err_max_all_mm"] = cell.robot.ik_err_max * 1000


def pile_pick_cycle(cell: Cell5b, spec, n: int, snap, situation: str = "stack"):
    """寝かせた本を、平積みの山の一番上から取る（試行5 の slide_under_cycle の写し。変えたのは [5b] の 3 か所だけ:
    爪の側を +y（トレーの側）に固定・箱の壁の確かめを ±y に・後半を carry_and_place に分けた）。

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
    e = xb * np.where(xb[:, 1] >= 0, 1.0, -1.0)[:, None]  # [5b] 爪の側（+y 側の本の縁。寝かせた本の横幅は ±y を向く）
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
    # [5b] 左右の指の外面が箱の ±y の壁に当たらないか（e はほぼ +y）
    by = cell.at(*BOX_C)[1]
    ey = np.clip(e[:, 1], 1e-3, None)
    fits &= u_rf0 + FINGER_PLATE_T + 0.003 < (by + BOX_IN[1] / 2 - p0[:, 1]) / ey
    fits &= -u_lf0 + FINGER_PLATE_T + 0.003 < (p0[:, 1] - (by - BOX_IN[1] / 2)) / ey
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
        down = -np.sign(mb[:, 2, 2])  # [5b] 本の下面の向き（本の座標 z）。寝かせた本は表紙が下（試行5 は表が上だけ）
        under[side] = (np.abs(loc[:, 0]) < w / 2 - 0.002) & (loc[:, 2] * down > t / 2 - 0.001)
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


    # [5b] 後半（持ち上げる・運ぶ・置く）は試行5 と同じ
    yield from carry_and_place(cell, spec, n, snap, p0, quat, e, left_low, f_top, t_start)



def carry_and_place(cell: Cell5b, spec, n: int, snap, p0, quat, e, left_low, f_top, t_start):
    """試行5 の slide_under_cycle の 7・8（持ち上げる → 運ぶ → 置いて爪を抜く）の写し。中身は変えていない。"""
    r, d, rec = cell.robot, cell.design, cell.rec
    book = cell.target
    w, h, t = spec.size_m
    fo = d.finger_open_m
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
            still = np.where(loc[:, 2] * np.sign(mb[:, 2, 2]) < 0, w / 2 - np.abs(loc[:, 0]) + 0.004, 0.0)  # [5b] 表裏を問わない  # 先端までの残りの深さ（探る点は先端の 4 mm 手前）
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



def floor_pick_cycle(cell: Cell5b, spec, n: int, snap):
    """台の縁の無い床（箱の床）に寝かせた本を取る。ちりとり方式に要る「載っている面の縁」が無い。
    吸引あり: 爪の側の縁の近くに吸盤を当てて吸着し、FLOOR_PEEL だけ引いて縁を数 mm めくり、その下へ右の爪を入れる
    （吸引の役目は「端を数 mm めくる」だけ。持ち上げは爪）。吸引なし: めくらずに同じ手順（くさびだけ）。
    両側の爪の左は、試行5 の平置きと同じく、右の爪で本を約 6° 傾けて左の爪を床に沿って押し込む。後半は試行5 と同じ。"""
    r, d, rec = cell.robot, cell.design, cell.rec
    book = cell.target
    w, h, t = spec.size_m
    plan = grip_plan(spec)
    w_est = spec.weight_estimated_kg * GRAVITY
    f_push = float(min(plan["grip_limit_n"], max(8.0, 3.0 * w_est)))  # 試行5 と同じ
    f_side = float(min(plan["grip_limit_n"], max(3.0, 0.75 * w_est)))
    f_top = float(np.clip(0.5 * w_est, 2.0, 10.0))
    rec.update(f_push_n=f_push, f_side_n=f_side, f_top_n=f_top)
    t_start = r.world.sim_time
    p0, m0 = book_frame(book)
    rec["p0"], rec["up0"] = p0.copy(), m0[:, 2, 2].copy()
    xb = m0[:, :, 0].copy()
    xb[:, 2] = 0.0
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    e = xb * np.where(xb[:, 1] >= 0, 1.0, -1.0)[:, None]
    quat = lip_side_quat(e)
    gy = -e
    z_s = p0[:, 2] - t / 2
    L_r, L_l, fo = d.lip_len_r, d.lip_len_l, d.finger_open_m
    u_rf0 = np.full(n, w / 2 + MARGIN + L_r)
    u_lf0 = np.full(n, -(w / 2 + L_l + MARGIN_L))
    # 吸盤（手のひらの中心）は爪の側の縁にできるだけ寄せる（めくるてこを長く）。左の指が届く範囲で
    lo, hi_ = u_rf0 - fo, u_lf0 + fo
    u_p = np.clip(np.full(n, w / 2 - FLOOR_CUP_IN), lo, np.maximum(lo, hi_))
    fits = lo <= hi_
    by = cell.at(*BOX_C)[1]
    ey = np.clip(e[:, 1], 1e-3, None)
    fits &= u_rf0 + FINGER_PLATE_T + 0.003 < (by + BOX_IN[1] / 2 - p0[:, 1]) / ey
    fits &= -u_lf0 + FINGER_PLATE_T + 0.003 < (p0[:, 1] - (by - BOX_IN[1] / 2)) / ey
    rec.update(fits_env=fits, palm_u_mm=u_p * 1000, push_mm=np.zeros(n), support_edge_mm=np.full(n, np.nan))
    q_r0, q_l0 = np.clip(u_rf0 - u_p, 0, fo), np.clip(u_p - u_lf0, 0, fo)
    left_low = d.fingertip_z + (d.lip_t_m if d.lips == "both" else 0.0)
    g_z = z_s + 0.002 + left_low  # 左のいちばん下を床の 2 mm 上に
    lift_down = np.clip(np.full(n, left_low + 0.0015 - d.lip_lowest_z), -d.lift_stroke_m, d.lift_stroke_m)  # 右の爪の下面を床の 0.5 mm 上
    rec["lift_down_mm"] = lift_down * 1000
    palm = p0 + u_p[:, None] * e
    palm[:, 2] = g_z

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
    snap(cell, "1 blade beside the edge (floor)")

    # めくる（吸引ありだけ）: 吸盤を上面に当てて吸着し、引く
    g_now = r.gripper.get_pos().cpu().numpy()
    top_gz = np.einsum("ni,ni->n", p0 - g_now, r.approach_axis()) - t / 2
    edge_low0 = _edge_bottom(book, spec, e)
    if cell.suction:
        r.suction.attach(book, SURFACES[spec.surface]["suction_max_n"])
        q_cup = np.clip(top_gz - CUP_TIP_RETRACTED_Z + 0.003, 0.0, PLUNGER_STROKE)
        yield from g_joints(r, PLUNGER, q_cup, 40)
        r.suction.on[:] = True
        yield 15
        yield from g_joints(r, PLUNGER, np.clip(q_cup - FLOOR_PEEL, 0.0, PLUNGER_STROKE), 40)
        yield 15
    rec["peel_rise_mm"] = (_edge_bottom(book, spec, e) - edge_low0) * 1000
    rec["sealed"] = r.suction.sealed.copy() if cell.suction else np.zeros(n, bool)
    # 右の爪を本の縁の下へ（右の指の内面を本の縁の 2 mm 外まで寄せる）
    yield from g_joints(r, FINGERS[1], np.clip(w / 2 + SLIDE_STOP - u_p, 0, fo), PUSH_STEPS)
    yield 20
    snap(cell, "2 blade pushed under the peeled edge")
    if d.lips == "both":
        raise_w = float(np.clip(w * math.tan(math.radians(RAISE_TILT_BOTH_DEG)), 0.008, 0.020))
        yield from g_joints(r, LIFT, np.clip(r.target[:, LIFT] - raise_w, -d.lift_stroke_m, d.lift_stroke_m), 60)
        lower = aim.copy()
        lower[:, 2] -= 0.0025
        yield from g_cart(r, lower, quat, 30, n_way=3)
        q_now = r.target[:, FINGERS[0]].copy()
        yield from g_joints(r, FINGERS[0], np.clip(q_now - L_l, 0, fo), 150)
        yield 10
        yield from g_cart(r, aim, quat, 30, n_way=3)
        yield from g_joints(r, LIFT, np.clip(r.target[:, LIFT] + RAISE_LIFT, -d.lift_stroke_m, d.lift_stroke_m), 40)
    else:
        yield from g_joints(r, LIFT, np.clip(r.target[:, LIFT] + RAISE_LIFT, -d.lift_stroke_m, d.lift_stroke_m), 60)
    yield 10
    pb, mb = book_frame(book)
    depth, under = {}, {}
    for side, p in lip_points(cell):
        loc = to_book(p, pb, mb)
        depth[side] = (w / 2 - np.abs(loc[:, 0])) * 1000
        down = -np.sign(mb[:, 2, 2])  # [5b] 本の下面の向き（本の座標 z）。寝かせた本は表紙が下（試行5 は表が上だけ）
        under[side] = (np.abs(loc[:, 0]) < w / 2 - 0.002) & (loc[:, 2] * down > t / 2 - 0.001)
    rec["insert_depth_mm"] = depth
    rec["inserted"] = under.get("r", np.zeros(n, bool))
    rec["inserted_l"] = under.get("l", np.ones(n, bool))
    snap(cell, "3 blade under, raised")

    # 6. 左右から軽く挟み、上から吸盤で押さえる（試行5 と同じ）
    set_forces(r, fingers=f_side, plunger=f_top)
    g_now = r.gripper.get_pos().cpu().numpy()
    s_book = np.einsum("ni,ni->n", pb - g_now, gy)
    q_sq = np.stack([s_book + w / 2 - SQUEEZE, -s_book + w / 2 - SQUEEZE], -1)
    yield from g_joints(r, FINGERS, np.clip(q_sq, 0, fo), 40)
    top_gz = np.einsum("ni,ni->n", pb - g_now, r.approach_axis()) - t / 2
    yield from g_joints(r, PLUNGER, np.clip(top_gz - CUP_TIP_RETRACTED_Z + 0.004, 0.0, PLUNGER_STROKE), 40)
    yield 20
    yield from carry_and_place(cell, spec, n, snap, p0, quat, e, left_low, f_top, t_start)


FLOOR_CUP_IN = 0.018  # 床の本: 吸盤の中心を爪の側の縁からこれだけ内側に（吸盤の半径 16 mm + 2 mm）
FLOOR_PEEL = 0.006  # 吸盤を引く量（縁を数 mm めくる）


def _edge_bottom(book, spec, e) -> np.ndarray:
    """本の e の側の縁の下の辺の高さ（2 つの角の平均）。"""
    w, h, t = spec.size_m
    pb, mb = book_frame(book)
    cw = pb[:, None, :] + np.einsum("nij,kj->nki", mb, box_corners(w, h, t))
    side = np.einsum("nki,ni->nk", cw - pb[:, None, :], e) > 0
    zs = np.where(side, cw[:, :, 2], np.inf)
    lowest2 = np.sort(zs, axis=1)[:, :2]
    return lowest2.mean(axis=1)


class Probe5b(T5.Probe):
    """試行5 の Probe と同じ判定。違いは 1 点だけ: 立っている本（傾き 45° 超）に触れているのは「寄りかかり」として
    壁と同じに数える（立てて並べた本は隣の本に寄りかかって立つ。plausibility.py の「壁に立てかけた本は数えない」の拡張）。"""

    STANDING_DEG = 45.0

    def __init__(self, scene, world: World, cells: list, n: int):
        super().__init__(scene, world, cells, n)
        self.link_book = np.full(self.n_links + 1, -1, int)
        self.link_book[self.book_links] = np.arange(len(self.book_links))
        self.lean_contacts = [np.zeros(n, bool) for _ in cells]  # 取る本が、立った本に寄りかかっていた（参考）

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
        pos = solver.get_links_pos(links_idx=bl).cpu().numpy().reshape(self.n, len(bl), 3)
        quat = solver.get_links_quat(links_idx=bl).cpu().numpy().reshape(self.n, len(bl), 4)
        up_z = quat_to_mat(quat)[:, :, 2, 2]
        standing = np.degrees(np.arccos(np.clip(np.abs(up_z), 0.0, 1.0))) > self.STANDING_DEG  # (n, B)
        is_a = la[:, :, None] == bl[None, None, :]
        is_b = lb[:, :, None] == bl[None, None, :]
        other_cat = np.where(is_a, cb[:, :, None], 0) + np.where(is_b, ca[:, :, None], 0)
        other_link = np.where(is_a, lb[:, :, None], la[:, :, None])
        ob = self.link_book[other_link]  # 相手の本の番号（本でなければ -1）
        ob_standing = standing[np.arange(self.n)[:, None, None], np.clip(ob, 0, None)] & (ob >= 0)
        lean = (is_a | is_b) & (other_cat == 3) & ob_standing
        other_cat = np.where(lean, 2, other_cat)
        other_hand = (is_a & hb[:, :, None]) | (is_b & ha[:, :, None])
        touch_robot = (other_cat == 1).any(axis=1)
        touch_wall = (other_cat == 2).any(axis=1)
        touch_sup = (other_cat == 3).any(axis=1)
        pen_hand = np.where(other_hand, pen[:, :, None], 0.0).max(axis=1, initial=0.0)
        f_hand = np.where(other_hand, force[:, :, None], 0.0).sum(axis=1)
        f_link = np.zeros_like(f_hand)
        for hl in self.hand_links:
            f_link = np.maximum(f_link, np.where(other_hand & (other_link == hl), force[:, :, None], 0.0).sum(axis=1))
        for ci in range(len(self.cells)):
            sel = np.nonzero(self.book_cell == ci)[0]
            self.total_force[ci] = np.maximum(self.total_force[ci], f_hand[:, sel].max(axis=1))
            self.lean_contacts[ci] |= lean[:, :, sel[-1]].any(axis=1)
        if len(self.lip_idx):
            lp = solver.get_links_pos(links_idx=self.lip_idx).cpu().numpy().reshape(self.n, len(self.lip_idx), 3)
            lm = quat_to_mat(solver.get_links_quat(links_idx=self.lip_idx).cpu().numpy().reshape(self.n, len(self.lip_idx), 4))
        t = self.world.sim_time
        dt = CHECK_EVERY * DT
        for ci, c in enumerate(self.cells):
            sel = np.nonzero(self.book_cell == ci)[0]
            inside = np.zeros((self.n, len(sel)), bool)
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
                bl_i = bl[sel[j]]
                on_b = (lb == bl_i) & ha
                on_a = (la == bl_i) & hb
                fz = np.where(on_b, fvec[..., 2], 0.0) - np.where(on_a, fvec[..., 2], 0.0)
                zb = np.einsum("nci,ni->nc", cpos - pos[:, None, sel[j]], bm[:, :, 2] * np.sign(bm[:, 2:3, 2]))  # 上向きの軸（表裏を問わない）
                below = zb < -c.rec["size"][2] / 4
                self.support[ci].append(np.where((on_a | on_b) & below, fz, 0.0).sum(axis=1)
                                        / (self.mass[ci] * GRAVITY))
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
    names = [s for s in args.designs.split(",") if s]
    modes = {"on": [True], "off": [False], "both": [True, False]}[args.suction]
    render = set(s for s in args.render.split(",") if s)
    rng = np.random.default_rng(args.seed)
    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=4),
                     rigid_options=gs.options.RigidOptions(enable_self_collision=False),
                     vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    cells: list[Cell5b] = []
    for name in names:
        for on in modes:
            key = f"{name}:{'on' if on else 'off'}"
            c = Cell5b(key, DESIGNS[name], on, len(cells) * CELL_DX)
            build_cell(scene, c, spec, args.situation, out, key in render)
            cells.append(c)
    t_build = time.time()
    scene.build(n_envs=n)
    build_s = time.time() - t_build
    world = World(scene, n, DT)
    for c in cells:
        c.robot = Robot(world, c.robot_ent, c.key, home=home_for(c.design), base_xy=(c.x0 + ROBOT_XY[0], ROBOT_XY[1]))
        c.robot.setup()
        c.fits = True
        c.rec["size"] = spec.size_m
        c.rec["mass"] = spec.weight_kg
    lean = np.radians(LEAN_DEG + rng.uniform(-LEAN_JITTER_DEG, LEAN_JITTER_DEG, n))
    stand_y = rng.uniform(-STAND_JITTER_Y, STAND_JITTER_Y, (n, N_NEIGHBOURS + 1))
    pile_jit = np.stack([rng.uniform(-PILE_JITTER_XY, PILE_JITTER_XY, n), rng.uniform(-PILE_JITTER_XY, PILE_JITTER_XY, n),
                         np.radians(rng.uniform(-PILE_JITTER_DEG, PILE_JITTER_DEG, n))], -1)
    for c in cells:
        place_books(c, spec, lean, stand_y, pile_jit)
    world.step(150)
    probe = Probe5b(scene, world, cells, n)
    world.monitors.append(probe)
    for c in cells:
        for name, cam in c.cams.items():
            if name in ("side", "place"):
                cam.start_recording(save_to_filename=str(out / f"{name}_{c.key.replace(':', '_')}.mp4"), fps=25)
                ctx["recording"].append(cam)

    def snap(cell: Cell5b, label: str) -> None:
        cell.rec.setdefault("phase_t", {})[label] = world.sim_time
        if "side" in cell.cams:
            cam = cell.cams["place"] if label[0] in "567" else cell.cams["side"] if label[0] == "L" else cell.cams["front"]
            rgb, *_ = cam.render()
            cell.frames.append((label, np.asarray(rgb)))
            if label[0] in "1234":  # ちりとりの間は横からも撮る
                rgb, *_ = cell.cams["side"].render()
                cell.frames.append((label + " (side)", np.asarray(rgb)))

    # 置いたあと落ち着いた姿勢を、乱していないかの基準にする（立った本は寄りかかって少し動く）
    for c in cells:
        snap(c, "L0 start")
    p_ref = [[book_frame(b) for b in c.books[:-1]] for c in cells]
    stand0 = [forward_tilt(book_frame(c.target)[1]) for c in cells]
    probe.active = True
    t_sim0 = world.sim_time
    t_wall0 = time.time()
    run_generators(world, [standing_cycle(c, spec, n, snap) for c in cells])
    wall_s = time.time() - t_wall0
    probe.active = False

    rows = []
    cell_res = {}
    lx, ly = TRAY_IN
    t = spec.thickness_m
    for ci, c in enumerate(cells):
        mon = probe.mons[ci]
        rec = c.rec
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
        # 乱していないか: 山の本は平らのまま 3 cm 以内、残りの立った本は立ったまま 3 cm 以内
        others_ok = np.ones(n, bool)
        pile_ok = np.ones(n, bool)
        stand_ok = np.ones(n, bool)
        for bi, (b, (p_start, _)) in enumerate(zip(c.books[:-1], p_ref[ci])):
            pb, mb = book_frame(b)
            moved = np.linalg.norm(pb - p_start, axis=1) < 0.03
            if bi < c.n_stand:
                stand_ok &= (np.abs(mb[:, 2, 2]) < math.cos(math.radians(45))) & moved
            else:
                pile_ok &= (np.abs(mb[:, 2, 2]) > 0.97) & moved
        others_ok = pile_ok & stand_ok
        lay_ok = rec["lay_done"] & rec["laid_flat"] & rec["laid_on_support"]
        sup = np.array(probe.support[ci]) if probe.support[ci] else np.zeros((1, n))
        sup_med = np.median(sup, axis=0)
        from_below = rec["inserted"] & rec["inserted_l"] & (sup_med >= SUPPORT_MIN_RATIO)
        rec["support_ratio_median"] = sup_med
        rec["from_below"] = from_below
        clean_release = rec["release_rise_mm"] <= 5.0
        success_raw = (lay_ok & rec["fits_env"] & rec["lifted"] & rec["carried"] & placed & others_ok & from_below
                       & clean_release)
        success = success_raw & mon.ok
        modes_l = []
        for i in range(n):
            if success[i]:
                modes_l.append("ok")
            elif success_raw[i]:
                modes_l.append("implausible:" + "+".join(k for k in KINDS if mon.violated[k][i]))
            elif not rec["cup_on_head"][i]:
                modes_l.append("lay:cup_missed_head")
            elif not rec["lay_done"][i]:
                modes_l.append("lay:not_flat_in_time" + ("(on_pile_edge)" if rec["on_pile_edge_seen"][i] else ""))
            elif not (rec["laid_flat"][i] and rec["laid_on_support"][i]):
                modes_l.append("lay:moved_after_release")
            elif not rec["fits_env"][i]:
                modes_l.append("no_fit")
            elif not (rec["inserted"][i] and rec["inserted_l"][i]):
                modes_l.append("not_inserted" + ("" if not rec["lifted"][i] else "(pinch_lift)"))
            elif not rec["lifted"][i]:
                modes_l.append("lift_failed")
            elif not rec["carried"][i]:
                modes_l.append("dropped_in_carry")
            elif not inside[i]:
                modes_l.append("place_outside")
            elif rec["withdraw_short_mm"][i] > 0:
                modes_l.append("cannot_withdraw")
            elif rec["release_rise_mm"][i] > 5.0:
                modes_l.append("dragged_on_release")
            elif not (flat[i] and on_floor[i]):
                modes_l.append("place_not_flat")
            elif not released[i]:
                modes_l.append("not_released")
            elif not pile_ok[i]:
                modes_l.append("pile_disturbed")
            elif not stand_ok[i]:
                modes_l.append("standing_disturbed")
            elif not from_below[i]:
                modes_l.append("held_by_pinch(support<0.5)")
            else:
                modes_l.append("other")
        rec.update(placed=placed, place_inside=inside, place_flat=flat, place_on_floor=on_floor, others_ok=others_ok,
                   pile_ok=pile_ok, stand_ok=stand_ok, lay_ok=lay_ok, end_pos=pe, end_up_z=up_end,
                   target_leaned_on_standing=probe.lean_contacts[ci], phi_settled_deg=np.degrees(stand0[ci]))
        pe_env = mon.per_env()
        cell_res[c.key] = {
            "design": c.design.name, "suction_assist": c.suction,
            "success_n": int(success.sum()), "success_raw_n": int(success_raw.sum()),
            "implausible_n": int((success_raw & ~mon.ok).sum()), "violations_any_n": int((~mon.ok).sum()),
            "violations": {k: int(mon.violated[k].sum()) for k in KINDS},
            "lay_ok_n": int(lay_ok.sum()),
            "failure_modes": {m: modes_l.count(m) for m in sorted(set(modes_l))},
            "peak_hand_force_n": {"median": float(np.median(mon.peak_hand_force_n)), "max": float(mon.peak_hand_force_n.max())},
            "max_penetration_mm": float(mon.max_penetration_m.max() * 1000),
            "peak_total_hand_force_n": {"median": float(np.median(probe.total_force[ci])), "max": float(probe.total_force[ci].max())},
            "lay_s": float(rec.get("lay_s", float("nan"))),
            "cycle_s": float(rec.get("total_cycle_s", float("nan"))),
            "forces_n": {k: rec.get(k) for k in ("f_press_n", "f_push_n", "f_side_n", "f_top_n")},
            "per_env": {"mode": modes_l, **{k: v for k, v in rec.items() if k not in ("size", "p0", "off_before_carry")},
                        **pe_env},
        }
        rows.append({
            "run_book": spec.book_id, "surface": spec.surface, "situation": args.situation, "cell": c.key,
            "design": c.design.name, "suction": "on" if c.suction else "off", "n": n,
            "success": int(success.sum()), "success_raw": int(success_raw.sum()),
            "implausible": int((success_raw & ~mon.ok).sum()), "viol_any": int((~mon.ok).sum()),
            **{f"viol_{k}": int(mon.violated[k].sum()) for k in KINDS},
            "lay_ok": int(lay_ok.sum()),
            "peak_force_med_n": round(float(np.median(mon.peak_hand_force_n)), 1),
            "peak_force_max_n": round(float(mon.peak_hand_force_n.max()), 1),
            "max_pen_mm": round(float(mon.max_penetration_m.max() * 1000), 2),
            "lay_s": round(float(rec.get("lay_s", float("nan"))), 2),
            "cycle_s": round(float(rec.get("total_cycle_s", float("nan"))), 2),
            "support_med": round(float(np.median(sup_med)), 2),
            "modes": ";".join(f"{m}={modes_l.count(m)}" for m in sorted(set(modes_l))),
        })
        if c.frames:
            montage(c.frames, out / f"montage_{c.key.replace(':', '_')}.png")
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return {
        "criterion": "5b_standing",
        "acceptance": "寝かせて平ら → 爪で下から支えて持ち上げ・運ぶ間に落ちない・トレーの中に平ら・山と残りの立った本を乱さない、"
                      "かつありえない状態が無い",
        "acceptance_ok": any(v["success_n"] >= math.ceil(0.8 * n) for v in cell_res.values()),
        "n_envs": n,
        "book": {"book_id": spec.book_id, "format": spec.format_name, "size_mm": [round(v * 1000, 1) for v in spec.size_m],
                 "weight_g": round(spec.weight_kg * 1000, 1), "weight_est_g": round(spec.weight_estimated_kg * 1000, 1),
                 "surface": spec.surface},
        "situation": args.situation,
        "scene": {"box_in_m": BOX_IN, "box_wall_h_m": BOX_WALL_H, "standing_books": N_NEIGHBOURS + 1,
                  "neighbours_behind": N_NEIGHBOURS, "lean_deg": [LEAN_DEG, LEAN_JITTER_DEG], "stand_gap_m": STAND_GAP,
                  "pile": pile_geometry(spec) if args.situation == "pile" else None},
        "lay_params": {"press_n": cells[0].rec.get("f_press_n"), "press_ratio": LAY_PRESS_RATIO, "step_deg": LAY_STEP_DEG,
                       "steps_per": LAY_STEPS_PER, "beta_deg": [LAY_BETA_MIN_DEG, LAY_BETA_MAX_DEG],
                       "lift_plan_m": LAY_LIFT_PLAN, "q_r_m": LAY_Q_R, "q_l_m": LAY_Q_L},
        "plausibility_rules": {**Rules().describe(), "lean": "立っている本（傾き 45° 超）に触れているのは壁に立てかけたのと同じに数える"},
        "designs": {c.design.name: c.design.params() for c in cells},
        "cells": cell_res,
        "carry": {"distance_m_median": float(np.median(np.concatenate([c.rec["carry_m"] for c in cells if "carry_m" in c.rec]
                                                                     or [np.array([np.nan])]))),
                  "duration_s": CARRY_STEPS * DT, "profile": "min_jerk",
                  "peak_acc_m_s2_planned_median": float(5.7735 * np.median(np.concatenate(
                      [c.rec["carry_m"] for c in cells if "carry_m" in c.rec] or [np.array([np.nan])])) / (CARRY_STEPS * DT) ** 2)},
        "wall_build_s": build_s, "wall_run_s": wall_s, "sim_s": world.sim_time - t_sim0,
    }


if __name__ == "__main__":
    main()
