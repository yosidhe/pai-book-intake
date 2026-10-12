"""試行4: 検品トレー（縁の低いトレー + 真上のカメラ）の中で、ロボット A だけで本を裏返す（はみ出しなし）。

背景: 受け渡し台の縁から本をはみ出させて裏返す案（試行2-7）は落下の恐れがあり、裏返し専用の腕も置かない。
     本はトレーに平らに置き、真上のカメラでバーコードが見えなければ A が裏返す（やりとりは inspection.py）。

裏返しの動き（倒し裏返し）
  1. 本は表紙（バーコードのある +z 面）を下にしてトレーの中央に平らに置かれている（±1 cm・±5° のばらつき）
  2. A は真上から吸盤を当てて本を浮かせ、指で本の天地（長い辺の両端）を挟む。
     指の開閉軸を本の長い軸に合わせるのが要点（横幅を挟むと、本を立てたときに指と手のひらが本の下に来て
     床に当たる。手のひらは開閉軸の向きに 26 cm あるため）
  3. 本の長い軸まわりに手首を tilt（90°＋α）回し、本を長い辺で立てる。本の長い軸はトレーの半径の向き、
     手は本の横（接線の向き）で指先は水平。α は手の側へ倒す向き
  4. 本の下の辺をトレーの床の release_mm 上まで下ろし、吸着を切って指を開き、手をすぐ真上へ逃がす
  5. 本は手の側（=吸盤を当てていた面の側）へ倒れて、バーコードの面が上になる。
     α が倒れ始める角度（atan(厚み/横幅)）より小さいと本は立ったまま残るので、--push のときは
     指を閉じた手を本の向こう側から下ろし、上の辺を手前へ押して倒す
  6. 真上のカメラで読む（環境 0 だけ描画。zxing-cpp で回しながら読み、向きも出す）

倒れる先は必ず手の側（本の向こう側へ倒れると、元と同じ面が下になる）。手は倒れる本の通り道にいるので、
放したらすぐ逃がす。立てる位置は、倒れた後の本の中心がトレーの中央に戻るように決める（2 回目の裏返しも同じ動き）。

この試行で分かったこと
  - 指先を外向き（半径の向き）にして立てる案は、腕は届くが joint5 が限界（2.9 rad）に張り付き、手先の直線補間でも
    関節空間の補間でも腕が振り回されて本を投げた。本の長い軸を半径の向きに置き、指先を接線の向きにすると
    IK の残差 0.5 mm 以下で、関節が限界から離れた解がある
  - 90° 以上回す動きは手先の直線補間（move_linear）にしない。手首を回した関節角を初期値に IK を解き、関節空間で補間する
  - ツール v2 で本の天地を挟めるのは長さ約 175 mm まで（指の最大開き 210 mm − 右の爪 25 mm − 隙間）。
    四六判（188 mm）は、吸盤で引き上げても（爪に当たる）、指で囲んで下ろしても（爪が本の上に載る）挟めなかった。
    吸盤だけで持つ案も、持ち上げた後に本が振れて吸着が外れた（爪付きの指先に面を押し当てても同じ）

判定（環境ごと）: 落ち着いた後、本の z 軸が上を向き（up_z > 0.9。最初は −1）、四隅がすべてトレーの内側、
               本がテーブルの上（落ちていない）。環境 0 はバーコードが読めること
記録: 裏返しの時間（シミュレーションの秒）・指の力の最大・IK の残差・着地位置のばらつき

使い方: python spikes/trial4_tipflip.py --book book-03 --surface smooth --n-envs 16
出力: /workspace/outputs/trial4_tipflip/<run_id>/（overview_env0.mp4・closeup_env0.mp4・montage_env0.png・
      topcam_before/after_env0.png・bus.jsonl・result.json）
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import genesis as gs
from pai_book_intake.book_assets import make_book_assets
from pai_book_intake.books import DEFAULT_CATALOG, SURFACES, grip_plan, load_catalog
from pai_book_intake.bus import Bus
from pai_book_intake.geometry import mat_to_quat, quat_mul, quat_to_mat, quat_z, yaw_of
from pai_book_intake.inspection import InspectionSession, image_yaw_to_world, wrap_deg
from pai_book_intake.runlog import classify, new_run, save_result
from pai_book_intake.tool import (CUP_TIP_EXTENDED_Z, CUP_TIP_RETRACTED_Z, FINGER_OPEN, FINGERS, FINGERTIP_Z, LIP_LEN,
                                  LIP_T, PLUNGER, PLUNGER_STROKE, Robot, World, build_robot_xml)
from pai_book_intake.vision import read_isbn_with_yaw

TRIAL = "trial4_tipflip"
DT = 0.01
TABLE_TOP = 0.30
GRAY = (0.80, 0.80, 0.78)
TRAY_COLOR = (0.55, 0.57, 0.60)
RIM_T = 0.01
TOPCAM_H = 0.30  # トレーの床からカメラまで（試行2-8・P1 の台と同じ）
# 持ち方（--grasp）
#   pinch   吸盤で本を指の間へ引き上げ（プランジャ HOLD_PLUNGER まで）、指で本の天地を挟む。
#           右の爪が中心から 80 mm まで内側へ張り出しているので、挟めるのは長さ PINCH_MAX_H まで。
#           四六判（188 mm）は、指を開き切っても爪が本の端の上に載り、引き上げても下ろして囲んでも挟めなかった
#           （試行4: 引き上げ 16/16 取り落とし、囲む案も爪が本の上に載った）
#   suction 指は閉じたまま、吸盤（伸ばしたまま）だけで持つ。表面ごとの吸着力の上限（滑らか 12 N・ざらざら 3 N）が頼り
#   auto    長さが PINCH_MAX_H 以下なら pinch、超えれば suction
HOLD_PLUNGER = 0.015  # pinch: 本の底が右の爪より上に来る深さ（本の面は gripper の z=0.155）
SUCTION_HOLD_PLUNGER = FINGERTIP_Z + LIP_T - CUP_TIP_RETRACTED_Z - 0.004  # 爪の下面より 4 mm 奥まで引く
# 本を指の間の中央に置くと、挟めるのは 2 ×（105 − 25）− 10 = 150 mm まで（新書 173 mm は 16/16 取り落とし）。
# 150 mm を超える本は、本の中心を左の指（爪のない側）へ PINCH_OFFSET ずらして掴み、左の指は本の端の位置で止め、
# 右の指（爪あり）だけを閉じて挟む。これで 2 × 105 − 25 − 10 = 175 mm まで
PINCH_CENTER_MAX_H = 2 * (FINGER_OPEN - LIP_LEN) - 0.010  # 0.150 m
PINCH_OFFSET = LIP_LEN / 2  # 12.5 mm（左の指の内面と爪の先のちょうど中央）
PINCH_MAX_H = 2 * FINGER_OPEN - LIP_LEN - 0.010  # 0.175 m
LIFT_DZ = 0.10  # 挟んだ後、本の底をトレーの床からこれだけ上げる
PRE_DZ = 0.06  # 立てる位置の真上の中継点
PUSH_H_RATIO = 0.7  # 押すときの指先の高さ（本の横幅に対する割合）


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="book-03", help="カタログの book_id（book-03 文庫 / book-07 四六判）")
    ap.add_argument("--surface", choices=list(SURFACES), default=None, help="表面（省略時はカタログのまま）")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tilt-deg", type=float, default=95.0, help="手首を本の長い軸まわりに回す角度（90 で鉛直）")
    ap.add_argument("--release-mm", type=float, default=5.0, help="放すときの本の下の辺とトレーの床の隙間")
    ap.add_argument("--push", action=argparse.BooleanOptionalAction, default=True,
                    help="放した後、本の向こう側から上の辺を押して倒す")
    ap.add_argument("--tray-in", type=float, nargs=2, default=[0.36, 0.30],
                    help="トレーの内寸（v = 接線の向き, u = 半径の向き）m。本の長い軸は u、倒れる向きは v")
    ap.add_argument("--rim-mm", type=float, default=18.0)
    ap.add_argument("--tray-r", type=float, default=0.50, help="A の台座からトレーの中心まで（A の左手 +y 方向）")
    ap.add_argument("--grasp", choices=["auto", "pinch", "suction"], default="auto",
                    help="pinch: 指で天地を挟む（長さ 175 mm まで）/ suction: 吸盤だけ / auto: 長さで選ぶ")
    ap.add_argument("--stand-fix", action=argparse.BooleanOptionalAction, default=True,
                    help="立てた後、本の実測の向きと目標の差を手の向きで 1 回打ち消す")
    ap.add_argument("--hand-side", type=int, choices=[1, -1], default=1,
                    help="立てるときの指先の向き（+1: トレーの v 方向 = 世界 -x、-1: その逆）")
    ap.add_argument("--jitter-cm", type=float, default=1.0)
    ap.add_argument("--jitter-deg", type=float, default=5.0)
    ap.add_argument("--no-render", action="store_true", help="動画とカメラを使わない（環境 0 の読み取りもしない）")
    return ap.parse_args()


class Tray:
    """トレーの座標: 原点は床の中心、u = A の台座から外向き（世界 +y）、v = z × u（世界 -x）。"""

    def __init__(self, r: float, inner: tuple[float, float], rim_h: float, hand_side: int = 1):
        self.c = np.array([0.0, r])
        self.u = np.array([0.0, 1.0])
        self.v = np.array([-1.0, 0.0])
        # 本の長い軸は u（半径の向き）。手を水平にしたときの指先の向き a は接線の向き（±v）。
        # 指先を外向き（+u）にする案は、腕が届いても手首（joint5）が限界に張り付き、IK が枝を飛んで腕が振り回された
        self.hs = int(hand_side)
        self.a = self.hs * self.v
        self.len_v, self.len_u = inner
        self.rim_h = rim_h
        self.floor = TABLE_TOP

    def world(self, u, v, z):
        u, v, z = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (u, v, z)))
        xy = self.c + u[..., None] * self.u + v[..., None] * self.v
        return np.concatenate([xy, z[..., None]], -1)

    def local(self, p):
        d = np.asarray(p)[..., :2] - self.c
        return d @ self.u, d @ self.v

    def at(self, along_a, along_l, z):
        """a（指先の向き）と l（本の長い軸 = u）で書いた位置。"""
        return self.world(along_l, self.hs * np.asarray(along_a, dtype=float), z)

    @property
    def a_yaw(self) -> float:
        return math.atan2(self.a[1], self.a[0])

    @property
    def u_yaw(self) -> float:
        return math.atan2(self.u[1], self.u[0])


def book_corners(pos, quat, size):
    """(n, 8, 3) 本の 8 隅（世界）。"""
    w, h, t = size
    loc = np.array([[sx * w / 2, sy * h / 2, sz * t / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return pos[:, None, :] + np.einsum("nij,kj->nki", quat_to_mat(quat), loc)


def build(args, spec, out: Path, tray: Tray):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    lv, lu = tray.len_v, tray.len_u
    # テーブル（トレーより一回り大きい）とトレーの縁。トレーの床はテーブルの上面
    tw, td = lv + 0.10, lu + 0.10
    c = tray.world(0.0, 0.0, TABLE_TOP / 2)
    scene.add_entity(gs.morphs.Box(size=(tw, td, TABLE_TOP), pos=tuple(c), fixed=True),
                     surface=gs.surfaces.Default(color=GRAY))
    h = tray.rim_h
    for (su, sv), (pu, pv) in [
        ((RIM_T, lv + 2 * RIM_T), (-lu / 2 - RIM_T / 2, 0.0)),
        ((RIM_T, lv + 2 * RIM_T), (lu / 2 + RIM_T / 2, 0.0)),
        ((lu, RIM_T), (0.0, -lv / 2 - RIM_T / 2)),
        ((lu, RIM_T), (0.0, lv / 2 + RIM_T / 2)),
    ]:
        # u = 世界 +y、v = 世界 -x なので、箱の寸法は (x, y) = (sv, su)
        p = tray.world(pu, pv, TABLE_TOP + h / 2)
        scene.add_entity(gs.morphs.Box(size=(sv, su, h), pos=tuple(p), fixed=True),
                         surface=gs.surfaces.Default(color=TRAY_COLOR))
    obj, tex = make_book_assets(spec, out)
    s = SURFACES[spec.surface]
    book = scene.add_entity(
        gs.morphs.Mesh(file=str(obj), pos=tuple(tray.world(0.0, 0.0, TABLE_TOP + 0.05))),
        material=gs.materials.Rigid(rho=spec.density, friction=s["friction"]),
        surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(tex))),
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=str(build_robot_xml(out, rounded=True)), pos=(0.0, 0.0, 0.0)))
    cams = {}
    if not args.no_render:
        tc = tray.world(0.0, 0.0, TABLE_TOP)
        cams["top"] = scene.add_camera(res=(1920, 1440), pos=(tc[0], tc[1], TABLE_TOP + TOPCAM_H),
                                       lookat=tuple(tc), up=(tray.u[0], tray.u[1], 0.0), fov=40, env_idx=0)
        cams["overview"] = scene.add_camera(res=(960, 540), pos=(1.05, 1.25, 1.05), lookat=(0.0, 0.38, 0.38),
                                            fov=45, env_idx=0)
        # 本の長い軸の向き（トレーの外側 +u）から見る: 倒れる様子が横から見える
        side = tray.world(0.62, 0.0, TABLE_TOP + 0.16)
        cams["closeup"] = scene.add_camera(res=(800, 600), pos=tuple(side),
                                           lookat=tuple(tray.world(0.0, 0.0, TABLE_TOP + 0.07)), fov=50, env_idx=0)
        # 取るときの指と本の端（倒れる向き a の側から見る）
        cams["pickside"] = scene.add_camera(res=(800, 450), pos=tuple(tray.at(-0.45, 0.0, TABLE_TOP + 0.05)),
                                            lookat=tuple(tray.at(0.0, 0.0, TABLE_TOP + 0.03)), fov=35, env_idx=0)
    scene.build(n_envs=args.n_envs)
    return scene, book, robot, cams


class Probe:
    """毎歩、指の力の最大を記録する（World.step の hook として渡す）。"""

    def __init__(self, robot: Robot, n: int):
        self.r = robot
        self.active = False
        self.finger_force_max = np.zeros(n)
        self.watch_book = None  # 押す間だけ、指が本に触れたかを毎歩見る
        self.touched = np.zeros(n, bool)

    def __call__(self):
        if self.active:
            f = np.abs(self.r.e.get_dofs_control_force(FINGERS).cpu().numpy()).max(axis=1)
            self.finger_force_max = np.maximum(self.finger_force_max, f)
        if self.watch_book is not None:
            self.touched |= self.r.touching(self.watch_book)[0]


def mats_to_quats(m: np.ndarray) -> np.ndarray:
    return np.stack([mat_to_quat(mi) for mi in m])


def main() -> None:
    args = parse_args()
    run_id, out, meta = new_run(TRIAL, {k: v for k, v in vars(args).items()}, extra={"genesis": gs.__version__})
    bus = Bus(run_id, out / "bus.jsonl")
    ctx: dict = {"recording": []}
    result: dict = {}
    error = None
    try:
        result = run(args, out, bus, ctx)
    except BaseException as e:  # 状態 failed として記録してから、そのまま上げ直す
        error = e
        result = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        raise
    finally:
        for cam in ctx["recording"]:
            cam.stop_recording()
        bus.close()
        result["status"] = classify(bool(result.get("acceptance_ok", False)), error)
        save_result(out, meta, result)
        print(f"run_id={run_id} status={result['status']} out={out}")


def run(args, out: Path, bus: Bus, ctx: dict) -> dict:
    catalog = load_catalog(DEFAULT_CATALOG)
    spec = catalog.by_id(args.book)
    if args.surface is not None:
        spec = dataclasses.replace(spec, surface=args.surface)
    w, hgt, t = spec.size_m
    n = args.n_envs
    rng = np.random.default_rng(args.seed)
    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    tray = Tray(args.tray_r, tuple(args.tray_in), args.rim_mm / 1000, args.hand_side)
    t_build = time.time()
    scene, book, ent, cams = build(args, spec, out, tray)
    build_s = time.time() - t_build
    world = World(scene, n, DT)
    bus.set_sim_clock(lambda: world.sim_time)
    r = Robot(world, ent, "robot-a-pick", base_xy=(0.0, 0.0))
    plan = grip_plan(spec)
    r.setup(grip_limit_n=plan["grip_limit_n"])
    probe = Probe(r, n)
    orig_step = world.step
    world.step = lambda k=1, hook=None: orig_step(k, (lambda: (probe(), hook and hook())))  # noqa: E731

    # ---- 本をトレーの中央に、表紙（バーコード）を下にして置く ----
    jc, jd = args.jitter_cm / 100, math.radians(args.jitter_deg)
    du, dv = rng.uniform(-jc, jc, n), rng.uniform(-jc, jc, n)
    p0 = tray.world(du, dv, np.full(n, TABLE_TOP + t / 2 + 0.002))
    # 本の横幅（x）を a に、長い軸（y）を u に。x 軸まわりに 180° 回して表紙を下へ
    q0 = quat_mul(quat_z(tray.a_yaw + rng.uniform(-jd, jd, n)), np.tile([0.0, 1.0, 0.0, 0.0], (n, 1)))
    book.set_pos(p0)
    book.set_quat(q0)
    world.step(100)  # 待機姿勢を指令し続けながら落ち着かせる
    pos_start = book.get_pos().cpu().numpy()
    up_start = quat_to_mat(book.get_quat().cpu().numpy())[:, 2, 2]

    frames: list[tuple[str, np.ndarray]] = []

    def snap(label: str) -> None:
        if "closeup" in cams:
            rgb, *_ = cams["closeup"].render()
            frames.append((label, np.asarray(rgb)))

    if cams:
        for name in ("overview", "closeup", "pickside"):
            cams[name].start_recording(save_to_filename=str(out / f"{name}_env0.mp4"), fps=25)
            ctx["recording"].append(cams[name])
    snap("0 start (barcode down)")

    # ---- 検品: 置いた直後の撮影（attempt 0） ----
    sessions = [InspectionSession(bus, i, spec.book_id) for i in range(n)]
    scan_before = scan(args, cams, out, "before", bus, sessions, book, spec, tray, up_start)

    # ---- 裏返し ----
    need = np.array([s.state == "wait_flip" for s in sessions])
    if not need.all():
        raise RuntimeError(f"表紙を下にして置いたのに、裏返しの依頼が出ない環境がある: {np.nonzero(~need)[0]}")
    t0 = world.sim_time
    probe.active = True
    phase_t = {}
    rec = flip(r, book, spec, tray, args, snap, phase_t, probe)
    probe.active = False
    t_motion = world.sim_time - t0
    r.go_home(120)
    world.step(150)
    t_cycle = world.sim_time - t0
    snap("6 settled")
    for i in range(n):
        sessions[i].on_flip_done(ok=bool(rec["held_at_release"][i]))

    # ---- 判定 ----
    pe = book.get_pos().cpu().numpy()
    qe = book.get_quat().cpu().numpy()
    up_z = quat_to_mat(qe)[:, 2, 2]
    corners = book_corners(pe, qe, spec.size_m)
    cu, cv = tray.local(corners)
    inside = (np.abs(cu) <= tray.len_u / 2).all(axis=1) & (np.abs(cv) <= tray.len_v / 2).all(axis=1)
    lying = np.abs(pe[:, 2] - (TABLE_TOP + t / 2)) < 0.006
    on_table = pe[:, 2] > TABLE_TOP - 0.01
    flipped = up_z > 0.9
    success = flipped & inside & lying & on_table
    eu, ev = tray.local(pe)
    scan_after = scan(args, cams, out, "after", bus, sessions, book, spec, tray, up_z)
    if cams:
        success_env0_barcode = bool(scan_after["isbn_env0"] == spec.isbn13)
        success_rendered = success.copy()
        success_rendered[0] = success[0] and success_env0_barcode
    else:
        success_env0_barcode = None
        success_rendered = success.copy()

    modes = []
    for i in range(n):
        if success_rendered[i]:
            # 手の中で本がずれても（held_at_release=False）、結果が正しければ成功。ずれは per_env と slipped_n で見る
            modes.append("ok")
        elif not rec["held"][i]:
            modes.append("pick_failed")
        elif not rec["held_at_release"][i]:
            modes.append("slipped_in_hand")
        elif not on_table[i]:
            modes.append("fell_off_table")
        elif not inside[i]:
            modes.append("outside_tray")
        elif not flipped[i]:
            modes.append("standing" if abs(up_z[i]) < 0.5 else "not_flipped")
        elif not lying[i]:
            modes.append("not_flat")
        elif i == 0 and success_env0_barcode is False:
            modes.append("barcode_unread")
        else:
            modes.append("ok")
    n_ok = int(success_rendered.sum())
    if frames:
        montage(frames, out / "montage_env0.png")
    yaw_end = np.degrees(yaw_of(qe))
    result = {
        "criterion": "4_tipflip",
        "acceptance": "本の z 軸が上（up_z>0.9）・四隅がトレーの内側・平ら・テーブルの上。環境 0 はバーコードも読める。"
                      "16 環境の 80% 以上",
        "acceptance_ok": bool(n_ok >= math.ceil(0.8 * n)),
        "success_n": n_ok,
        "n_envs": n,
        "book": {"book_id": spec.book_id, "format": spec.format_name, "size_mm": [round(v * 1000, 1) for v in spec.size_m],
                 "weight_g": round(spec.weight_kg * 1000, 1), "surface": spec.surface,
                 "tip_angle_deg": round(math.degrees(math.atan2(t, w)), 2)},
        "params": {"tilt_deg": args.tilt_deg, "release_mm": args.release_mm, "push": args.push,
                   "tray_in_m": list(args.tray_in), "rim_mm": args.rim_mm, "tray_r_m": args.tray_r,
                   "grasp": rec["grasp"], "stand_fix": args.stand_fix, "hold_plunger_m": HOLD_PLUNGER, "lift_dz_m": LIFT_DZ, "grip_limit_n": plan["grip_limit_n"],
                   "suction_max_n": plan["suction_max_n"]},
        "failure_modes": {m: modes.count(m) for m in sorted(set(modes))},
        "slipped_n": int((~rec["held_at_release"] & rec["held"]).sum()),  # 手の中で 1.5 cm 以上ずれた環境の数
        "flip_motion_s": round(t_motion, 3),
        "flip_cycle_s": round(t_cycle, 3),
        "phase_s": phase_t,
        "finger_force_max_n": {"max": float(probe.finger_force_max.max()),
                               "median": float(np.median(probe.finger_force_max))},
        "ik_err_max_mm": {"carry_max": float(rec["ik_err_carry"].max() * 1000),
                          "stand_pos_err_max": float(rec["stand_pos_err"].max() * 1000)},
        "landing_mm": {"mean_u": float(eu.mean() * 1000), "mean_v": float(ev.mean() * 1000),
                       "std_u": float(eu.std() * 1000), "std_v": float(ev.std() * 1000),
                       "max_dist": float(np.hypot(eu, ev).max() * 1000)},
        "barcode_env0": {"before": scan_before.get("env0"), "after": scan_after.get("env0"),
                         "read_ok": success_env0_barcode},
        "wall_build_s": build_s,
        "per_env": {"mode": modes, "success": success_rendered, "up_z_start": up_start, "up_z_end": up_z,
                    "inside_tray": inside, "lying_flat": lying, "on_table": on_table,
                    "end_u_mm": eu * 1000, "end_v_mm": ev * 1000, "end_yaw_deg": yaw_end,
                    "start_u_mm": tray.local(pos_start)[0] * 1000, "start_v_mm": tray.local(pos_start)[1] * 1000,
                    "finger_force_max_n": probe.finger_force_max, **{k: v for k, v in rec.items()}},
    }
    return result


def flip(r: Robot, book, spec, tray: Tray, args, snap, phase_t: dict, probe: Probe) -> dict:
    """倒し裏返し（全環境で同じ手順）。"""
    world, n = r.world, r.n
    w, hgt, t = spec.size_m
    floor = tray.floor
    tick = [world.sim_time]

    def lap(name):
        phase_t[name] = round(world.sim_time - tick[0], 3)
        tick[0] = world.sim_time

    # ---- 1. 持ち上げる（pinch: 指の開閉軸 = 本の長い軸で天地を挟む / suction: 吸盤だけ） ----
    grasp = args.grasp if args.grasp != "auto" else ("pinch" if hgt <= PINCH_MAX_H else "suction")
    r.suction.attach(book, SURFACES[spec.surface]["suction_max_n"])
    r.target[:, FINGERS] = FINGER_OPEN if grasp == "pinch" else 0.0
    r.target[:, PLUNGER] = PLUNGER_STROKE
    p0 = book.get_pos().cpu().numpy()
    q_b = book.get_quat().cpu().numpy()
    long_yaw = np.arctan2(quat_to_mat(q_b)[:, 1, 1], quat_to_mat(q_b)[:, 0, 1])  # 本の y 軸（長い軸）の向き
    q_pick = r.down_quat_toward(long_yaw, p0[:, :2])
    top = p0[:, 2] + t / 2
    offset = PINCH_OFFSET if (grasp == "pinch" and hgt > PINCH_CENTER_MAX_H) else 0.0
    gy = quat_to_mat(q_pick)[:, :, 1]  # 左の指が開く向き（gripper の +y）
    p0 = p0 - offset * gy  # 手の中心を、本の中心から右の指の側へずらす（本は左の指の側に寄る）
    above = p0.copy()
    above[:, 2] = top + CUP_TIP_EXTENDED_Z + 0.10
    r.move_linear(above, q_pick, 150)
    touch = p0.copy()
    touch[:, 2] = top + CUP_TIP_EXTENDED_Z - 0.012  # PD のたわみを見込んで押し付ける（flat_pick と同じ）
    r.move_linear(touch, q_pick, 80, n_way=4)
    r.step(15)
    _, cup_touch = r.touching(book)
    r.suction.on[:] = True
    r.step(10)
    # suction: 吸盤を引いて本の面を閉じた指先（と爪）に押し当てる（吸盤 1 点だと本が振れて吸着が外れた）
    hold = HOLD_PLUNGER if grasp == "pinch" else SUCTION_HOLD_PLUNGER
    r.move_joints(PLUNGER, np.full(n, hold), 80)
    r.step(10)
    sealed_before_close = r.suction.sealed.copy()
    pb_before = book.get_pos().cpu().numpy()
    if grasp == "pinch":
        # 中央で掴むときは両方を閉じる。ずらして掴むときは、左の指を本の端で止め、右の指だけ閉じる
        left = np.full(n, hgt / 2 + offset if offset > 0 else 0.0)
        r.move_joints(FINGERS, np.stack([left, np.zeros(n)], -1), 60)
        r.step(20)
    sealed_after_close = r.suction.sealed.copy()
    shift_close = np.linalg.norm(book.get_pos().cpu().numpy() - pb_before, axis=1)
    lift = p0.copy()
    lift[:, 2] = floor + LIFT_DZ + t + CUP_TIP_RETRACTED_Z + hold
    r.move_linear(lift, q_pick, 100)
    r.step(20)
    pb = book.get_pos().cpu().numpy()
    f_touch, _ = r.touching(book)
    opening = r.finger_opening()
    if grasp == "pinch":
        held = (pb[:, 2] - p0[:, 2] > 0.6 * LIFT_DZ) & f_touch & (np.abs(opening - hgt) < 0.02)
    else:
        held = (pb[:, 2] - p0[:, 2] > 0.6 * LIFT_DZ) & r.suction.sealed
    lap("pick")
    snap("1 picked (fingers on head/foot)")

    # ---- 2. 立てる姿勢を決める（手の中の本の位置と向きは実測） ----
    g, gq = r.gripper_pose()
    rg = quat_to_mat(gq)
    rb = quat_to_mat(book.get_quat().cpu().numpy())
    off_local = np.einsum("nji,nj->ni", rg, pb - g)  # 本の中心（gripper 座標）
    r_bg = np.einsum("nji,njk->nik", rg, rb)  # 本の向き（gripper 座標）
    alpha = math.radians(args.tilt_deg - 90.0)
    a3 = np.array([tray.a[0], tray.a[1], 0.0])
    l3 = np.array([tray.u[0], tray.u[1], 0.0])
    s = np.sign(rb[:, :, 1] @ l3)  # 長い軸の向きは今のまま（± を変えると手首が 180° 回る）
    y_t = s[:, None] * l3
    z_t = np.tile(a3 * math.cos(alpha) + np.array([0.0, 0.0, 1.0]) * math.sin(alpha), (n, 1))
    x_t = np.cross(y_t, z_t)
    r_bt = np.stack([x_t, y_t, z_t], -1)  # 列 = 本の x, y, z（世界）
    r_gt = np.einsum("nij,nkj->nik", r_bt, r_bg)
    q_stand = mats_to_quats(r_gt)
    # 倒れた後の本の中心がトレーの中央（u=0）に来るように、立てる位置を外側へずらす
    u_c = w / 2 + t / 2
    loc = np.array([[sx * w / 2, sy * hgt / 2, sz * t / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    zmin = np.einsum("nij,kj->nki", r_bt, loc)[:, :, 2].min(axis=1)
    c_t = tray.at(np.full(n, u_c), np.zeros(n), floor + args.release_mm / 1000 - zmin)
    g_t = c_t - np.einsum("nij,nj->ni", r_gt, off_local)
    pre = g_t.copy()
    pre[:, 2] += PRE_DZ

    # ---- 3. 回しながら立てる位置の上へ → 下ろす ----
    # 手先の直線補間で 90° 以上回すと、途中で IK が別の枝へ飛んで腕が振り回された（試行4 の最初の実行）。
    # そこで立てる直前の姿勢を、今の関節角の手首を tilt ぶん回した角度を初期値にして IK で解き、
    # 関節空間で補間して動かす。初期値は何通りか試し、残差と関節の移動量の小さいものを環境ごとに選ぶ
    r.ik_err_max = np.zeros(n)
    q_pre, seed_sign, pre_ik_err = ik_seeded(r, pre, q_stand, math.radians(args.tilt_deg))
    r.ik_err_max = np.maximum(r.ik_err_max, pre_ik_err)
    r.move_joints(np.arange(7), q_pre, 220)
    r.step(20)
    r.move_linear(g_t, q_stand, 80, n_way=6)
    stand_err = r.correct(g_t, q_stand)
    # 実測した本の向きと目標の向きの差を、手の向きで 1 回だけ打ち消す（手の中で本が回った分・手首のたわみの分）。
    # 新書（ずらして掴む）では本が手の反対側へ約 10° 傾いて立ち、16 環境中 13 で逆向きに倒れた
    up_before_fix = quat_to_mat(book.get_quat().cpu().numpy())[:, 2, 2]
    if args.stand_fix:
        g_now, gq_now = r.gripper_pose()
        rg_now = quat_to_mat(gq_now)
        rb_now = quat_to_mat(book.get_quat().cpu().numpy())
        c_now = book.get_pos().cpu().numpy()
        d_rot = np.einsum("nij,nkj->nik", r_bt, rb_now)  # 目標 × 実測ᵀ（世界）
        rg_new = np.einsum("nij,njk->nik", d_rot, rg_now)
        off_now = np.einsum("nji,nj->ni", rg_now, c_now - g_now)
        g_new = c_t - np.einsum("nij,nj->ni", rg_new, off_now)
        q_new = mats_to_quats(rg_new)
        r.move_linear(g_new, q_new, 60, n_way=6)
        stand_err = r.correct(g_new, q_new, iters=1)
        q_stand = q_new
    ik_carry = r.ik_err_max.copy()
    pb2 = book.get_pos().cpu().numpy()
    g2, gq2 = r.gripper_pose()
    off2 = np.einsum("nji,nj->ni", quat_to_mat(gq2), pb2 - g2)
    held_rel = held & (np.linalg.norm(off2 - off_local, axis=1) < 0.015)
    gap = (book_corners(pb2, book.get_quat().cpu().numpy(), spec.size_m)[:, :, 2].min(axis=1) - floor)
    up_mid = quat_to_mat(book.get_quat().cpu().numpy())[:, 2, 2]
    lap("rotate_and_lower")
    snap("2 standing, before release")

    # ---- 4. 放して、手をすぐ真上へ逃がす ----
    sealed_at_release = r.suction.sealed.copy()
    r.suction.on[:] = False
    r.step(2)
    if grasp == "pinch":
        r.move_joints(FINGERS, np.full((n, 2), min(FINGER_OPEN, hgt / 2 + 0.02)), 12)
    r.move_joints(PLUNGER, np.zeros(n), 10)  # 吸盤を本の面から離す（本は手の側へ倒れるので先に引く）
    z_hi = floor + args.release_mm / 1000 + w + FINGERTIP_Z + 0.06
    up = r.gripper.get_pos().cpu().numpy()
    up[:, 2] = z_hi
    r.move_linear(up, q_stand, 50, n_way=5)
    r.step(30)
    up_after_release = quat_to_mat(book.get_quat().cpu().numpy())[:, 2, 2]
    lap("release_and_clear")
    snap("3 released, hand up")

    # ---- 5. 立ったまま残った本を、向こう側から手前へ押して倒す ----
    push_contact = np.zeros(n, bool)
    push_ik_err = np.zeros(n)
    if args.push:
        r.move_joints(FINGERS, np.zeros((n, 2)), 30)
        start_u = u_c + t / 2 + 0.035
        tip_z = floor + PUSH_H_RATIO * w
        q_push = r.down_quat_toward(np.full(n, tray.a_yaw), tray.at(np.full(n, start_u), np.zeros(n), 0.0)[:, :2])
        hi = tray.at(np.full(n, start_u), np.zeros(n), np.full(n, z_hi))
        q_hi, _, hi_err = ik_seeded(r, hi, q_push, math.radians(args.tilt_deg))  # 下向きへ戻すのも関節空間で
        push_ik_err = hi_err
        r.move_joints(np.arange(7), q_hi, 150)
        low = tray.at(np.full(n, start_u), np.zeros(n), np.full(n, tip_z + FINGERTIP_Z + LIP_T))
        r.move_linear(low, q_push, 80, n_way=6)
        r.correct(low, q_push, iters=1)
        snap("4 push start")
        end = tray.at(np.full(n, u_c - t / 2 - 0.01), np.zeros(n), np.full(n, tip_z + FINGERTIP_Z + LIP_T))
        probe.watch_book, probe.touched[:] = book, False
        r.move_linear(end, q_push, 80, n_way=8)
        probe.watch_book = None
        push_contact = probe.touched.copy()  # False なら、押す前に倒れていた
        back = end.copy()
        back[:, 2] = z_hi
        r.move_linear(back, q_push, 60, n_way=6)
        lap("push")
        snap("5 after push")
    return {"held": held, "cup_touch_at_pick": cup_touch, "sealed_before_close": sealed_before_close,
            "sealed_after_close": sealed_after_close, "book_shift_on_close_mm": shift_close * 1000,
            "raised_by_suction_mm": (pb_before[:, 2] - p0[:, 2]) * 1000, "grasp": grasp, "pinch_offset_m": offset,
            "sealed_at_release": sealed_at_release, "suction_sealed": r.suction.ever_sealed.copy(),
            "opening_mm": opening * 1000, "held_at_release": held_rel, "release_gap_mm": gap * 1000,
            "up_z_standing": up_mid, "up_z_before_fix": up_before_fix, "up_z_after_release": up_after_release, "push_contact": push_contact,
            "ik_err_carry": ik_carry, "stand_pos_err": stand_err, "wrist_seed_sign": seed_sign,
            "pre_ik_err_mm": pre_ik_err * 1000, "push_ik_err_mm": push_ik_err * 1000}


def ik_seeded(r: Robot, pos: np.ndarray, quat: np.ndarray, rot: float):
    """joint5・6・7 のどれかを ±rot 回した関節角（と今の関節角）を初期値に IK を解き、環境ごとに良いほう（位置と向きの残差＋関節の移動量）を返す。"""
    q_now = r.e.get_qpos().cpu().numpy()
    best_q, best_cost, best_sign, best_err = None, None, None, None
    seeds = [(j, sg) for j in (4, 5, 6) for sg in (+1.0, -1.0)] + [(0, 0.0)]
    for j, sign in seeds:
        seed = q_now.copy()
        seed[:, j] += sign * rot
        q, err = r.e.inverse_kinematics(link=r.gripper, pos=pos, quat=quat, dofs_idx_local=np.arange(7),
                                        init_qpos=seed, return_error=True)
        q = q.cpu().numpy()[:, :7]
        e = err.cpu().numpy().reshape(r.n, -1)
        pos_err = np.linalg.norm(e[:, :3], axis=1)
        cost = pos_err + 0.05 * np.linalg.norm(e[:, 3:], axis=1) + 0.002 * np.abs(q - q_now[:, :7]).sum(axis=1)
        if best_q is None:
            best_q, best_cost, best_err = q, cost, pos_err
            best_sign = np.full(r.n, (j + 1) * sign)
        else:
            take = cost < best_cost
            best_q = np.where(take[:, None], q, best_q)
            best_err = np.where(take, pos_err, best_err)
            best_sign = np.where(take, (j + 1) * sign, best_sign)
            best_cost = np.minimum(cost, best_cost)
    return best_q, best_sign, best_err


def scan(args, cams, out: Path, tag: str, bus: Bus, sessions, book, spec, tray: Tray, up_z) -> dict:
    """真上のカメラで読む（環境 0）。ほかの環境は正解（表紙が上なら見える）を使い、そのことを記録する。"""
    res = {"isbn_env0": None}
    n = len(sessions)
    for i in range(n):
        if i == 0 and cams:
            rgb, *_ = cams["top"].render()
            rgb = np.asarray(rgb)
            Image.fromarray(rgb).save(out / f"topcam_{tag}_env0.png")
            isbn, yaw_img = read_isbn_with_yaw(rgb)
            sessions[0].on_scan(isbn, yaw_img, source="camera")
            info = {"isbn": isbn, "correct": isbn == spec.isbn13, "yaw_img_deg": yaw_img}
            if isbn is not None:
                yaw_world = image_yaw_to_world(yaw_img, math.degrees(tray.u_yaw))
                truth = math.degrees(yaw_of(book.get_quat().cpu().numpy())[0])  # 本の x 軸（読む向き）
                info.update(yaw_world_deg=yaw_world, yaw_truth_deg=truth, yaw_err_deg=wrap_deg(yaw_world - truth))
            res["env0"] = info
            res["isbn_env0"] = isbn
        else:
            visible = bool(up_z[i] > 0.9)
            truth_yaw = math.degrees(yaw_of(book.get_quat().cpu().numpy())[i]) if visible else None
            # 画像の角度へ戻して渡す（カメラの上 = u）
            yaw_img = None if truth_yaw is None else wrap_deg(truth_yaw - math.degrees(tray.u_yaw) + 90.0)
            sessions[i].on_scan(spec.isbn13 if visible else None, yaw_img, source="ground_truth")
    return res


def montage(frames, path: Path) -> None:
    hs = [f for _, f in frames]
    h, w = hs[0].shape[:2]
    cols = 3
    rows = math.ceil(len(frames) / cols)
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for k, (label, f) in enumerate(frames):
        x, y = (k % cols) * w, (k // cols) * h
        canvas.paste(Image.fromarray(f), (x, y))
        d.rectangle([x, y, x + 8 * len(label) + 12, y + 22], fill=(255, 255, 255))
        d.text((x + 6, y + 5), label, fill=(0, 0, 0))
    canvas.save(path)


if __name__ == "__main__":
    main()
