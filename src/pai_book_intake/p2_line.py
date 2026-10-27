"""買取ラインを一通り回す（P2・スクリプト制御）。全部のロボットを配置し、箱から本棚・廃棄箱までを通す。

2026-10-20 のユーザー指示: 当初想定していたロボットをすべて配置し、精度が悪くても作業を一通り回せるようにする。
本棚は遠くにあり（位置は assets/site/site_v1.json に保管）、ロボット B が台車で移動し、空きを認識して、価値のある本を
置いて記録するところまで。

登場するもの（配置は site_v1.json）
  ロボット A（固定）   箱の平積みの一番上から、背の側で挟む手（試行8 の P10・板 ii・爪 中心+10 mm）で取り、査定台に置く
  査定台＋真上のカメラ  仕切りの無い台。置き場所（スロット）はソフトで決める。カメラは台全体の真上から、置かれた本へ向きと
                        画角を合わせて（パン・チルト・ズーム）バーコードを読む（B1）
  査定（ソフト）       ISBN → カタログ → 本棚か廃棄か（appraisal.py）。運搬許可（release.py）
  ロボット B（台車）    許可の出た本を台から取り、廃棄箱なら底に平らに置く。本棚なら台車で本棚の前へ移動し、
                        台車のカメラ（深さ）で空きを見つけ、本を置いて記録する。置き方は --shelf-mode で選ぶ:
                        flat（既定）= 背を手前に平らに置き、床に空きが無ければ積む（02_確認事項 B8 の回答 c。締切を優先）
                        standing = 立てて小口から差し込む（B4。この手では指が止まって入らない。B8 の a・b で直す）
  本棚                パタパタ・ブックキーパー v2 を横に長くしたもの（keeper.py）
  オブザーバ          全メッセージを見て、止める・箱ごとの報告書を書く

1 冊ずつ順に処理する（A と B の並行はまだ。査定台の 2 スロットは使い回す）。

まだ入っていないもの（結果の not_implemented に書く）
  - 裏返し（B3: 空中で回す）。バーコードが見えない本は、人の判断待ち（tray.held）として台に残す
  - 箱の中の縦置きの本（B2）。今の箱は平積みだけ
  - A と B の並行動作

描画は環境 0 だけ（WSL2 の Docker は CPU 描画）。環境 0 以外は、バーコードは本の向き（表紙が上か）から、本棚の空きは
置いた本の位置から決める（どちらも結果に source として残す）。

使い方: python -m pai_book_intake.p2_line --n-envs 1 --books 4
出力: /workspace/outputs/p2_line/<run_id>/（result.json・bus.jsonl・報告書・inventory.sqlite・動画・バーコード画像・
      本棚の点群から見た空き）
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .appraisal import appraise
from .book_assets import make_book_assets
from .books import DEFAULT_CATALOG, SURFACES, load_catalog
from .bus import Bus
from .geometry import quat_to_mat, quat_z
from .hand_design import ENV_BIT, PinchHand
from .inventory import Inventory, Placement
from .keeper import (BASE_T, KeeperSpec, choose_flat_slot, choose_slot, choose_stack, flat_book_center_y, free_spans,
                     keeper_mjcf, occupancy_from_points, spans_from_books)
from .observer import Observer, ObserverConfig
from .pinch import (PinchRobot, book_frame, build_pinch_robot_xml, g_cart_chain, g_drive, g_home, g_move_book,
                    g_move_book_joint, g_pick, g_place_flat, g_shelve, lip_side_quat, min_jerk, run_generators,
                    shelve_grip_offset, standing_quat, touching)
from .release import ReleaseGate
from .runlog import classify, new_run, save_result
from .site import DEFAULT_SITE, load_site
from .tool import World
from .vision import decode_isbn

TRIAL = "p2_line"
DT = 0.01
TABLE_FRICTION = 0.3  # 試行8 と同じ（既定 1.0 だと本の表面の違いが消える）
CARD = (0.72, 0.56, 0.38)
GRAY = (0.80, 0.80, 0.78)
BIN_COLOR = (0.35, 0.37, 0.40)
WOOD = (0.62, 0.45, 0.28)
HAND = PinchHand(name="P10", rubber_mu=1.0, note="試行8 の基準（ゴム μ1.0）")
PLATE_MODE, CLAW_MODE = "ii", "center"  # 試行8: 支えの中心が重心から最も近い（6 mm）
DEFAULT_PILE = ("book-09", "book-04", "book-10", "book-03")  # 下から上。上から: 文庫（棚）・B5（棚）・文庫（廃棄）・
# A5 の重い本（廃棄）
NOT_IMPLEMENTED = ["flip (B3)", "standing books in the box (B2)", "A and B in parallel",
                   "standing books on the shelf (keeper fingers jam; 02_確認事項 B8 a/b). The default shelf mode is flat (B8 c)"]
SHELF_MODES = ("flat", "standing")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--books", type=int, default=4, help="平積みの上から何冊処理するか")
    ap.add_argument("--pile", default=",".join(DEFAULT_PILE), help="箱の平積み（下から上の book_id）")
    ap.add_argument("--face-down", default="", help="裏向き（バーコードが下）に入れる book_id（カンマ区切り）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--site", type=Path, default=DEFAULT_SITE)
    ap.add_argument("--video", action="store_true", help="全体と本棚の動画を撮る（環境 0）")
    ap.add_argument("--barcode-res", type=int, nargs=2, default=[1920, 1440])
    ap.add_argument("--shelf-mode", choices=SHELF_MODES, default="flat",
                    help="本棚への置き方。flat = 平らに置いて積む（B8 c・既定）、standing = 立てて差し込む（B4）")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import genesis as gs

    run_id, out, meta = new_run(TRIAL, {k: (str(v.name) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                extra={"genesis": gs.__version__})
    bus = Bus(run_id, out / "bus.jsonl")
    observer = Observer(bus, ObserverConfig(max_consecutive_failures=3, barcode_unreadable_limit=3, shelf_capacity=20))
    ctx: dict = {"recording": []}
    result: dict = {}
    error = None
    try:
        result = run(args, run_id, out, bus, observer, ctx)
    except BaseException as e:
        error = e
        result = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        raise
    finally:
        for cam in ctx["recording"]:
            cam.stop_recording()
        result["reports"] = [p.name for p in observer.write_reports(out, title=f"{TRIAL} {run_id}")]
        bus.close()
        result["status"] = classify(bool(result.get("acceptance_ok", False)), error)
        save_result(out, meta, result)
        print(f"run_id={run_id} status={result['status']} out={out}")


# ---------- シーン ----------

def _box(scene, size, pos, color, yaw=0.0):
    import genesis as gs

    return scene.add_entity(gs.morphs.Box(size=tuple(float(v) for v in size), pos=tuple(float(v) for v in pos),
                                          euler=(0.0, 0.0, math.degrees(yaw)), fixed=True,
                                          contype=ENV_BIT, conaffinity=ENV_BIT),
                            material=gs.materials.Rigid(friction=TABLE_FRICTION),
                            surface=gs.surfaces.Default(color=color))


def _open_box(scene, center, inner, floor, wall_h, wall_t, color):
    """上が開いた箱。floor は床板の上面の高さ。"""
    cx, cy = center
    ix, iy = inner
    _box(scene, (ix + 2 * wall_t, iy + 2 * wall_t, floor), (cx, cy, floor / 2), color)
    for size, off in [((wall_t, iy + 2 * wall_t, wall_h), (-ix / 2 - wall_t / 2, 0)),
                      ((wall_t, iy + 2 * wall_t, wall_h), (ix / 2 + wall_t / 2, 0)),
                      ((ix, wall_t, wall_h), (0, -iy / 2 - wall_t / 2)),
                      ((ix, wall_t, wall_h), (0, iy / 2 + wall_t / 2))]:
        _box(scene, size, (cx + off[0], cy + off[1], wall_h / 2), color)


def build(args, out: Path, site, pile_specs):
    import genesis as gs

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=4),
                     rigid_options=gs.options.RigidOptions(enable_self_collision=False,
                                                           friction_cone=gs.friction_cone.elliptic,
                                                           constraint_timeconst=0.01),
                     vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane(contype=ENV_BIT, conaffinity=ENV_BIT))
    bx = site["box"]
    _open_box(scene, bx["center"], bx["inner"], bx["floor"], bx["wall_h"], bx["wall_t"], CARD)
    tb = site["appraisal_table"]
    _box(scene, (*tb["size"], tb["top"]), (*tb["center"], tb["top"] / 2), GRAY)
    bn = site["bin"]
    _open_box(scene, bn["center"], bn["inner"], bn["floor"], bn["wall_h"], bn["wall_t"], BIN_COLOR)
    shelf = site.shelf()
    keeper = KeeperSpec(width=shelf.width, depth=shelf.depth, n_fingers=shelf.n_fingers, pitch=shelf.pitch)
    stand_c = shelf.to_world([shelf.width / 2, shelf.depth / 2, 0.0])
    _box(scene, (shelf.width + 0.06, shelf.depth + 0.04, shelf.stand_h), (stand_c[0], stand_c[1], shelf.stand_h / 2), WOOD,
         yaw=shelf.yaw)
    kxml = out / "keeper.xml"
    kxml.write_text(keeper_mjcf(keeper, ENV_BIT))
    o = shelf.to_world([0.0, 0.0, 0.0])
    keeper_ent = scene.add_entity(gs.morphs.MJCF(file=str(kxml), pos=tuple(float(v) for v in o),
                                                 euler=(0.0, 0.0, math.degrees(shelf.yaw))))
    books = []
    for i, spec in enumerate(pile_specs):
        obj, tex = make_book_assets(spec, out)
        s = SURFACES[spec.surface]
        books.append(scene.add_entity(
            gs.morphs.Mesh(file=str(obj), pos=(bx["center"][0], bx["center"][1], 0.5 + 0.06 * i)),
            material=gs.materials.Rigid(rho=spec.density, friction=s["friction"]),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(tex)))))
    ra = site["robot_a"]
    ent_a = scene.add_entity(gs.morphs.MJCF(file=str(build_pinch_robot_xml(out, HAND)), pos=(*ra["base"], 0.0),
                                            euler=(0.0, 0.0, math.degrees(ra["yaw"]))))
    ent_b = scene.add_entity(gs.morphs.MJCF(file=str(build_pinch_robot_xml(out, HAND, mobile=True)), pos=(0.0, 0.0, 0.0)))
    tc = tb["center"]
    cams = {
        "barcode": scene.add_camera(res=tuple(args.barcode_res), pos=(tc[0], tc[1], tb["top"] + tb["camera_height"]),
                                    lookat=(tc[0], tc[1], tb["top"]), up=(1.0, 0.0, 0.0), fov=14, env_idx=0),
        # 台車の深さカメラ。640×240・縦 60° では棚の幅 400 mm が約 170 画素（2.3 mm/画素）しかなく、2 mm の区間に点が入らず
        # 「見えない＝埋まっている」になって空きが細切れになった（20261020T015615）。横 1280・縦 30° で約 1 mm/画素
        "shelf_depth": scene.add_camera(res=(1280, 240), pos=(0.0, 0.0, 1.0), lookat=(0.0, 1.0, 1.0), fov=30, env_idx=0),
    }
    sc = shelf.to_world([shelf.width / 2, 0.0, 0.15])
    cams["shelf_close"] = scene.add_camera(res=(800, 600), pos=(float(sc[0]) + 0.75, float(sc[1]) - 0.55, float(sc[2]) + 0.45),
                                           lookat=tuple(float(v) for v in sc), fov=50, env_idx=0)
    if args.video:
        cams["overview"] = scene.add_camera(res=(960, 540), pos=(1.9, -0.9, 1.7), lookat=(0.3, 0.6, 0.2), fov=50, env_idx=0)
        sc = shelf.to_world([shelf.width / 2, shelf.depth / 2, 0.1])
        cams["shelf_view"] = scene.add_camera(res=(960, 540), pos=(float(sc[0]) + 1.0, float(sc[1]) - 1.3, 1.1),
                                              lookat=tuple(float(v) for v in sc), fov=45, env_idx=0)
    scene.build(n_envs=args.n_envs)
    world = World(scene, args.n_envs, DT)
    rob_a = PinchRobot(world, ent_a, "robot-a-pick", HAND, base_xy=tuple(ra["base"]))
    rob_b = PinchRobot(world, ent_b, "robot-b-sort", HAND, base_pose=tuple(site["robot_b"]["station"]))
    rob_a.setup()
    rob_b.setup()
    keeper_ent.set_dofs_position(np.full((args.n_envs, keeper.n_fingers), math.radians(-35.0)),
                                 np.arange(keeper.n_fingers))
    return scene, world, rob_a, rob_b, books, keeper_ent, keeper, cams


def stack_books(books, specs, site, n, rng) -> None:
    bx = site["box"]
    z = np.full(n, bx["floor"])
    for b, spec in zip(books, specs):
        t = spec.thickness_m
        zc = z + t / 2 + 0.002
        pos = np.stack([bx["center"][0] + rng.uniform(-0.01, 0.01, n), bx["center"][1] + rng.uniform(-0.01, 0.01, n), zc], -1)
        b.set_pos(pos)
        b.set_quat(quat_z(rng.uniform(-math.radians(5), math.radians(5), n)))
        z = zc + t / 2


# ---------- 1 冊ぶん ----------

def run(args, run_id: str, out: Path, bus: Bus, obs: Observer, ctx: dict) -> dict:
    import genesis as gs

    site = load_site(args.site)
    catalog = load_catalog(DEFAULT_CATALOG)
    pile_ids = [s for s in args.pile.split(",") if s]
    face_down = {s for s in args.face_down.split(",") if s}
    pile = [catalog.by_id(b) for b in pile_ids]
    if args.books > len(pile):
        raise ValueError(f"--books は平積みの冊数（{len(pile)}）以下")
    n = args.n_envs
    rng = np.random.default_rng(args.seed)
    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    t_build = time.time()
    scene, world, ra, rb, books, keeper_ent, keeper, cams = build(args, out, site, pile)
    build_s = time.time() - t_build
    bus.set_sim_clock(lambda: world.sim_time)
    stack_books(books, pile, site, n, rng)
    for b, spec in zip(books, pile):
        if spec.book_id in face_down:  # 裏返して入れる（x 軸まわり 180°）
            q = b.get_quat().cpu().numpy()
            from .geometry import quat_mul
            b.set_quat(quat_mul(q, np.broadcast_to(np.array([0.0, 1.0, 0.0, 0.0]), (n, 4))))
    world.step(100)
    shelf = site.shelf()
    tb = site["appraisal_table"]
    bn = site["bin"]
    station = np.array(site["robot_b"]["station"])
    rbs = site["robot_b"]
    inv = Inventory(out / "inventory.sqlite")
    gates = [ReleaseGate(bus, catalog, n_slots=len(tb["slots"]), shelf_capacity=20, env=i) for i in range(n)]
    bus.publish("observer", "*", "note", "run.start",
                {"n_envs": n, "pile_bottom_to_top": pile_ids, "face_down": sorted(face_down), "site": site.site_id,
                 "shelf": {"shelf_id": shelf.shelf_id, "origin": list(shelf.origin), "yaw": shelf.yaw},
                 "hand": HAND.params(), "plate_mode": PLATE_MODE, "claw_mode": CLAW_MODE, "shelf_mode": args.shelf_mode,
                 "not_implemented": NOT_IMPLEMENTED})
    for name in ("overview", "shelf_view"):
        if name in cams:
            cams[name].start_recording(save_to_filename=str(out / f"{name}_env0.mp4"), fps=25)
            ctx["recording"].append(cams[name])
    t0 = time.time()
    per_book = []
    for k in range(args.books):
        idx = len(pile) - 1 - k
        rec = process_book(k, idx, books, pile, site, shelf, keeper, keeper_ent, tb, bn, station, rbs, ra, rb, world,
                           cams, gates, bus, obs, inv, run_id, out, catalog, args.shelf_mode)
        per_book.append(rec)
    wall = time.time() - t0
    for name in ("overview", "shelf_view"):
        if name in cams:
            cams[name].stop_recording()
    ctx["recording"].clear()
    for i, g in enumerate(gates):
        for it in g.remaining():
            bus.publish("appraisal", "observer", "event", "tray.remaining",
                        {"env": i, "book_id": it.book_id, "state": it.state, "reason": it.hold_reason})
    shelf_rows = inv.on_shelf(shelf.shelf_id)
    inv.close()
    n_ok = [sum(1 for b in obs.box(i).books if b["outcome"] == "succeeded") for i in range(n)]
    expected = [sum(1 for r in per_book if r["expected_flow"][i]) for i in range(n)]
    shelved_ok = [sum(1 for r in shelf_rows if r["env"] == i and r["inside"] and (r["upright"] if r["pose"] == "standing"
                                                                                 else r["spine_front"] and r["tilt_deg"] < 10.0))
                  for i in range(n)]
    acceptance_ok = all(n_ok[i] == expected[i] for i in range(n))
    return {
        "criterion": "p2_line_end_to_end",
        "acceptance": "全環境で、処理した本のうち、バーコードが読める向きの本は全部、A が取り出して台に置き、B が行き先へ届けた"
                      "（本棚は棚の中で、flat なら背が手前で平ら・standing なら立っている。廃棄箱は箱の中で平ら）。"
                      "棚に置いた本は inventory.sqlite に記録した",
        "shelf_mode": args.shelf_mode,
        "acceptance_ok": acceptance_ok,
        "books_succeeded_per_env": n_ok, "books_expected_per_env": expected, "shelved_ok_per_env": shelved_ok,
        "inventory": shelf_rows,
        "not_implemented": NOT_IMPLEMENTED,
        "stop_reasons": [obs.box(i).stop_reason for i in range(n)],
        "sim_time_s": world.sim_time, "wall_build_s": build_s, "wall_run_s": wall,
        "site": site.raw, "per_book": per_book,
    }


def process_book(k, idx, books, pile, site, shelf, keeper, keeper_ent, tb, bn, station, rbs, ra, rb, world, cams, gates,
                 bus, obs, inv, run_id, out, catalog, shelf_mode: str = "flat") -> dict:
    n = world.n
    book, spec = books[idx], pile[idx]
    rec: dict = {"book_id": spec.book_id, "title": spec.title}
    alive = np.array([not obs.is_stopped(i) for i in range(n)])
    rec["expected_flow"] = alive.copy()

    # ---- A: 箱から取り、査定台に置く ----
    slot = [gates[i].free_slot() for i in range(n)]
    if any(s is None for s in slot):
        bus.publish("robot-a-pick", "observer", "event", "line.waiting", {"reason": "tray_full"})
        rec["skipped"] = "tray_full"
        rec["expected_flow"] = np.zeros(n, bool)
        return rec
    lower = [(books[j], pile[j]) for j in range(idx)]
    bx = site["box"]
    o: dict = {}
    bus.publish("robot-a-pick", "observer", "event", "pick.start", {"book_id": spec.book_id})
    run_generators(world, [g_pick(ra, book, spec, o, lower=lower, surface_z=None if lower else bx["floor"],
                                  carry_bottom_z=bx["wall_h"] + 0.04, plate_mode=PLATE_MODE, claw_mode=CLAW_MODE)])
    grip_a = o["grip"]
    slot_xy = np.array([tb["slots"][s] for s in slot])
    run_generators(world, [g_move_book(ra, grip_a, slot_xy)])
    run_generators(world, [g_place_flat(ra, grip_a, tb["top"], o)])
    run_generators(world, [g_home(ra, 120)])
    pb, mb = book_frame(book)
    w, h, t = spec.size_m
    on_table = (np.abs(pb[:, 2] - (tb["top"] + t / 2)) < 0.006) & (np.abs(mb[:, 2, 2]) > 0.97)
    rec["a"] = {"pick": grip_a.rec, "place": o.get("place", {}), "on_table": on_table, "slot": slot,
                "pos_err_from_slot_mm": np.linalg.norm(pb[:, :2] - slot_xy, axis=1) * 1000}
    for i in range(n):
        if not alive[i]:
            continue
        bus.publish("robot-a-pick", "inspect-scan", "event", "place.done",
                    {"env": i, "book_id": spec.book_id, "slot": int(slot[i]), "on_table": bool(on_table[i])})
        if on_table[i]:
            gates[i].place(spec.book_id)
        else:
            _result(bus, "robot-a-pick", i, spec, None, "rejected", None, 0, "not_on_table")
            alive[i] = False

    # ---- 点検カメラ: 本へ向けてバーコードを読む ----
    isbn = [None] * n
    up_z = mb[:, 2, 2]
    rec["expected_flow"] &= up_z > 0  # 裏向きの本は裏返し（未実装）が要るので、届かなくても失敗に数えない
    if alive[0]:
        cam = cams["barcode"]
        view = max(w, h) + 0.03
        cam.set_pose(pos=(float(pb[0, 0]), float(pb[0, 1]), tb["top"] + tb["camera_height"]),
                     lookat=(float(pb[0, 0]), float(pb[0, 1]), tb["top"]), up=(1.0, 0.0, 0.0))
        # 画角は本が収まるだけ（ズーム）。Genesis のカメラは作った後に画角を変えられないので、作るときの 14° を使う
        rgb, *_ = cam.render()
        rgb = np.asarray(rgb)
        Image.fromarray(rgb).save(out / f"barcode_{spec.book_id}_env0.png")
        isbn[0] = decode_isbn(rgb)
        rec["barcode_view_m"] = view
        bus.publish("inspect-scan", "appraisal", "event", "barcode.read",
                    {"env": 0, "book_id": spec.book_id, "ok": isbn[0] is not None, "isbn": isbn[0],
                     "correct": isbn[0] == spec.isbn13, "source": "camera"})
    for i in range(1, n):
        if alive[i]:
            isbn[i] = spec.isbn13 if up_z[i] > 0 else None
            bus.publish("inspect-scan", "appraisal", "event", "barcode.read",
                        {"env": i, "book_id": spec.book_id, "ok": isbn[i] is not None, "isbn": isbn[i],
                         "source": "ground_truth"})
    rec["isbn_read"] = isbn
    decisions = [None] * n
    for i in range(n):
        if not alive[i]:
            continue
        gates[i].on_scan(spec.book_id, isbn[i])
        if isbn[i] is None:
            # 裏返し（B3）はまだ。人の判断待ちとして台に残す
            bus.publish("inspect-scan", "observer", "event", "escalation",
                        {"env": i, "book_id": spec.book_id, "reason": "barcode_unreadable", "flip": "not_implemented"})
            gates[i].on_escalated(spec.book_id, "barcode_unreadable")
            alive[i] = False
            continue
        d = decisions[i] = appraise(isbn[i], catalog)
        bus.publish("appraisal", "robot-b-sort", "event", "decision",
                    {"env": i, "book_id": spec.book_id, "destination": d.destination, "value_jpy": d.value_jpy,
                     "reason": d.reason})
        gates[i].on_appraisal(spec.book_id, d)
        if d.destination == "shelf" and shelf_mode == "standing" and shelve_grip_offset(spec, HAND, keeper) is None:
            # この手では、この高さの本を本棚の床まで下ろせない（手のひらと壁の指が本棚の後ろの壁・指の軸に当たる）。
            # B は取らずに台に残し、人の判断を待つ（Obsidian 02_確認事項 B8 で相談）
            bus.publish("appraisal", "observer", "event", "escalation",
                        {"env": i, "book_id": spec.book_id, "reason": "shelf_geometry", "height_mm": round(h * 1000)})
            gates[i].on_escalated(spec.book_id, "shelf_geometry")
            rec["expected_flow"][i] = False

    # ---- B: 許可を確かめて取る ----
    cleared = np.zeros(n, bool)
    dest = [None] * n
    for i in range(n):
        if not alive[i]:
            continue
        rel = {it.book_id: it for it in gates[i].pickable()}
        if spec.book_id in rel:
            dest[i] = gates[i].take(spec.book_id, rel[spec.book_id].clearance_id)
            cleared[i] = True
    rec["cleared"] = cleared
    rec["destination"] = dest
    if not cleared.any():
        return rec
    rb.freeze(~cleared)
    o_b: dict = {}
    standing = shelf_mode == "standing" and any(dd == "shelf" for dd in dest)
    off = shelve_grip_offset(spec, HAND, keeper) if standing else 0.0
    rec["b_grip_off_mm"] = None if off is None else off * 1000
    run_generators(world, [g_pick(rb, book, spec, o_b, surface_z=tb["top"], carry_bottom_z=tb["top"] + 0.10,
                                  plate_mode=PLATE_MODE, claw_mode=CLAW_MODE, grip_off=off or 0.0)])
    grip_b = o_b["grip"]
    rec["b_pick"] = grip_b.rec
    delivered = np.zeros(n, bool)
    to_shelf = np.array([d == "shelf" for d in dest]) & cleared
    to_bin = np.array([d == "discard" for d in dest]) & cleared
    if to_bin.any():
        rb.unfreeze()
        rb.freeze(~to_bin)
        # 関節空間で運ぶ（手先の直線で運ぶと、台から廃棄箱へ向きを変える途中で本を落とした。20261023T010753）
        mv: dict = {}
        run_generators(world, [g_move_book_joint(rb, grip_b, np.tile(bn["center"], (n, 1)), grip_b.quat,
                                                 bn["wall_h"] + 0.05, out=mv)])
        # 廃棄箱の本は積み重なる。置く面は、これまでに入れた本の上面の最も高い所（実測）
        surf = np.array([BIN_TOP.get(i, bn["floor"]) for i in range(n)])
        run_generators(world, [g_place_flat(rb, grip_b, surf, o_b)])
        run_generators(world, [g_home(rb, 120)])
        pb, mb = book_frame(book)
        in_bin = ((np.abs(pb[:, 0] - bn["center"][0]) < bn["inner"][0] / 2) & (np.abs(pb[:, 1] - bn["center"][1]) < bn["inner"][1] / 2)
                  & (pb[:, 2] < surf + t / 2 + 0.006) & (np.abs(mb[:, 2, 2]) > 0.97))
        delivered |= to_bin & in_bin
        for i in np.nonzero(to_bin & in_bin)[0]:
            BIN_TOP[int(i)] = max(BIN_TOP.get(int(i), bn["floor"]), float(pb[i, 2] + t / 2))
        rec["bin"] = {"move": mv, "place": o_b.get("place", {}), "surface_z": surf, "in_bin_flat": in_bin}
    if to_shelf.any():
        rb.unfreeze()
        rb.freeze(~to_shelf)
        place = yield_shelf if shelf_mode == "standing" else yield_shelf_flat
        rec["shelf"] = place(rb, grip_b, spec, book, shelf, keeper, keeper_ent, station, rbs, world, cams, bus, inv,
                             run_id, out, to_shelf, dest, isbn, decisions, k)
        delivered |= to_shelf & rec["shelf"]["ok"]
    rb.unfreeze()
    for i in range(n):
        if not cleared[i]:
            continue
        gates[i].on_delivered(spec.book_id, bool(delivered[i]))
        d = decisions[i]
        _result(bus, "robot-b-sort", i, spec, isbn[i], "succeeded" if delivered[i] else "rejected", dest[i],
                d.value_jpy, d.reason if delivered[i] else f"not_delivered_to_{dest[i]}")
    rec["delivered"] = delivered
    return rec


def scan_shelf(rb, shelf, keeper, cams, books_on_shelf_x: list[list[tuple[float, float]]], out: Path, tag: str,
               **occ_kw) -> list:
    """台車のカメラで本棚の空きを見る（環境 0 は深さカメラの点群、ほかは置いた本の位置から）。環境ごとに (spans, source)。"""
    n = rb.n
    res = []
    bp = rb.base_pose()
    cam = cams["shelf_depth"]
    yaw = bp[0, 2]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    pos = np.array([bp[0, 0], bp[0, 1], shelf.stand_h + shelf.cam_height]) + fwd * shelf.cam_forward
    cam.set_pose(pos=tuple(float(v) for v in pos), lookat=tuple(float(v) for v in pos + fwd), up=(0.0, 0.0, 1.0))
    pc, mask = cam.render_pointcloud(world_frame=True)
    pts = np.asarray(pc).reshape(-1, 3)
    m = np.asarray(mask).reshape(-1).astype(bool)
    pts_s = shelf.to_shelf(pts[m])
    left, occ = occupancy_from_points(pts_s, keeper, **occ_kw)
    spans0 = free_spans(left, occ, 0.002, keeper.usable_x)
    np.savez_compressed(out / f"shelf_scan_{tag}_env0.npz", left=left, occupied=occ)
    for i in range(n):
        if i == 0:
            res.append((spans0, "depth_camera"))
        else:
            res.append((spans_from_books(books_on_shelf_x[i], keeper), "ground_truth"))
    return res


def _drive_to_shelf(rb, grip, book, shelf, rbs, world, bus, rec: dict) -> None:
    """本を台車の横の高い所へ寄せて、台車で本棚の正面へ行く。"""
    n = world.n
    # 台車で運ぶ姿勢: 本を台車の横の高い所へ寄せる。前に寄せると、手と本が台車の深さカメラの前をふさぎ、
    # 空の棚の 4 割が「埋まっている」に見えた（20261020T020301）
    bp = rb.base_pose()
    fwd = np.stack([np.cos(bp[:, 2]), np.sin(bp[:, 2]), np.zeros(n)], -1)
    left = np.stack([-np.sin(bp[:, 2]), np.cos(bp[:, 2]), np.zeros(n)], -1)
    tuck = np.concatenate([bp[:, :2], np.zeros((n, 1))], -1) + fwd * 0.25 + left * 0.30
    run_generators(world, [g_move_book(rb, grip, tuck, bottom_z=shelf.stand_h + 0.20)])
    front = shelf.front_pose()
    bus.publish("robot-b-sort", "observer", "event", "base.drive", {"to": "shelf", "shelf_id": shelf.shelf_id, "pose": front})
    run_generators(world, [g_drive(rb, front, rbs["max_speed"], rbs["max_yaw_rate"], DT)])
    bp = rb.base_pose()
    rec["base_err_m"] = np.linalg.norm(bp[:, :2] - np.array(front[:2]), axis=1)
    rec["still_held_after_drive"] = touching(rb, book)


def _drive_to_station(rb, station, rbs, world) -> None:
    """本棚から待機位置へ戻る。待機位置の手前の点（station_approach）を通る（まっすぐ戻ると廃棄箱に当たることがある）。"""
    ap = rbs.get("station_approach")
    if ap is not None:
        run_generators(world, [g_drive(rb, (ap[0], ap[1], float(station[2])), rbs["max_speed"], rbs["max_yaw_rate"], DT)])
    run_generators(world, [g_drive(rb, tuple(station), rbs["max_speed"], rbs["max_yaw_rate"], DT)])


BIN_TOP: dict = {}  # 環境ごとの、廃棄箱に入れた本の上面の最も高い所（世界の z。置いた後に測った値）
SHELF_BOOKS_X: dict = {}  # 環境ごとの、棚に置いた本の x の範囲（深さカメラを描かない環境の空きを出すため）


def yield_shelf(rb, grip, spec, book, shelf, keeper, keeper_ent, station, rbs, world, cams, bus, inv, run_id, out, mask,
                dest, isbn, decisions, k) -> dict:
    n = world.n
    w, h, t = spec.size_m
    d = rb.design
    rec: dict = {}
    t0 = world.sim_time
    _drive_to_shelf(rb, grip, book, shelf, rbs, world, bus, rec)
    # 空きを見る
    shelf_x = [SHELF_BOOKS_X.setdefault(i, []) for i in range(n)]
    scans = scan_shelf(rb, shelf, keeper, cams, shelf_x, out, f"{k}_{spec.book_id}_before")
    q_st, palm = standing_quat(rb, grip, shelf)
    hand_clear = d.fingertip_z + 0.04 - t
    slots = [choose_slot(sp, t, int(palm[i]), hand_clear) for i, (sp, _) in enumerate(scans)]
    rec["free_spans"] = [sp for sp, _ in scans]
    rec["free_space_source"] = [src for _, src in scans]
    rec["palm_side"] = palm
    rec["slot"] = slots
    for i in range(n):
        if mask[i]:
            bus.publish("robot-b-sort", "observer", "event", "shelf.scan",
                        {"env": i, "shelf_id": shelf.shelf_id, "free_spans_m": [[round(a, 4), round(b, 4)] for a, b in scans[i][0]],
                         "source": scans[i][1], "slot": None if slots[i] is None else [round(v, 4) for v in slots[i]]})
    ok = np.zeros(n, bool)
    has_slot = np.array([s is not None for s in slots]) & mask
    if has_slot.any():
        # 置き場所が決まったら、台車を「手を正面に水平に伸ばして差し込める」位置へ動かす。棚の正面から手を横向きにすると、
        # 本を立てる途中で腕の IK が解の枝の間を飛び、本を落とした（20261020T024421）
        go = slot_base_pose(shelf, q_st[0], slots[int(np.argmax(has_slot))], spec, d)
        rec["slot_base_pose"] = go
        bus.publish("robot-b-sort", "observer", "event", "base.drive", {"to": "slot", "pose": [round(v, 3) for v in go]})
        run_generators(world, [g_drive(rb, go, rbs["max_speed"], rbs["max_yaw_rate"], DT)])
        rb.freeze(~has_slot)
        slot_arr = np.array([s if s is not None else (0.0, t) for s in slots])
        def snap(label: str) -> None:
            rgb, *_ = cams["shelf_close"].render()
            Image.fromarray(np.asarray(rgb)).save(out / f"shelf_{k}_{spec.book_id}_{label.split()[0]}_env0.png")

        run_generators(world, [g_shelve(rb, grip, shelf, keeper, slot_arr, q_st, rec, snap=snap)])
        rb.unfreeze()
        pb, mb = book_frame(book)
        ps = shelf.to_shelf(pb)
        up_y = mb[:, 2, 1]  # 本の天（+y）の鉛直成分
        tilt = np.degrees(np.arccos(np.clip(up_y, -1, 1)))
        inside = (ps[:, 0] > 0) & (ps[:, 0] < keeper.width) & (ps[:, 1] > 0) & (ps[:, 1] < keeper.depth) & (ps[:, 2] < 0.05 + h / 2)
        upright = tilt < 10.0
        released = ~touching(rb, book)
        angles = keeper_ent.get_dofs_position().cpu().numpy()
        lifted = [list(np.nonzero(angles[i] > math.radians(-20))[0].tolist()) for i in range(n)]
        ok = has_slot & upright & inside & released
        intr = rec["shelve"]["palm_shelf_intrusion_mm"]
        rec["physically_valid"] = intr < 1.0  # 手のひらが本棚を 1 mm 以上すり抜けたら、物理的に正しくない（B8 で相談）
        rec.update(tilt_deg=tilt, inside=inside, upright=upright, released=released, book_shelf_pos=ps, fingers_lifted=lifted)
        scans_after = scan_shelf(rb, shelf, keeper, cams, shelf_x, out, f"{k}_{spec.book_id}_after")
        for i in range(n):
            if not has_slot[i]:
                continue
            x_lo, x_hi = ps[i, 0] - t / 2, ps[i, 0] + t / 2
            if ok[i]:
                shelf_x[i].append((float(x_lo), float(x_hi)))
            # 置いた後のカメラで、その x が埋まって見えるか（環境 0 だけ。置いた位置の確かめを別の方法で）
            seen = None
            if i == 0:
                spans_after = scans_after[0][0]
                seen = not any(a <= ps[i, 0] <= b for a, b in spans_after)
            dd = decisions[i]
            inv.add(Placement(run_id=run_id, env=i, shelf_id=shelf.shelf_id, book_id=spec.book_id, isbn=isbn[i],
                              title=spec.title, value_jpy=dd.value_jpy, x_lo_m=float(x_lo), x_hi_m=float(x_hi),
                              fingers=lifted[i], free_space_source=scans[i][1], placed_sim_s=world.sim_time,
                              upright=bool(upright[i]), inside=bool(inside[i]), spine_y_m=float(ps[i, 1] - w / 2),
                              tilt_deg=float(tilt[i]), note="" if seen is None else f"camera_sees_occupied={seen}"))
            bus.publish("robot-b-sort", "observer", "event", "shelf.placed",
                        {"env": i, "shelf_id": shelf.shelf_id, "book_id": spec.book_id, "x_m": [round(float(x_lo), 4), round(float(x_hi), 4)],
                         "upright": bool(upright[i]), "inside": bool(inside[i]), "tilt_deg": round(float(tilt[i]), 1),
                         "fingers_lifted": lifted[i], "camera_sees_occupied": seen})
    for i in range(n):
        if mask[i] and not has_slot[i]:
            bus.publish("robot-b-sort", "observer", "event", "escalation",
                        {"env": i, "book_id": spec.book_id, "reason": "shelf_no_space"})
    # 戻る
    run_generators(world, [g_home(rb, 150)])
    bus.publish("robot-b-sort", "observer", "event", "base.drive", {"to": "station"})
    _drive_to_station(rb, station, rbs, world)
    rec["ok"] = ok
    rec["shelf_s"] = world.sim_time - t0
    return rec


FLAT_REACH = 0.50  # 台車の中心から、平らに置く本の中心までの水平の距離
FLAT_MAX_TOP_Z = 0.12  # 平らに積む高さの上限（棚の座標。床板の下面から）
FLAT_RELEASE_OPEN = 0.004  # 本棚で放すときに指を開く量（全開にすると壁が本棚の指を跳ね上げる）
FLAT_BACKOFF = 0.35  # 置いた後、待機姿勢へ戻す前に台車を下げる距離
FLAT_CARRY_Z = 0.16  # 棚の上へ運ぶときの本の底の高さの下限（棚の座標）。跳ね上がった指の先（約 0.17）の近くまで
FLAT_Z_BAND = (0.005, 0.050)  # 空きを見る高さ。平らな本は薄い（10 mm 前後）ので、立てる場合の 12 mm より下から見る
FLAT_Y_MIN = -0.10  # 平らな本の背は棚の手前の縁からはみ出す（B5 で 74 mm）ので、その面も「埋まっている」に数える
SHELF_PILES: dict = {}  # 環境ごとの平積みの山 [{x_lo, x_hi, top_z, n}]（棚の座標。置いた後に測った値）


def yield_shelf_flat(rb, grip, spec, book, shelf, keeper, keeper_ent, station, rbs, world, cams, bus, inv, run_id, out,
                     mask, dest, isbn, decisions, k) -> dict:
    """本棚へ平らに置く（02_確認事項 B8 の回答 c）。背を手前・小口を奥にし、天地を棚の幅（x）に沿わせる。
    小口は休んでいる指の先より手前で止める（手の小口の側の壁が本の底まで下りるため）。床に空きが無ければ、記録した山の上に積む。"""
    n = world.n
    w, h, t = spec.size_m
    d = rb.design
    rec: dict = {"mode": "flat"}
    t0 = world.sim_time
    _drive_to_shelf(rb, grip, book, shelf, rbs, world, bus, rec)
    shelf_x = [SHELF_BOOKS_X.setdefault(i, []) for i in range(n)]
    piles = [SHELF_PILES.setdefault(i, []) for i in range(n)]
    scans = scan_shelf(rb, shelf, keeper, cams, shelf_x, out, f"{k}_{spec.book_id}_before", z_band=FLAT_Z_BAND,
                       y_min=FLAT_Y_MIN)
    yc = flat_book_center_y(keeper, w, d.wall_t_m)
    plans: list = []
    for i, (sp, src) in enumerate(scans):
        if not mask[i] or yc is None:
            plans.append(None)
            continue
        x = choose_flat_slot(sp, h)
        if x is not None:
            plans.append({"x": x, "level": 0, "surface_z": BASE_T, "pile": None, "source": src})
            continue
        st = choose_stack(piles[i], h, FLAT_MAX_TOP_Z - t, keeper.usable_x)
        if st is None:
            plans.append(None)
            continue
        j, xc = st
        pl = piles[i][j]
        plans.append({"x": (xc - h / 2, xc + h / 2), "level": pl["n"], "surface_z": pl["top_z"], "pile": j,
                      "source": "inventory"})
    rec["free_spans"] = [sp for sp, _ in scans]
    rec["free_space_source"] = [src for _, src in scans]
    rec["center_y_m"] = yc
    rec["plan"] = plans
    has = np.array([p is not None for p in plans]) & mask
    for i in range(n):
        if mask[i]:
            pl = plans[i]
            bus.publish("robot-b-sort", "observer", "event", "shelf.scan",
                        {"env": i, "shelf_id": shelf.shelf_id, "mode": "flat",
                         "free_spans_m": [[round(a, 4), round(b, 4)] for a, b in scans[i][0]], "source": scans[i][1],
                         "slot": None if pl is None else [round(v, 4) for v in pl["x"]],
                         "stack_level": None if pl is None else pl["level"],
                         "slot_source": None if pl is None else pl["source"]})
    ok = np.zeros(n, bool)
    if has.any():
        i0 = int(np.argmax(has))
        pls = [p if p is not None else plans[i0] for p in plans]
        x_c = np.array([(p["x"][0] + p["x"][1]) / 2 for p in pls])
        surf = np.array([p["surface_z"] for p in pls])
        target = shelf.to_world(np.stack([x_c, np.full(n, yc), surf], -1))
        fwd_s = shelf.dir_world([0.0, 1.0, 0.0])
        base = target[i0] - fwd_s * FLAT_REACH
        go = (float(base[0]), float(base[1]), float(math.atan2(fwd_s[1], fwd_s[0])))
        rec["slot_base_pose"] = go
        bus.publish("robot-b-sort", "observer", "event", "base.drive", {"to": "slot", "pose": [round(v, 3) for v in go]})
        run_generators(world, [g_drive(rb, go, rbs["max_speed"], rbs["max_yaw_rate"], DT)])
        rb.freeze(~has)
        # 手は真下向き。右の指（背の側のゴムと爪）を棚の手前に、小口の側の壁を奥にする
        q_flat = lip_side_quat(np.broadcast_to(shelf.dir_world([0.0, -1.0, 0.0]), (n, 3)).copy())
        grip_f = dataclasses.replace(grip, quat=q_flat)
        snap_cam = cams["shelf_close"]

        def snap(label: str) -> None:  # 画像（環境 0）と、その時点で跳ね上がっている本棚の指（環境 0。何をした時に跳ねたかを探す）
            rgb, *_ = snap_cam.render()
            Image.fromarray(np.asarray(rgb)).save(out / f"shelf_{k}_{spec.book_id}_{label}_env0.png")
            a0 = keeper_ent.get_dofs_position().cpu().numpy()[0]
            rec.setdefault("fingers_lifted_env0_at", {})[label] = np.nonzero(a0 > math.radians(-20))[0].tolist()

        snap("start")
        # 関節空間で向きを変えて棚の上へ運び、真下へ下ろして放す（廃棄箱と同じ g_place_flat）。運ぶ高さは、本の底（＝手の
        # 小口の側の壁の下端）が本棚の指より上。置く面の 60 mm 上へ関節空間で直接運ぶと、途中で手が指 13 枚を跳ね上げた
        # （20261023T010134）
        mv: dict = {}
        carry = shelf.stand_h + np.maximum(surf + 0.06, FLAT_CARRY_Z)
        run_generators(world, [g_move_book_joint(rb, grip_f, target[:, :2], q_flat, carry, out=mv)])
        rec["move"] = mv
        snap("above")
        o: dict = {}
        run_generators(world, [g_place_flat(rb, grip_f, shelf.stand_h + surf, o, release_open=FLAT_RELEASE_OPEN)])
        snap("placed")
        rec["place"] = o.get("place", {})
        # 手を棚の手前へまっすぐ引いてから待機姿勢へ戻す。棚の上から関節空間で直接戻すと、手が本棚の指を跳ね上げた
        # （20261023T014456。置くまでは 0 枚、戻した後に 3〜11 枚）
        g_back = rb.gripper.get_pos().cpu().numpy() + shelf.dir_world([0.0, -1.0, 0.0]) * 0.25
        g_back[:, 2] += 0.05
        run_generators(world, [g_cart_chain(rb, g_back, q_flat, 100, n_way=10, profile=min_jerk)])
        snap("back")
        # 待機姿勢では手が台車の正面の低い所に来るので、置いた位置のままだと手が本棚に重なる。台車を下げてから戻す
        # （20261023T015123: 引いてから戻しても、待機姿勢で指 4 枚を跳ね上げた）
        bp = rb.base_pose()[i0]
        fwd_b = np.array([math.cos(bp[2]), math.sin(bp[2])])
        away = (float(bp[0] - fwd_b[0] * FLAT_BACKOFF), float(bp[1] - fwd_b[1] * FLAT_BACKOFF), float(bp[2]))
        run_generators(world, [g_drive(rb, away, rbs["max_speed"], rbs["max_yaw_rate"], DT)])
        run_generators(world, [g_home(rb, 150)])
        snap("home")
        rb.unfreeze()
        pb, mb = book_frame(book)
        ps = shelf.to_shelf(pb)
        tilt = np.degrees(np.arccos(np.clip(np.abs(mb[:, 2, 2]), 0, 1)))  # 本の厚みの向きと鉛直のなす角
        spine_front = np.einsum("ni,i->n", mb[:, :, 0], shelf.dir_world([0.0, -1.0, 0.0])) > 0.9  # 本の +x（背）が手前
        inside = ((ps[:, 0] > 0) & (ps[:, 0] < keeper.width) & (ps[:, 1] > 0) & (ps[:, 1] < keeper.depth)
                  & (np.abs(ps[:, 2] - (surf + t / 2)) < 0.008))
        released = ~touching(rb, book)
        angles = keeper_ent.get_dofs_position().cpu().numpy()
        lifted = [list(np.nonzero(angles[i] > math.radians(-20))[0].tolist()) for i in range(n)]
        ok = has & inside & spine_front & (tilt < 10.0) & released
        rec.update(tilt_deg=tilt, inside=inside, spine_front=spine_front, released=released, book_shelf_pos=ps,
                   pos_err_x_mm=(ps[:, 0] - x_c) * 1000, pos_err_y_mm=(ps[:, 1] - yc) * 1000, fingers_lifted=lifted)
        scans_after = scan_shelf(rb, shelf, keeper, cams, shelf_x, out, f"{k}_{spec.book_id}_after", z_band=FLAT_Z_BAND,
                                 y_min=FLAT_Y_MIN)
        for i in range(n):
            if not has[i]:
                continue
            pl = plans[i]
            # 置いた本の x の範囲（天地 h が x に沿う。置いた後に測った位置から）
            x_lo, x_hi = float(ps[i, 0] - h / 2), float(ps[i, 0] + h / 2)
            top = float(ps[i, 2] + t / 2)
            if ok[i]:
                if pl["pile"] is None:
                    shelf_x[i].append((x_lo, x_hi))
                    piles[i].append({"x_lo": x_lo, "x_hi": x_hi, "top_z": top, "n": 1})
                else:
                    piles[i][pl["pile"]].update(x_lo=x_lo, x_hi=x_hi, top_z=top, n=pl["level"] + 1)
            seen = None
            if i == 0 and pl["level"] == 0:  # 置いた後のカメラで、その x が埋まって見えるか（床の本だけ。積んだ本は見えない高さ）
                seen = not any(a <= ps[i, 0] <= b for a, b in scans_after[0][0])
            dd = decisions[i]
            inv.add(Placement(run_id=run_id, env=i, shelf_id=shelf.shelf_id, book_id=spec.book_id, isbn=isbn[i],
                              title=spec.title, value_jpy=dd.value_jpy, x_lo_m=x_lo, x_hi_m=x_hi,
                              fingers=lifted[i], free_space_source=pl["source"], placed_sim_s=world.sim_time,
                              upright=False, inside=bool(inside[i]), spine_y_m=float(ps[i, 1] - w / 2),
                              tilt_deg=float(tilt[i]), pose="flat", stack_level=int(pl["level"]), top_z_m=top,
                              spine_front=bool(spine_front[i]),
                              note="" if seen is None else f"camera_sees_occupied={seen}"))
            bus.publish("robot-b-sort", "observer", "event", "shelf.placed",
                        {"env": i, "shelf_id": shelf.shelf_id, "book_id": spec.book_id, "pose": "flat",
                         "x_m": [round(x_lo, 4), round(x_hi, 4)], "stack_level": int(pl["level"]),
                         "inside": bool(inside[i]), "spine_front": bool(spine_front[i]),
                         "tilt_deg": round(float(tilt[i]), 1), "camera_sees_occupied": seen})
    for i in range(n):
        if mask[i] and not has[i]:
            bus.publish("robot-b-sort", "observer", "event", "escalation",
                        {"env": i, "book_id": spec.book_id, "reason": "shelf_no_space" if yc is not None else "shelf_too_deep"})
    bus.publish("robot-b-sort", "observer", "event", "base.drive", {"to": "station"})
    _drive_to_station(rb, station, rbs, world)
    bp = rb.base_pose()
    rec["station_err_m"] = np.linalg.norm(bp[:, :2] - station[:2], axis=1)
    rec["station_yaw_err_deg"] = np.degrees((bp[:, 2] - station[2] + math.pi) % (2 * math.pi) - math.pi)
    rec["arm_err_from_home_rad"] = np.abs(rb.e.get_qpos().cpu().numpy()[:, rb.arm] - rb.home[rb.arm]).max(axis=1)
    rec["ok"] = ok
    rec["shelf_s"] = world.sim_time - t0
    return rec


SLOT_REACH = 0.45  # 台車の中心から、差し込む途中の手（gripper）までの水平の距離


def slot_base_pose(shelf, quat_stand: np.ndarray, slot, spec, d) -> tuple[float, float, float]:
    """立てた本を棚の slot へ差し込むときの台車の位置と向き。手の向き（手のひら → 指先 = z）を台車の正面に合わせ、
    差し込む途中の手の位置が台車の正面 SLOT_REACH に来るようにする。"""
    zg = quat_to_mat(quat_stand[None])[0][:, 2]
    zg_h = np.array([zg[0], zg[1], 0.0])
    zg_h /= np.linalg.norm(zg_h)
    w, h, t = spec.size_m
    x_c = (slot[0] + slot[1]) / 2
    book_mid = shelf.to_world([x_c, -0.02, 0.0])  # 差し込みの途中（小口が棚の手前の縁を越えたあたり）
    grip = book_mid - zg_h * (d.fingertip_z - t / 2)
    base = grip - zg_h * SLOT_REACH
    return float(base[0]), float(base[1]), float(math.atan2(zg_h[1], zg_h[0]))


def _result(bus, from_task, i, spec, isbn, outcome, dest, value, reason) -> None:
    bus.publish(from_task, "observer", "event", "book.result",
                {"env": int(i), "book_id": spec.book_id, "title": spec.title, "isbn": isbn, "outcome": outcome,
                 "destination": None if dest is None else str(dest), "value_jpy": int(value), "reason": reason})


if __name__ == "__main__":
    main()
