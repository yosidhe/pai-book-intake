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

A と B は並行して動く（既定。2026-10-28 ユーザーの要件）
  - A は、置いた本の査定が終わったら（査定から A への appraisal.done。行き先つき）、すぐ次の本を取りに行く。
    査定台の 2 スロットが両方ふさがっていれば、空くまで待つ。B の作業待ちの列へはプログラムの中で渡す（通知にしない）
  - B は、査定が済んでいて運んでよい本があれば、置かれた順に運ぶ。無ければ待機位置で待つ
  - 査定台の上に手を入れるのは一度に 1 台だけ（zone["table"]）。もう 1 台は台の手前で待つ
  - --sequential で、前の「1 冊ずつ A → B」に戻せる（比べるため）

保留棚（02_確認事項 B14。配置データに hold_table があるとき）
  - 査定できなかった本（読めない・カタログに無い・本と ISBN が合わない）は、B が保留棚へ平らに積む（廃棄箱と同じ置き方）
  - 積むと崩れるおそれがあれば（hold.py）、取らずに台に残し（hold_shelf_full）、オブザーバが箱ごと止める
  - 本棚が満杯で置けない本（shelf_full）は保留棚に載せない（台に残して止める）
  - 読めない本が続いても、保留棚に置けなくなるまでは止めない

B の台車は、本棚へ行くときも戻るときも station_approach を通り、向きはそこで変える（B15。site.b_route）。
B（台車と腕）が廃棄箱・保留棚・査定台に触れた歩数を毎歩数え（ContactCounter）、0 回を合格の条件にする

まだ入っていないもの（結果の not_implemented に書く）
  - 裏返し（B3: 空中で回す）。バーコードが見えない本は、保留棚へ移す（保留棚が無ければ人の判断待ちとして台に残す）
  - 箱の中の縦置きの本（B2）。今の箱は平積みだけ

描画は環境 0 だけ（WSL2 の Docker は CPU 描画）。環境 0 以外は、バーコードは本の向き（表紙が上か）から、本棚の空きは
置いた本の位置から決める（どちらも結果に source として残す）。

A を ACT で動かす（--a-policy act --ckpt <act_train の checkpoint>。03_ACT組込み計画）
  - 取り出し → 運搬 → 配置 → 待機姿勢を、学習済みの ACT が 50 Hz で動かす（act_a.act_episode と同じ）。置き場所は台帳の空きから
    選んで観測に入れる。ACT は途中で止められないので、動かし始める前に査定台を確保する
  - 本の位置と向きは、スクリプト制御と同じく正解値を使う

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
from .hold import FULL_REASON, HOLDABLE, StackRule, TopBook, inset_from_edge, plan_stack
from .inventory import Inventory, Placement
from .keeper import (BASE_T, KeeperSpec, choose_flat_slot, choose_slot, choose_stack, flat_book_center_y, free_spans,
                     keeper_mjcf, occupancy_from_points, spans_from_books)
from .observer import Observer, ObserverConfig
from .pinch import (PinchRobot, book_frame, build_pinch_robot_xml, g_cart_chain, g_drive, g_home, g_move_book,
                    g_joints, g_move_book_joint, g_pick, g_place_flat, g_shelve, lip_side_quat, min_jerk, run_generators,
                    shelve_grip_offset, standing_quat, touching)
from .release import HELD, HOLD, ReleaseGate
from .runlog import classify, new_run, save_result
from .site import DEFAULT_SITE, b_route, load_site
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
# B が待機位置で待つときの腕（関節 1〜7）。HOME のままだと手先が査定台の奥の縁（台座から 0.47 m 前・高さ 0.59 m）の
# 上にかかり、点検カメラの画像の 2 つ目の置き場所に写り込んでバーコードが読めなかった（p2_line 20261027T211526・
# 20261027T214721 の book-05）。肩と肘をたたんで手先を 0.25 m 前（台の縁より 0.19 m 奥）・高さ 0.54 m へ引く。
# 手のひらは下向きのまま。取りに行く前に HOME へ戻すので、ほかの動きは HOME から始まる前のまま
B_WAIT_ARM = np.array([0.0, -1.0, 0.0, -2.6, 0.0, 1.6, -0.785])
NOT_IMPLEMENTED = ["flip (B3)", "standing books in the box (B2)",
                   "standing books on the shelf (keeper fingers jam; 02_確認事項 B8 a/b). The default shelf mode is flat (B8 c)"]
SHELF_MODES = ("flat", "standing")
A_POLICIES = ("scripted", "act")


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
    ap.add_argument("--inject-camera-dirty-from", type=int, default=-1,
                    help="わざと失敗させる: この冊目（0 から）以降、点検カメラの画像を汚す（-1 = しない）。"
                         "読めない本が続くと、オブザーバが箱ごと止める（02_確認事項 B9）")
    ap.add_argument("--shelf-capacity", type=int, default=20,
                    help="この箱に割り当てた本棚の冊数。超える本は台に残し（shelf_full）、オブザーバが箱ごと止める。"
                         "「本棚に空きがない」を起こすため（02_確認事項 B9）")
    ap.add_argument("--hold-max-stack-m", type=float, default=None,
                    help="保留棚に積む高さの上限（台の上面から。省略時は配置データの max_stack_m）。"
                         "「保留棚に積めない」を起こすため（02_確認事項 B14）")
    ap.add_argument("--sequential", action="store_true",
                    help="A と B を並行させず、1 冊ずつ A → B の順に処理する（前の動き。比べるため）")
    ap.add_argument("--a-policy", choices=A_POLICIES, default="scripted",
                    help="A の取り出し・運搬・配置・待機姿勢を動かすもの。act = 学習済みの ACT（--ckpt。03_ACT組込み計画）")
    ap.add_argument("--ckpt", type=Path, help="--a-policy act の checkpoint（act_train の final など）")
    ap.add_argument("--ensemble", type=float, default=None, help="ACT の temporal ensembling の係数（省略で使わない）")
    args = ap.parse_args()
    if args.a_policy == "act" and args.ckpt is None:
        ap.error("--a-policy act には --ckpt が要る")
    return args


def main() -> None:
    args = parse_args()
    import genesis as gs

    run_id, out, meta = new_run(TRIAL, {k: (ckpt_name(v) if k == "ckpt" and v else str(v.name) if isinstance(v, Path) else v)
                                        for k, v in vars(args).items()},
                                extra={"genesis": gs.__version__})
    bus = Bus(run_id, out / "bus.jsonl")
    # 読めない本は人の判断待ちとして査定台に残る。台の置き場所は 2 つなので、3 冊続けて読めないまで待つと、3 冊目を置く前に
    # 台が埋まり、止まらずに待ち続けた（2026-10-28）。置き場所の数だけ続けて読めなければ、カメラの故障を疑って止める
    # 保留棚があれば、読めない本は保留棚へ移すので、読めない本が続いても止めない（止めるのは保留棚に積めないとき。B14）
    hold_on = "hold_table" in load_site(args.site).raw
    observer = Observer(bus, ObserverConfig(max_consecutive_failures=3, barcode_unreadable_limit=None if hold_on else 2,
                                            shelf_capacity=20))
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


def ckpt_name(p: Path) -> str:
    """checkpoint を、絶対パスでなく「act_train の run_id/フォルダ名」で記録する（例: 20261028T160045-6cf9ae/final）"""
    p = Path(p)
    return f"{p.parent.name}/{p.name}"


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
    ents = [_box(scene, (ix + 2 * wall_t, iy + 2 * wall_t, floor), (cx, cy, floor / 2), color)]
    for size, off in [((wall_t, iy + 2 * wall_t, wall_h), (-ix / 2 - wall_t / 2, 0)),
                      ((wall_t, iy + 2 * wall_t, wall_h), (ix / 2 + wall_t / 2, 0)),
                      ((ix, wall_t, wall_h), (0, -iy / 2 - wall_t / 2)),
                      ((ix, wall_t, wall_h), (0, iy / 2 + wall_t / 2))]:
        ents.append(_box(scene, size, (cx + off[0], cy + off[1], wall_h / 2), color))
    return ents


def build(args, out: Path, site, pile_specs):
    import genesis as gs

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=4),
                     rigid_options=gs.options.RigidOptions(enable_self_collision=False,
                                                           friction_cone=gs.friction_cone.elliptic,
                                                           constraint_timeconst=0.01,
                                                           # ACT は力の上限を環境ごとに出す（act_a.apply_action）
                                                           batch_dofs_info=args.a_policy == "act"),
                     vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane(contype=ENV_BIT, conaffinity=ENV_BIT))
    bx = site["box"]
    _open_box(scene, bx["center"], bx["inner"], bx["floor"], bx["wall_h"], bx["wall_t"], CARD)
    tb = site["appraisal_table"]
    # B が触れてはいけない物（02_確認事項 B15。ContactCounter で毎歩数える）
    obstacles = {"appraisal_table": [_box(scene, (*tb["size"], tb["top"]), (*tb["center"], tb["top"] / 2), GRAY)]}
    bn = site["bin"]
    obstacles["bin"] = _open_box(scene, bn["center"], bn["inner"], bn["floor"], bn["wall_h"], bn["wall_t"], BIN_COLOR)
    if "hold_table" in site.raw:
        ht = site["hold_table"]
        obstacles["hold_table"] = [_box(scene, (*ht["size"], ht["top"]), (*ht["center"], ht["top"] / 2), GRAY)]
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
    return scene, world, rob_a, rob_b, books, keeper_ent, keeper, cams, obstacles


# 手が、本を取る面・置く面に触れるのは、この手の動き方そのもの（爪を本と面のすき間に入れる・面まで下ろして放す）なので
# 接触に数えず、別に数える（hand_on_surface）。工程（phase）・障害物の部品・B の部品の 3 つが全部合うときだけ。
# 直す前の実行（20261101T003116-6d04d8）では、査定台で取るときの爪と指 11 回・廃棄箱の床に置くときの爪 5 回がこれだった
SURFACE_PHASES = {"appraisal_table": ("b_pick",), "bin": ("bin_place",), "hold_table": ("hold_place",)}


class ContactCounter:
    """B（台車と腕）が障害物（廃棄箱・保留棚・査定台）に触れている歩数を毎歩数える（02_確認事項 B15）。
    B の geom が片側にある接触だけを見るので、本と箱の接触（廃棄箱に入れた本など）は数えない。
    手が取る面・置く面に触れるのは SURFACE_PHASES のとおり別に数える（廃棄箱は床だけ。壁は数える）。
    触れ始めた所ごとに、そのときの工程（phase）・B の部品・シミュの中の時刻と長さを残す（どこで当たったかを探すため）"""

    def __init__(self, robot, obstacles: dict, world, diagnostic: dict | None = None):
        """diagnostic: 数えるが合格の条件に入れない物（{名前: 物体の列}）。例: 本（B が隣の本に触れたかを探すため）"""
        import torch

        self.e, self.world, self.n, self.robot = robot.e, world, world.n, robot
        self.links = robot.e._solver.links
        self.counted = sorted(obstacles)
        allv = {**obstacles, **(diagnostic or {})}
        self.names = sorted(allv)
        self.ranges = {k: [(int(o.geom_start), int(o.geom_end)) for o in v] for k, v in allv.items()}
        # 置き面: 査定台と保留棚は全部、廃棄箱は床（_open_box の最初の物体）だけ
        self.surface = {k: self.ranges[k][:1] if k == "bin" else self.ranges[k] for k in self.counted}
        arm = {"base"} | {f"link{j}" for j in range(8)}
        hand = [int(lk.idx) for lk in robot.e.links if lk.name not in arm]
        self.hand_names = sorted(lk.name for lk in robot.e.links if lk.name not in arm)
        self.is_hand = torch.zeros(len(self.links), dtype=torch.bool)
        self.is_hand[hand] = True
        self.phase = "start"
        self.steps = {k: np.zeros(self.n, int) for k in self.names}
        self.surface_steps = {k: np.zeros(self.n, int) for k in self.names}
        self.prev = {k: np.zeros(self.n, bool) for k in self.names}
        self.open: dict = {}
        self.episodes: list[dict] = []

    def _in(self, other, ranges):
        import torch

        m = torch.zeros_like(other, dtype=torch.bool)
        for lo, hi in ranges:
            m |= (other >= lo) & (other < hi)
        return m

    def __call__(self) -> None:
        import torch

        ci = self.e.get_contacts()
        ga, gb, vm, la, lb = ci["geom_a"], ci["geom_b"], ci["valid_mask"], ci["link_a"], ci["link_b"]
        if ga.ndim == 1:
            ga, gb, vm, la, lb = ga[None], gb[None], vm[None], la[None], lb[None]
        hit = np.zeros((len(self.names), self.n), bool)
        surf = np.zeros((len(self.names), self.n), bool)
        hits = [None] * len(self.names)
        if ga.shape[-1] > 0:
            mine_a = (ga >= self.e.geom_start) & (ga < self.e.geom_end)
            other = torch.where(mine_a, gb, ga)
            own_link = torch.where(mine_a, la, lb)
            hand = self.is_hand.to(own_link.device)[own_link.long()]
            for j, k in enumerate(self.names):
                m = self._in(other, self.ranges[k]) & vm
                if self.phase in SURFACE_PHASES.get(k, ()):
                    s = m & hand & self._in(other, self.surface[k])
                    surf[j] = s.any(dim=1).cpu().numpy()
                    m = m & ~s
                hits[j] = m
                hit[j] = m.any(dim=1).cpu().numpy()
        t = round(self.world.sim_time, 2)
        for j, k in enumerate(self.names):
            self.steps[k] += hit[j]
            self.surface_steps[k] += surf[j]
            for i in np.nonzero(hit[j] & ~self.prev[k])[0]:
                sel = hits[j][int(i)]
                own = own_link[int(i)][sel].cpu().numpy().tolist()
                oth = other[int(i)][sel].cpu().numpy().tolist()
                parts = sorted({p for g in oth for p, (lo, hi) in enumerate(self.ranges[k]) if lo <= g < hi})
                pos = ci["position"][int(i)][sel] if ci["position"].ndim == 3 else ci["position"][sel]
                ep = {"env": int(i), "obstacle": k, "phase": self.phase, "sim_time": t,
                      "b_links": sorted({self.links[x].name for x in own}),
                      "obstacle_parts": parts,  # 廃棄箱は 0 = 床・1〜4 = 壁（-x・+x・-y・+y）
                      "position": [round(float(v), 3) for v in pos[0].cpu().numpy()],
                      "gripper": [round(float(v), 3) for v in self.robot.gripper.get_pos().cpu().numpy()[int(i)]],
                      "steps": 0}
                self.episodes.append(ep)
                self.open[(k, int(i))] = ep
            for i in np.nonzero(hit[j])[0]:
                self.open[(k, int(i))]["steps"] += 1
            self.prev[k] = hit[j]

    def summary(self) -> dict:
        per_env = [int(sum(self.steps[k][i] for k in self.counted)) for i in range(self.n)]
        by_phase: dict = {}
        for ep in self.episodes:
            if ep["obstacle"] not in self.counted:
                continue
            key = f"{ep['obstacle']}@{ep['phase']}"
            by_phase[key] = by_phase.get(key, 0) + 1
        return {"steps_in_contact": {k: v.tolist() for k, v in self.steps.items() if k in self.counted},
                "total_steps_per_env": per_env,
                "episodes_by_obstacle_phase": by_phase,
                "episodes": [ep for ep in self.episodes if ep["obstacle"] in self.counted],
                "diagnostic_episodes": [ep for ep in self.episodes if ep["obstacle"] not in self.counted],
                "hand_on_surface_steps": {k: v.tolist() for k, v in self.surface_steps.items()},
                "hand_links": self.hand_names, "surface_phases": SURFACE_PHASES,
                "note": "B（台車と腕）の geom と障害物の接触。本と箱の接触は数えない。手が取る面・置く面に触れた歩数は"
                        " hand_on_surface_steps に別に数える（合格の条件には入れない）"}


CONTACTS: list = []  # 今の ContactCounter（工程の名前を書き込むため。無ければ何もしない）


def _phase(name: str) -> None:
    if CONTACTS:
        CONTACTS[0].phase = name


def dirty_lens(rgb: np.ndarray) -> np.ndarray:
    """わざと起こす故障: 点検カメラのレンズの汚れ（ぼかして白っぽい膜をかける）。"""
    from PIL import ImageFilter

    img = Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=14))
    a = np.asarray(img).astype(np.float32)
    return np.clip(0.45 * a + 0.55 * 200.0, 0, 255).astype(np.uint8)


def stage_rates(per_book: list[dict], catalog) -> dict:
    """環境 0 の工程ごとの成功率（02_確認事項 B10）。分母はその工程まで来た本。"""
    st = {k: [0, 0] for k in ("pick", "barcode", "appraisal", "transport", "free_space", "shelve", "discard", "hold",
                              "overall")}

    def add(k, ok):
        st[k][0] += int(bool(ok))
        st[k][1] += 1

    for r in per_book:
        if "a" not in r:
            continue
        add("overall", r["delivered"][0] if "delivered" in r else False)
        add("pick", r["a"]["on_table"][0])
        if not r["a"]["on_table"][0] or "isbn_read" not in r:
            continue
        isbn = r["isbn_read"][0]
        add("barcode", isbn is not None)
        if (r.get("destination") or [None])[0] == HOLD:  # 査定できなかった本（保留棚へ。B14）
            add("transport", (r.get("b_pick") or {}).get("lifted", [False])[0])
            if "hold" in r:
                add("hold", r["hold"]["in_hold_flat"][0])
            continue
        if isbn is None:
            continue
        b = catalog.by_id(r["book_id"])
        add("appraisal", isbn == b.isbn13 and appraise(isbn, catalog).destination == appraise(b.isbn13, catalog).destination)
        dest = (r.get("destination") or [None])[0]
        if dest is None:
            continue
        add("transport", (r.get("b_pick") or {}).get("lifted", [False])[0])
        if dest == "discard" and "bin" in r:
            add("discard", r["bin"]["in_bin_flat"][0])
        if dest == "shelf" and "shelf" in r:
            plan = r["shelf"].get("plan", [None])[0]
            add("free_space", plan is not None)
            if plan is not None:
                add("shelve", r["shelf"]["ok"][0])
    return {k: {"ok": v[0], "n": v[1], "rate": (v[0] / v[1]) if v[1] else None} for k, v in st.items()}


def carry_z(book, spec, bx) -> float:
    """A が持ち上げる高さ（本の底）。箱の壁の 40 mm 上か、本の今の底の 50 mm 上の高い方。10 冊を積むと上の本は壁より上にあり、
    壁の高さに固定すると「持ち上げ」で下へ動いた（2026-10-28）"""
    z = float((book.get_pos().cpu().numpy()[:, 2] - spec.thickness_m / 2).max())
    return max(bx["wall_h"] + 0.04, z + 0.05)


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
    scene, world, ra, rb, books, keeper_ent, keeper, cams, obstacles = build(args, out, site, pile)
    build_s = time.time() - t_build
    # 本は合格の条件に入れずに数える（B が取る本の隣の本に触れたかを探すため。obstacle_parts は平積みの下からの番号）
    contacts = ContactCounter(rb, obstacles, world, diagnostic={"books": books})
    CONTACTS[:] = [contacts]
    world.monitors.append(contacts)
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
    gates = [ReleaseGate(bus, catalog, n_slots=len(tb["slots"]), shelf_capacity=args.shelf_capacity, env=i) for i in range(n)]
    bus.publish("observer", "*", "note", "run.start",
                {"n_envs": n, "pile_bottom_to_top": pile_ids, "face_down": sorted(face_down), "site": site.site_id,
                 "shelf": {"shelf_id": shelf.shelf_id, "origin": list(shelf.origin), "yaw": shelf.yaw},
                 "hand": HAND.params(), "plate_mode": PLATE_MODE, "claw_mode": CLAW_MODE, "shelf_mode": args.shelf_mode,
                 "not_implemented": NOT_IMPLEMENTED})
    for name in ("overview", "shelf_view"):
        if name in cams:
            cams[name].start_recording(save_to_filename=str(out / f"{name}_env0.mp4"), fps=25)
            ctx["recording"].append(cams[name])
    A_POLICY.clear()
    if args.a_policy == "act":
        from .act_a import ActPolicy

        pol = ActPolicy(args.ckpt, ensemble=args.ensemble)
        A_POLICY.update(policy=pol, steps=int(pol.meta["episode_ctrl_steps"]))
    ht = site.raw.get("hold_table")
    if ht is not None:
        HOLD_RULE[:] = [StackRule(max_stack_m=args.hold_max_stack_m if args.hold_max_stack_m is not None
                                  else float(ht.get("max_stack_m", 0.12)))]
    t0 = time.time()
    per_book = []
    stopped_at = None
    common = (books, pile, site, shelf, keeper, keeper_ent, tb, bn, station, rbs, ra, rb, world, cams, gates, bus, obs, inv,
              run_id, out, catalog, args.shelf_mode)

    def line_stopped(k: int) -> bool:
        if all(obs.is_stopped(i) for i in range(n)):
            # オブザーバが全部の箱を止めた。残りの本には触れない（止めた理由は報告書に出る）
            bus.publish("robot-a-pick", "observer", "event", "line.stopped",
                        {"at_book": k, "remaining": pile_ids[:len(pile) - k][::-1],
                         "reasons": [obs.box(i).stop_reason for i in range(n)]})
            return True
        return False

    run_generators(world, [g_b_wait(rb)])  # B は腕をたたんで待つ
    if args.sequential:
        for k in range(args.books):
            if line_stopped(k):
                stopped_at = k
                break
            idx = len(pile) - 1 - k
            dirty = 0 <= args.inject_camera_dirty_from <= k
            per_book.append(process_book(k, idx, *common, camera_dirty=dirty))
    else:
        per_book, stopped_at = run_parallel(args, n, pile, common, gates, obs, bus, world, line_stopped)
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
    contact = contacts.summary()
    acceptance_ok = all(n_ok[i] == expected[i] for i in range(n)) and all(c == 0 for c in contact["total_steps_per_env"])
    return {
        "criterion": "p2_line_end_to_end",
        "acceptance": "全環境で、処理した本のうち、バーコードが読める向きの本は全部、A が取り出して台に置き、B が行き先へ届けた"
                      "（本棚は棚の中で、flat なら背が手前で平ら・standing なら立っている。廃棄箱は箱の中で平ら）。"
                      "棚に置いた本は inventory.sqlite に記録した。B（台車と腕）が廃棄箱・保留棚・査定台に触れた歩数が 0（B15）",
        "b_contacts": contact,
        "shelf_mode": args.shelf_mode,
        "a_policy": args.a_policy,
        "ckpt": ckpt_name(args.ckpt) if args.ckpt else None,
        "ensemble": args.ensemble,
        "acceptance_ok": acceptance_ok,
        "books_succeeded_per_env": n_ok, "books_expected_per_env": expected, "shelved_ok_per_env": shelved_ok,
        "inventory": shelf_rows,
        "not_implemented": NOT_IMPLEMENTED + (["A and B in parallel (--sequential)"] if args.sequential else []),
        "parallel": not args.sequential,
        "cycle_s_per_book": cycle_summary(per_book, world.sim_time),
        "stop_reasons": [obs.box(i).stop_reason for i in range(n)],
        "stopped_at_book": stopped_at,
        "injected": {"camera_dirty_from": args.inject_camera_dirty_from},
        "stage_rates_env0": stage_rates(per_book, catalog),
        "hold_table": None if ht is None else {"rule": dataclasses.asdict(HOLD_RULE[0]),
                                               "books_per_env": [[b["book_id"] for b in obs.box(i).hold] for i in range(n)],
                                               "top_per_env": [None if HOLD_TOP.get(i) is None else dataclasses.asdict(HOLD_TOP[i])
                                                               for i in range(n)]},
        "sim_time_s": world.sim_time, "wall_build_s": build_s, "wall_run_s": wall,
        "site": site.raw, "per_book": per_book,
    }


class _Ticker:
    """World.step の毎歩の後に、ジェネレータを 1 歩ぶん進める（A を B の動きと同時に動かすため）。
    run_generators と同じ約束（yield k = k 歩待つ）。ジェネレータの中で world.step を呼んではいけない"""

    def __init__(self, gen):
        self.gen, self.wait, self.alive = gen, 0, True

    def __call__(self) -> None:
        if not self.alive:
            return
        if self.wait > 0:
            self.wait -= 1
        while self.alive and self.wait == 0:
            try:
                self.wait = int(next(self.gen))
            except StopIteration:
                self.alive = False


def run_parallel(args, n, pile, common, gates, obs, bus, world, line_stopped) -> tuple[list, int | None]:
    """A と B を並行して動かす。A はジェネレータとして毎歩進め（_Ticker）、B はこの関数の中で順に動かす。
    A は査定が済んだ本を queue に積む。B は queue の本を置かれた順に運び、無ければ待機位置で 1 歩ずつ待つ。"""
    zone: dict = {"table": None}
    queue: list[dict] = []
    state: dict = {"stopped_at": None, "b_idle": False}

    def g_a_worker():
        for k in range(args.books):
            if line_stopped(k):
                state["stopped_at"] = k
                return
            # 査定台が両方ふさがっていれば、B が持っていくまで待つ（止まった箱は数えない）
            # 台の本が全部「人の判断待ち」などで B が持っていけないとき（B が待機していて、渡した本も残っていない）は、
            # 待っても空かないので待たない（g_process_a が tray_full として飛ばす。--sequential と同じ扱い）
            waited = False
            while any(gates[i].free_slot() is None for i in range(n) if not obs.is_stopped(i)):
                if state.get("b_idle") and not any(gates[i].pickable() or gates[i].holdable() for i in range(n)):
                    break
                if not waited:
                    bus.publish("robot-a-pick", "observer", "event", "line.waiting", {"reason": "tray_full", "at_book": k})
                    waited = True
                yield 1
            idx = len(pile) - 1 - k
            dirty = 0 <= args.inject_camera_dirty_from <= k
            st: dict = {"k": k, "idx": idx}
            yield from g_process_a(k, idx, *common, camera_dirty=dirty, st=st, zone=zone)  # 査定の appraisal.done まで
            queue.append(st)  # B の作業待ちの列へ（プログラムの中の受け渡し。通知にはしない）

    ticker = _Ticker(g_a_worker())
    world.monitors.append(ticker)
    ticker()  # 最初の 1 歩ぶんを今進める
    per_book: list = []
    j = 0
    idle = False
    try:
        while True:
            if j < len(queue):
                st = queue[j]
                j += 1
                if st.get("skip_b"):
                    per_book.append(st["rec"])
                    continue
                while zone["table"] not in (None, "b"):  # A が台に手を入れている間は待つ
                    world.step(1)
                zone["table"] = "b"
                idle = state["b_idle"] = False
                per_book.append(process_book_b(st, st["k"], st["idx"], *common, zone=zone))
            elif not ticker.alive:
                break
            else:
                if not idle:
                    bus.publish("robot-b-sort", "observer", "event", "b.idle", {"waiting_for": "appraised_book"})
                    idle = state["b_idle"] = True
                world.step(1)  # 運べる本が無い。待機位置で待つ（A はこの歩みの中で動く）
    finally:
        world.monitors.remove(ticker)
    return per_book, state["stopped_at"]


def cycle_summary(per_book: list[dict], sim_time: float) -> dict:
    """1 冊あたりの時間（シミュの中の時間）。全体を冊数で割った値と、本ごとの A・B の時間"""
    rows = []
    for r in per_book:
        t = r.get("t_sim") or {}
        rows.append({"book_id": r.get("book_id"),
                     "a_s": (t["appraised"] - t["a_start"]) if {"a_start", "appraised"} <= t.keys() else None,
                     "b_s": (t["b_done"] - t["b_start"]) if {"b_start", "b_done"} <= t.keys() else None,
                     "wait_for_b_s": (t["b_start"] - t["appraised"]) if {"b_start", "appraised"} <= t.keys() else None})
    done = [r for r in per_book if r.get("t_sim", {}).get("b_done") is not None]
    return {"books": len(per_book), "sim_time_s": sim_time,
            "per_book_s": (sim_time / len(per_book)) if per_book else None,
            "delivered_books": len(done), "rows": rows}


def g_b_wait(rb, steps: int = 120):
    """B の腕を待機の形（B_WAIT_ARM）にたたむ。点検カメラの視野から手を出すため"""
    _phase("b_wait_arm")
    yield from g_joints(rb, rb.arm, np.tile(B_WAIT_ARM, (rb.n, 1)), steps)
    _phase("b_idle")


A_POLICY: dict = {}  # --a-policy act のとき {"policy": act_a.ActPolicy, "steps": 1 冊あたりの制御の回数}


def g_act_a(ra, policy, steps: int, book, spec, slot_xy: np.ndarray, out: dict):
    """A の取り出し → 運搬 → 配置 → 待機姿勢を ACT で動かす（act_a.act_episode と同じ中身。50 Hz で観測を読み直す）。
    歩みを譲るジェネレータ。out["pick"]["lifted"] に、本が 30 mm 以上上がったかを入れる（act_a の評価と同じ基準）"""
    from .act_a import CTRL_EVERY, apply_action, initial, observe

    init = initial(book, spec)
    z0 = book.get_pos().cpu().numpy()[:, 2]
    lift = np.zeros(ra.n)
    policy.reset()
    for _ in range(steps):
        s, e = observe(ra, book, spec, init, slot_xy, face_up=True)  # 裏向きの本は表向きの数値にして渡す（本は動かさない）
        apply_action(ra, policy.act(s, e))
        yield CTRL_EVERY
        lift = np.maximum(lift, book.get_pos().cpu().numpy()[:, 2] - z0)
    ra.set_forces_per_env(np.tile(ra.force, (ra.n, 1)))  # 力の上限を既定へ戻す
    out["pick"] = {"lifted": lift > 0.03, "lift_max_mm": lift * 1000, "ctrl_steps": steps}


def g_process_a(k, idx, books, pile, site, shelf, keeper, keeper_ent, tb, bn, station, rbs, ra, rb, world, cams, gates,
                bus, obs, inv, run_id, out, catalog, shelf_mode: str = "flat", camera_dirty: bool = False,
                st: dict | None = None, zone: dict | None = None):
    """A の工程（箱から取る → 査定台に置く → バーコードを読む → 査定）。歩みを譲るジェネレータ。
    結果は st に入れる（rec・alive・isbn・decisions など。B の工程 process_book_b が使う）。
    zone を渡すと、査定台の上に手を入れる間だけ zone["table"] = "a" にする（B と手がぶつからないように。並行のとき）"""
    st = {} if st is None else st
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
        st.update(rec=rec, skip_b=True)
        return
    lower = [(books[j], pile[j]) for j in range(idx)]
    bx = site["box"]
    o: dict = {}
    slot_xy = np.array([tb["slots"][s] for s in slot])
    if A_POLICY:
        # ACT は動作の途中で止められない（台の手前で待つことを学んでいない）ので、動かし始める前に台を確保する
        if zone is not None:
            while zone.get("table") not in (None, "a"):
                yield 1
            zone["table"] = "a"
        bus.publish("robot-a-pick", "observer", "event", "pick.start", {"book_id": spec.book_id, "policy": "act"})
        rec["t_sim"] = {"a_start": world.sim_time}
        yield from g_act_a(ra, A_POLICY["policy"], A_POLICY["steps"], book, spec, slot_xy, o)
        # 待機姿勢へそろえる（ACT が戻りきらないと、点検カメラに手が写る）。戻りきっていれば動かない
        yield from g_home(ra, 60)
        pick_rec = o["pick"]
    else:
        bus.publish("robot-a-pick", "observer", "event", "pick.start", {"book_id": spec.book_id})
        rec["t_sim"] = {"a_start": world.sim_time}
        yield from g_pick(ra, book, spec, o, lower=lower, surface_z=None if lower else bx["floor"],
                          carry_bottom_z=carry_z(book, spec, bx), plate_mode=PLATE_MODE, claw_mode=CLAW_MODE)
        grip_a = o["grip"]
        if zone is not None:  # 査定台に B の手が入っている間は、台の手前で待つ
            while zone.get("table") not in (None, "a"):
                yield 1
            zone["table"] = "a"
        yield from g_move_book(ra, grip_a, slot_xy)
        yield from g_place_flat(ra, grip_a, tb["top"], o)
        yield from g_home(ra, 120)
        pick_rec = grip_a.rec
    pb, mb = book_frame(book)
    w, h, t = spec.size_m
    on_table = (np.abs(pb[:, 2] - (tb["top"] + t / 2)) < 0.006) & (np.abs(mb[:, 2, 2]) > 0.97)
    rec["a"] = {"pick": pick_rec, "place": o.get("place", {}), "on_table": on_table, "slot": slot,
                "pos_err_from_slot_mm": np.linalg.norm(pb[:, :2] - slot_xy, axis=1) * 1000}
    for i in range(n):
        if not alive[i]:
            continue
        bus.publish("robot-a-pick", "inspect-scan", "event", "place.done",
                    {"env": i, "book_id": spec.book_id, "slot": int(slot[i]), "on_table": bool(on_table[i])})
        if on_table[i]:
            gates[i].place(spec.book_id, slot=int(slot[i]))  # 選び直さない（並行のときは、その間に別の場所が空く）
        else:
            _result(bus, "robot-a-pick", i, spec, None, "rejected", None, 0, "not_on_table")
            alive[i] = False
    on_tray = alive.copy()  # 台に載った本（査定の対象）

    # ---- 点検カメラ: 本へ向けてバーコードを読む ----
    isbn = [None] * n
    up_z = mb[:, 2, 2]
    hold_on = "hold_table" in site.raw
    if not hold_on:  # 裏向きの本は裏返し（未実装）が要るので、届かなくても失敗に数えない（保留棚があれば保留棚へ届く）
        rec["expected_flow"] &= up_z > 0
    if alive[0]:
        cam = cams["barcode"]
        view = max(w, h) + 0.03
        cam.set_pose(pos=(float(pb[0, 0]), float(pb[0, 1]), tb["top"] + tb["camera_height"]),
                     lookat=(float(pb[0, 0]), float(pb[0, 1]), tb["top"]), up=(1.0, 0.0, 0.0))
        # 画角は本が収まるだけ（ズーム）。Genesis のカメラは作った後に画角を変えられないので、作るときの 14° を使う
        rgb, *_ = cam.render()
        rgb = np.asarray(rgb)
        if camera_dirty:
            rgb = dirty_lens(rgb)
            rec["injected"] = "camera_dirty"
            if not hold_on:
                rec["expected_flow"][0] = False  # わざと読めなくした本は、届かないのが正しい（保留棚があれば保留棚へ届く）
        Image.fromarray(rgb).save(out / f"barcode_{spec.book_id}_env0.png")
        isbn[0] = decode_isbn(rgb)
        rec["barcode_view_m"] = view
        # 撮った時点の手先の位置（画像に手が写り込んで読めなかった。20261027T211526 の book-05）
        rec["grippers_at_scan"] = {"a": ra.gripper.get_pos().cpu().numpy()[0], "b": rb.gripper.get_pos().cpu().numpy()[0],
                                   "book": pb[0]}
        bus.publish("inspect-scan", "appraisal", "event", "barcode.read",
                    {"env": 0, "book_id": spec.book_id, "ok": isbn[0] is not None, "isbn": isbn[0],
                     "correct": isbn[0] == spec.isbn13, "source": "camera",
                     **({"injected_fault": "camera_dirty"} if camera_dirty else {})})
    if zone is not None and zone.get("table") == "a":  # 置いて、読み終えた。台を B に空ける
        zone["table"] = None
    rec["t_sim"]["a_on_table"] = world.sim_time
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
            # 点検工程は A へ裏返しを依頼する（修正依頼。inspection.py の流れ）。裏返し（B3）はまだ無いので、A は断り、
            # 人の判断待ちとして台に残す
            bus.publish("inspect-scan", "robot-a-pick", "feedback", "fix_request",
                        {"env": i, "book_id": spec.book_id, "action": "flip", "attempt": 1})
            bus.publish("robot-a-pick", "inspect-scan", "event", "fix.declined",
                        {"env": i, "book_id": spec.book_id, "action": "flip", "reason": "flip_not_implemented"})
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
    # 査定が終わった（行き先が決まった）ことを A へ知らせる。A はこれを受けてから次の本へ進む（台に載らなかった本は査定しない）
    for i in range(n):
        if on_tray[i]:
            dest, why = gates[i].outcome(spec.book_id, hold_table=hold_on)
            bus.publish("appraisal", "robot-a-pick", "event", "appraisal.done",
                        {"env": i, "book_id": spec.book_id, "destination": dest, "reason": why})
    rec["t_sim"]["appraised"] = world.sim_time
    st.update(rec=rec, alive=alive, isbn=isbn, decisions=decisions, skip_b=False)


def left_on_tray_after_stop(stopped: bool, cleared: bool, on_tray: bool) -> bool:
    """届かないのが正しい本か（オブザーバが箱を止めた後、A が台に置いた本に B が触れない）。A は並行して動くので、止まる前に
    取り出して台に置いた本がありうる（20261101T033205-3697ce の book-10）。A が台に置けなかった本（not_on_table）は
    失敗のまま数える（ACT の通し試験 20261101T121659-b9808a で、A の失敗 3 冊が分母から抜けて succeeded になった）"""
    return bool(stopped and not cleared and on_tray)


def process_book_b(st, k, idx, books, pile, site, shelf, keeper, keeper_ent, tb, bn, station, rbs, ra, rb, world, cams,
                   gates, bus, obs, inv, run_id, out, catalog, shelf_mode: str = "flat", zone: dict | None = None) -> dict:
    """B の工程（査定が済んだ本を許可を確かめて取り、本棚か廃棄箱へ運ぶ）。g_process_a の st を受け取る。
    zone を渡すときは、呼ぶ前に zone["table"] = "b" にしておく。台から離れたら空ける"""
    n = world.n
    book, spec = books[idx], pile[idx]
    w, h, t = spec.size_m
    rec, alive, isbn, decisions = st["rec"], st["alive"], st["isbn"], st["decisions"]

    def leave_table() -> None:
        if zone is not None and zone.get("table") == "b":
            zone["table"] = None

    rec.setdefault("t_sim", {})["b_start"] = world.sim_time
    # ---- B: 査定できなかった本は、保留棚に積めるかを確かめる（取る前に。B14） ----
    ht = site.raw.get("hold_table")
    to_hold_ok = np.zeros(n, bool)
    hold_plans: list = [None] * n
    for i in range(n):
        it = gates[i].items.get(spec.book_id)
        if ht is None or obs.is_stopped(i) or it is None or it.state != HELD or it.hold_reason not in HOLDABLE:
            continue
        plan = hold_plans[i] = plan_stack(ht["center"], ht["size"], ht["top"], HOLD_TOP.get(i), t, HOLD_RULE[0])
        if plan.ok:
            gates[i].release_to_hold(spec.book_id)
            to_hold_ok[i] = True
        else:
            # 積むと崩れるおそれがある・高さの上限。取らずに台に残す（オブザーバが箱ごと止める）
            bus.publish("robot-b-sort", "observer", "event", "escalation",
                        {"env": i, "book_id": spec.book_id, "reason": FULL_REASON, "why": plan.reason})
            gates[i].on_escalated(spec.book_id, FULL_REASON)
            rec["expected_flow"][i] = False
    if ht is not None:
        rec["hold_plan"] = [None if pl is None else dataclasses.asdict(pl) for pl in hold_plans]
    # ---- B: 許可を確かめて取る ----
    cleared = np.zeros(n, bool)
    dest = [None] * n
    for i in range(n):
        if not (alive[i] or to_hold_ok[i]):
            continue
        rel = {it.book_id: it for it in gates[i].pickable()}
        if spec.book_id in rel:
            dest[i] = gates[i].take(spec.book_id, rel[spec.book_id].clearance_id)
            cleared[i] = True
    for i in range(n):
        it = gates[i].items.get(spec.book_id)
        if alive[i] and not cleared[i] and it is not None and it.hold_reason in ("shelf_full", FULL_REASON):
            rec["expected_flow"][i] = False  # 本棚に空きが無くて台に残した本は、届かないのが正しい（--shelf-capacity）
        if left_on_tray_after_stop(obs.is_stopped(i), cleared[i], rec["a"]["on_table"][i]):
            rec["expected_flow"][i] = False
    rec["cleared"] = cleared
    rec["destination"] = dest
    if not cleared.any():
        leave_table()
        return rec
    _phase("b_home_before_pick")
    run_generators(world, [g_home(rb, 120)])  # 待機の形から HOME へ（取る動きは HOME から始める）
    rb.freeze(~cleared)
    o_b: dict = {}
    standing = shelf_mode == "standing" and any(dd == "shelf" for dd in dest)
    off = shelve_grip_offset(spec, HAND, keeper) if standing else 0.0
    rec["b_grip_off_mm"] = None if off is None else off * 1000
    _phase("b_pick")
    run_generators(world, [g_pick(rb, book, spec, o_b, surface_z=tb["top"], carry_bottom_z=tb["top"] + 0.10,
                                  plate_mode=PLATE_MODE, claw_mode=CLAW_MODE, grip_off=off or 0.0)])
    grip_b = o_b["grip"]
    rec["b_pick"] = grip_b.rec
    delivered = np.zeros(n, bool)
    # 持ち上げられなかった本は運ばない（手だけで運ぶと、行き先は台の上の本の位置から計算されるので、腕が大きく振れて
    # 廃棄箱に当たった。20261101T022539-b34495 の book-09）。手を開いて HOME へ戻し、本は台に残す（届かなかった扱い）
    lifted = np.asarray(grip_b.rec.get("lifted", np.ones(n, bool)), bool)
    failed = cleared & ~lifted
    if failed.any():
        bus.publish("robot-b-sort", "observer", "event", "escalation",
                    {"env": int(np.argmax(failed)), "book_id": spec.book_id, "reason": "b_pick_failed"})
        _phase("b_pick_failed_release")
        rb.unfreeze()
        rb.freeze(~failed)
        FL, FR = rb.i("finger_l_joint"), rb.i("finger_r_joint")
        run_generators(world, [g_joints(rb, [FL, FR], np.full((n, 2), rb.design.finger_open_m), 40)])
        g_up = rb.gripper.get_pos().cpu().numpy()
        g_up[:, 2] += 0.10
        run_generators(world, [g_cart_chain(rb, g_up, rb.gripper.get_quat().cpu().numpy(), 60, n_way=6, profile=min_jerk)])
        run_generators(world, [g_home(rb, 120)])
        rb.unfreeze()
        rb.freeze(~cleared | failed)
    rec["b_pick_failed"] = failed
    to_shelf = np.array([d == "shelf" for d in dest]) & cleared & ~failed
    to_bin = np.array([d == "discard" for d in dest]) & cleared & ~failed
    if to_bin.any():
        rb.unfreeze()
        rb.freeze(~to_bin)
        # 関節空間で運ぶ（手先の直線で運ぶと、台から廃棄箱へ向きを変える途中で本を落とした。20261023T010753）
        mv: dict = {}
        _phase("bin_move")
        # 置く位置: 廃棄箱の中心から、どの本も同じだけ B と反対の側（-x）へずらす（BIN_DROP_SHIFT。B15）
        drop = np.tile(np.asarray(bn["center"], float) + np.array([-BIN_DROP_SHIFT, 0.0]), (n, 1))
        rec["bin_drop_xy"] = drop
        g_lift_and_swing(rb, grip_b, drop, np.full(n, SWING_BOTTOM_Z), world, mv)
        leave_table()  # 本を持って廃棄箱の上へ移った
        # 廃棄箱の本は積み重なる。置く面は、これまでに入れた本の上面の最も高い所（実測）
        surf = np.array([BIN_TOP.get(i, bn["floor"]) for i in range(n)])
        _phase("bin_place")
        # 放すときは本棚と同じく、今の位置から少しだけ開いて真上へ抜く（全開にすると爪の側の指が B の側の壁へ寄る）
        run_generators(world, [g_place_flat(rb, grip_b, surf, o_b, release_open=BIN_RELEASE_OPEN)])
        _phase("bin_home")
        # 廃棄箱の上から関節空間で HOME へ戻すと、背の外へ出た爪が B の側の壁（+x）の上の縁をこすった
        # （20261101T011312-c1e311。手先は廃棄箱の中心の真上・高さ 0.33 m）。0.45 m まで上げても、HOME への途中で手が下がり、
        # 指と爪が査定台の側の壁（-y）の上の縁に触れた（20261101T012110-5de487）。手先を HOME_LIFT_Z まで真上へ上げてから戻す
        g_up = rb.gripper.get_pos().cpu().numpy()
        g_up[:, 2] = np.maximum(g_up[:, 2], HOME_LIFT_Z)
        run_generators(world, [g_cart_chain(rb, g_up, rb.gripper.get_quat().cpu().numpy(), 60, n_way=6, profile=min_jerk)])
        run_generators(world, [g_home(rb, 120)])
        pb, mb = book_frame(book)
        in_bin = ((np.abs(pb[:, 0] - bn["center"][0]) < bn["inner"][0] / 2) & (np.abs(pb[:, 1] - bn["center"][1]) < bn["inner"][1] / 2)
                  & (pb[:, 2] < surf + t / 2 + 0.006) & (np.abs(mb[:, 2, 2]) > 0.97))
        delivered |= to_bin & in_bin
        for i in np.nonzero(to_bin & in_bin)[0]:
            BIN_TOP[int(i)] = max(BIN_TOP.get(int(i), bn["floor"]), float(pb[i, 2] + t / 2))
        rec["bin"] = {"move": mv, "place": o_b.get("place", {}), "surface_z": surf, "in_bin_flat": in_bin,
                      "book_pos": pb, "book_x_axis_xy": mb[:, :2, 0]}
    to_hold = np.array([d == HOLD for d in dest]) & cleared & ~failed
    if to_hold.any():
        rec["hold"] = place_on_hold(rb, grip_b, book, spec, ht, world, to_hold, hold_plans, o_b, leave_table)
        delivered |= to_hold & rec["hold"]["in_hold_flat"]
    if to_shelf.any():
        rb.unfreeze()
        rb.freeze(~to_shelf)
        place = yield_shelf if shelf_mode == "standing" else yield_shelf_flat
        leave_table()  # このあと台車で本棚へ向かう
        rec["shelf"] = place(rb, grip_b, spec, book, shelf, keeper, keeper_ent, station, rbs, world, cams, bus, inv,
                             run_id, out, to_shelf, dest, isbn, decisions, k)
        delivered |= to_shelf & rec["shelf"]["ok"]
    rb.unfreeze()
    if not (cleared & ~delivered & ~failed).any():  # 本を持ったまま戻ったとき（本棚に空きが無かったなど）は、たたまない
        run_generators(world, [g_b_wait(rb)])  # 待機位置で腕をたたむ（点検カメラに写り込まないように）
    for i in range(n):
        if not cleared[i]:
            continue
        gates[i].on_delivered(spec.book_id, bool(delivered[i]))
        if dest[i] == HOLD:  # 査定できなかった本。保留棚へ届いたら成功（人が判断する）。査定額は無い
            why = gates[i].items[spec.book_id].hold_cause
            _result(bus, "robot-b-sort", i, spec, isbn[i], "succeeded" if delivered[i] else "rejected", HOLD, 0,
                    why if delivered[i] else f"not_delivered_to_{HOLD}")
            continue
        d = decisions[i]
        _result(bus, "robot-b-sort", i, spec, isbn[i], "succeeded" if delivered[i] else "rejected", dest[i],
                d.value_jpy, d.reason if delivered[i] else f"not_delivered_to_{dest[i]}")
    leave_table()
    rec["delivered"] = delivered
    rec["t_sim"]["b_done"] = world.sim_time
    return rec


def process_book(k, idx, books, pile, site, shelf, keeper, keeper_ent, tb, bn, station, rbs, ra, rb, world, cams, gates,
                 bus, obs, inv, run_id, out, catalog, shelf_mode: str = "flat", camera_dirty: bool = False) -> dict:
    """1 冊を A → B の順に処理する（--sequential のとき。並行のときは run_parallel）。"""
    st: dict = {}
    run_generators(world, [g_process_a(k, idx, books, pile, site, shelf, keeper, keeper_ent, tb, bn, station, rbs, ra, rb,
                                       world, cams, gates, bus, obs, inv, run_id, out, catalog, shelf_mode,
                                       camera_dirty=camera_dirty, st=st)])
    if st.get("skip_b"):
        return st["rec"]
    return process_book_b(st, k, idx, books, pile, site, shelf, keeper, keeper_ent, tb, bn, station, rbs, ra, rb, world,
                          cams, gates, bus, obs, inv, run_id, out, catalog, shelf_mode)



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
    _phase("shelf_tuck")
    tuck = np.concatenate([bp[:, :2], np.zeros((n, 1))], -1) + fwd * 0.25 + left * 0.30
    run_generators(world, [g_move_book(rb, grip, tuck, bottom_z=shelf.stand_h + 0.20)])
    front = shelf.front_pose()
    bus.publish("robot-b-sort", "observer", "event", "base.drive", {"to": "shelf", "shelf_id": shelf.shelf_id, "pose": front})
    _phase("drive_to_shelf")
    # station_approach まで向きを変えずに下がり、そこで向きを変えてから本棚へ（B15。site.b_route）
    route = b_route(rbs, rb.target[0, rb.base], front)
    rec["route"] = route
    for wp in route[1:]:
        run_generators(world, [g_drive(rb, wp, rbs["max_speed"], rbs["max_yaw_rate"], DT)])
    _phase("at_shelf")
    bp = rb.base_pose()
    rec["base_err_m"] = np.linalg.norm(bp[:, :2] - np.array(front[:2]), axis=1)
    rec["still_held_after_drive"] = touching(rb, book)


def _drive_to_station(rb, station, rbs, world) -> None:
    """本棚から待機位置へ戻る。待機位置の手前の点（station_approach）を通る（まっすぐ戻ると廃棄箱に当たることがある）。"""
    _phase("drive_to_station")
    # 行きと同じ道を逆に: station_approach まで本棚を向いたまま進み、そこで向きを変えてから待機位置へ（B15）
    for wp in b_route(rbs, rb.target[0, rb.base], station)[1:]:
        run_generators(world, [g_drive(rb, wp, rbs["max_speed"], rbs["max_yaw_rate"], DT)])


BIN_TOP: dict = {}  # 環境ごとの、廃棄箱に入れた本の上面の最も高い所（世界の z。置いた後に測った値）
HOLD_TOP: dict = {}  # 環境ごとの、保留棚のいちばん上の本（hold.TopBook。置いた後に測った値）
HOLD_RULE: list = [StackRule()]  # 保留棚に積むときの判定（--hold-max-stack-m で高さの上限を変える）
# 査定台から廃棄箱・保留棚へ運ぶときの本の底の高さ（床から）。台から 0.10 m 上のまま関節空間で振ると、途中で手が下がり、
# 爪が査定台に・指が廃棄箱の壁に触れた（20261101T003116-6d04d8 の book-09。B15）。先に真上へ持ち上げる。
# 0.30 m（壁の上 0.10 m）でも、関節空間の途中で手が約 0.10 m 下がって爪が壁の上の縁に触れた（20261101T012110-5de487）
SWING_BOTTOM_Z = 0.30
HOME_LIFT_Z = 0.55  # 廃棄箱に置いた後、HOME へ戻す前に手先を上げる高さ（床から）
BIN_RELEASE_OPEN = 0.010  # 廃棄箱で放すときに指を開く量（全開にすると壁に当たる）
# 爪を本の底から抜くと、爪は背の外へ約 0.12 m 出る。B は待機位置（廃棄箱の +x の側）から手を伸ばすので、爪はいつも
# B の側（+x）へ出る。廃棄箱の中心に置いた book-09（幅 148 mm）では、爪が B の側の壁に本の底の高さで当たった
# （20261101T010202-3ddad7）。どの本も同じだけ -x へずらす（本ごとに変えると下の本と揃わず、爪が本と本の間に挟まって
# book-09 を引きずり落とした。20261101T013929-236126）。0.03 m ずらすと、下ろす途中で腕が届く限りに近づいて手が 0.11 m 横へずれ、
# 本を落とした（20261101T014948-374fee）。廃棄の本で一番幅の広い book-09 で、背 + 爪 = 0.133 m < 壁 0.15 m
BIN_DROP_SHIFT = 0.015


def g_lift_and_swing(rb, grip, center_xy, bottom_z, world, out: dict) -> None:
    """持った本を、いまの位置の真上へ SWING_BOTTOM_Z（と bottom_z の高い方）まで持ち上げ、関節空間で center_xy の上へ運ぶ"""
    pb = grip.book.get_pos().cpu().numpy()
    lift = float(max(SWING_BOTTOM_Z, float(np.max(bottom_z))))
    run_generators(world, [g_move_book(rb, grip, pb[:, :2], steps=80, bottom_z=lift)])
    run_generators(world, [g_move_book_joint(rb, grip, center_xy, grip.quat, bottom_z, out=out)])


def place_on_hold(rb, grip, book, spec, ht, world, mask, plans, o_b, leave_table) -> dict:
    """持った本を保留棚へ平らに置く（廃棄箱と同じ置き方）。置く位置は、いちばん上の本の中心（1 冊目は台の中心）。
    置いた後に、平らで台（山）の上にあるかを測り、いちばん上の本（HOLD_TOP）を更新する"""
    n = world.n
    w, h, t = spec.size_m
    rb.unfreeze()
    rb.freeze(~mask)
    tgt = np.array([plans[i].target_xy if mask[i] else ht["center"] for i in range(n)], float)
    surf = np.array([plans[i].surface_z if mask[i] else ht["top"] for i in range(n)], float)
    mv: dict = {}
    _phase("hold_move")
    # 廃棄箱と同じく、真上へ持ち上げてから関節空間で運ぶ。運ぶ本の底は、山の上面の 50 mm 上か、床から 0.30 m の高い方
    g_lift_and_swing(rb, grip, tgt, np.maximum(surf + 0.05, SWING_BOTTOM_Z), world, mv)
    leave_table()  # 本を持って保留棚の上へ移った
    _phase("hold_place")
    # 放すときに指を全開にすると、小口の側の壁が奥へ動いて、下の本の上の本を押した（20261101T033205-3697ce の book-05 は
    # 65 mm・book-03 は 42 mm ずれ、山が階段のようにずれた）。廃棄箱と同じく少しだけ開く
    run_generators(world, [g_place_flat(rb, grip, surf, o_b, release_open=BIN_RELEASE_OPEN)])
    _phase("hold_home")
    run_generators(world, [g_home(rb, 120)])
    pb, mb = book_frame(book)
    tilt = np.degrees(np.arccos(np.clip(np.abs(mb[:, 2, 2]), 0, 1)))
    on_table = ((np.abs(pb[:, 0] - ht["center"][0]) < ht["size"][0] / 2) & (np.abs(pb[:, 1] - ht["center"][1]) < ht["size"][1] / 2))
    level = np.abs(pb[:, 2] - (surf + t / 2)) < 0.006
    ok = mask & on_table & level & (tilt < 3.0)
    support = np.full(n, np.nan)
    for i in np.nonzero(mask)[0]:
        i = int(i)
        below = HOLD_TOP.get(i)  # 置く前のいちばん上の本（1 冊目は台）
        support[i] = (inset_from_edge(pb[i, :2], ht["center"], 0.0, ht["size"]) if below is None
                      else inset_from_edge(pb[i, :2], below.center_xy, below.yaw, below.size_xy))
    for i in np.nonzero(ok)[0]:
        i = int(i)
        HOLD_TOP[i] = TopBook(center_xy=(float(pb[i, 0]), float(pb[i, 1])),
                              yaw=float(math.atan2(mb[i, 1, 0], mb[i, 0, 0])), size_xy=(float(w), float(h)),
                              top_z=float(pb[i, 2] + t / 2), tilt_deg=float(tilt[i]), support_inset_m=float(support[i]))
    rb.unfreeze()
    return {"move": mv, "place": o_b.get("place", {}), "surface_z": surf, "target_xy": tgt, "in_hold_flat": ok,
            "pos_err_mm": np.linalg.norm(pb[:, :2] - tgt, axis=1) * 1000, "tilt_deg": tilt, "support_inset_m": support}
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
