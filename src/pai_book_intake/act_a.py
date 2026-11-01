"""ロボット A の取り出し（箱の平積みの一番上 → 査定台）を ACT で学ぶための、デモ集めと評価。

工程はスクリプト版（p2_line の A と同じ g_pick → g_move_book → g_place_flat → g_home）を手本にする。
シーンは A・箱・査定台・本だけ（B・本棚・カメラは要らない。描画しないので速い）。

観測と行動（50 Hz。シミュの 2 歩に 1 回）
  observation.state             A の全関節の角度（腕 7・手 7）＋いま効いている力の上限 3（小口の側の壁・背の側の指・板）
  observation.environment_state 本の位置 3・本の x 軸と z 軸 6・寸法 3・取る前の本の位置 3 と背の向き 2・置き場所 2・
                                取る前の本の底の高さ 1（= 20）
  action                        A の全関節の目標 14 ＋ 力の上限 3（= 17）
本の位置と向きは、スクリプト版と同じくシミュの正解値を使う（スクリプト版も book_frame で正解値を読んでいる。条件をそろえる）。
画像は使わない（WSL2 の Docker は CPU 描画で、画像入りのデモを集めると間に合わない。2026-10-28）。

使い方
  デモ集め: python -m pai_book_intake.act_a collect --n-envs 32 --seed 1
  評価:     python -m pai_book_intake.act_a eval --policy scripted --n-envs 50 --seed 1000
            python -m pai_book_intake.act_a eval --policy act --ckpt /workspace/outputs/act_train/<run_id>/final
  比べる:   python -m pai_book_intake.act_a compare <run_a> <run_b>（同じ初期状態の 2 つの run で、片方だけ失敗した本と環境の組）
出力: /workspace/outputs/act_collect/<run_id>/（demos_<k>.npz・result.json）・/workspace/outputs/act_eval/<run_id>/
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

import numpy as np

from .books import DEFAULT_CATALOG, SURFACES, load_catalog
from .geometry import quat_z
from .runlog import classify, new_run, save_result
from .site import DEFAULT_SITE, load_site

DT = 0.01
CTRL_EVERY = 2  # 50 Hz
FORCE_DOFS = ("finger_l_joint", "finger_r_joint", "plate_z_joint")
ENV_DIM = 20
TRAIN_JITTER = (0.02, 10.0)  # 学習: 箱の中の本の初期位置 ±20 mm・±10°
EVAL_JITTER = (0.01, 5.0)  # 評価: ±10 mm・±5°（p2_line と同じ。02_確認事項 B10）
EVAL_PILE = ("book-09", "book-04", "book-10", "book-03")  # p2_line の既定の平積み（下から上）
NUDGE_AFTER_S = 1.5  # --nudge: 取り出しを始めてから本をずらすまで（g_pick が手を本の上へ運ぶ 1.5 s の終わり。下ろし始める前）
POS_TOL_M = 0.025  # 置き場所からのずれの許容（B6 の間隔 20〜25 mm より小さく）
PARK = (3.0, 3.0)  # 置き終えた本を退ける場所（次の本の邪魔をしない）


# ---------- シーン ----------

def build(n_envs: int, out: Path, site, pile_specs, video: bool = False, batch_dofs: bool = False, render_env: int = 0):
    """batch_dofs: 力の上限を環境ごとに変える（ACT の評価）。評価はスクリプト版も同じ設定で作り、条件をそろえる。
    render_env: 動画に撮る環境の番号"""
    import genesis as gs

    from .book_assets import make_book_assets
    from .p2_line import CARD, GRAY, HAND, _box, _open_box
    from .pinch import PinchRobot, build_pinch_robot_xml
    from .tool import World

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=4),
                     rigid_options=gs.options.RigidOptions(enable_self_collision=False,
                                                           friction_cone=gs.friction_cone.elliptic,
                                                           constraint_timeconst=0.01, batch_dofs_info=batch_dofs),
                     vis_options=gs.options.VisOptions(rendered_envs_idx=[render_env], show_world_frame=False),
                     show_viewer=False)
    from .hand_design import ENV_BIT
    scene.add_entity(gs.morphs.Plane(contype=ENV_BIT, conaffinity=ENV_BIT))
    bx = site["box"]
    _open_box(scene, bx["center"], bx["inner"], bx["floor"], bx["wall_h"], bx["wall_t"], CARD)
    tb = site["appraisal_table"]
    _box(scene, (*tb["size"], tb["top"]), (*tb["center"], tb["top"] / 2), GRAY)
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
    cam = None
    if video:
        cam = scene.add_camera(res=(960, 540), pos=(1.45, -0.75, 0.95), lookat=(0.45, 0.25, 0.1), fov=50,
                               env_idx=render_env)
    scene.build(n_envs=n_envs)
    world = World(scene, n_envs, DT)
    rob = PinchRobot(world, ent_a, "robot-a-pick", HAND, base_xy=tuple(ra["base"]))
    rob.setup()
    return scene, world, rob, books, cam


def stack(books, specs, site, n, rng, jitter) -> None:
    """箱の床から平積みにする（下から specs の順）。jitter = (位置 m, 向き deg)。"""
    bx = site["box"]
    dxy, dyaw = jitter
    z = np.full(n, bx["floor"])
    for b, spec in zip(books, specs):
        t = spec.thickness_m
        zc = z + t / 2 + 0.002
        pos = np.stack([bx["center"][0] + rng.uniform(-dxy, dxy, n), bx["center"][1] + rng.uniform(-dxy, dxy, n), zc], -1)
        b.set_pos(pos)
        b.set_quat(quat_z(rng.uniform(-math.radians(dyaw), math.radians(dyaw), n)))
        z = zc + t / 2


def park(book, k: int, n: int, t: float) -> None:
    book.set_pos(np.tile([PARK[0] + 0.4 * k, PARK[1], t / 2 + 0.002], (n, 1)))
    book.set_quat(np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)))


# ---------- 観測と行動 ----------

def force_idx(r) -> list[int]:
    return [r.i(k) for k in FORCE_DOFS]


def face_up_frame(mb: np.ndarray) -> np.ndarray:
    """裏向き（本の z 軸の z 成分 < 0）の本だけ、y 軸と z 軸の符号を反転して、表向きの本と同じ数値にする（本は直方体なので、
    背の向き＝x 軸のまわりに 180° 回しても形は同じ）。p2_line の裏向きは x 軸まわりの 180° で作っている。
    デモに裏向きの本が無く、z 軸の z 成分のばらつき（env_std）が 0.001 なので、裏向きのまま渡すと正規化で約 -2000 になる"""
    m = mb.copy()
    down = m[:, 2, 2] < 0
    m[down, :, 1:3] *= -1.0
    return m


def observe(r, book, spec, init: dict, slot_xy: np.ndarray, face_up: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """face_up: 裏向きの本を表向きと同じ数値にして渡す（ACT に渡すとき。デモの記録では使わない）"""
    from .pinch import book_frame

    q = r.e.get_qpos().cpu().numpy()
    f = r.force_now[:, force_idx(r)]
    state = np.concatenate([q, f], 1)
    pb, mb = book_frame(book)
    if face_up:
        mb = face_up_frame(mb)
    n = r.n
    env = np.concatenate([pb, mb[:, :, 0], mb[:, :, 2], np.tile(spec.size_m, (n, 1)), init["pos"], init["e"], slot_xy,
                          init["bottom_z"][:, None]], 1)
    return state.astype(np.float32), env.astype(np.float32)


def action_now(r) -> np.ndarray:
    return np.concatenate([r.target, r.force_now[:, force_idx(r)]], 1).astype(np.float32)


def apply_action(r, a: np.ndarray) -> None:
    nd = r.target.shape[1]
    r.target[:] = a[:, :nd]
    f = r.force.copy()[None].repeat(r.n, 0)
    f[:, force_idx(r)] = np.clip(a[:, nd:], 0.5, None)
    r.set_forces_per_env(f)


def initial(book, spec) -> dict:
    from .pinch import book_frame

    p0, m0 = book_frame(book)
    e = m0[:, :2, 0].copy()
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return {"pos": p0, "e": e, "bottom_z": p0[:, 2] - spec.thickness_m / 2}


class Recorder:
    """World.monitors に入れて、50 Hz で（観測, その歩みで効いていた行動）を記録する。"""

    def __init__(self, r, book, spec, init, slot_xy):
        self.r, self.book, self.spec, self.init, self.slot_xy = r, book, spec, init, slot_xy
        self.k = 0
        self.prev = observe(r, book, spec, init, slot_xy)
        self.S, self.E, self.A = [], [], []
        self.lift_max = np.zeros(r.n)
        self.z0 = book.get_pos().cpu().numpy()[:, 2]

    def __call__(self) -> None:
        self.k += 1
        z = self.book.get_pos().cpu().numpy()[:, 2]
        self.lift_max = np.maximum(self.lift_max, z - self.z0)
        if self.k % CTRL_EVERY:
            return
        s, e = self.prev
        self.S.append(s)
        self.E.append(e)
        self.A.append(action_now(self.r))
        self.prev = observe(self.r, self.book, self.spec, self.init, self.slot_xy)

    def arrays(self) -> dict:
        return {"state": np.stack(self.S, 1), "env": np.stack(self.E, 1), "action": np.stack(self.A, 1)}


# ---------- 1 冊ぶん ----------

def scripted_episode(world, r, book, spec, lower, site, slot_xy) -> dict:
    from .p2_line import CLAW_MODE, PLATE_MODE, carry_z
    from .pinch import g_home, g_move_book, g_pick, g_place_flat, run_generators

    bx, tb = site["box"], site["appraisal_table"]
    o: dict = {}
    run_generators(world, [g_pick(r, book, spec, o, lower=lower, surface_z=None if lower else bx["floor"],
                                  carry_bottom_z=carry_z(book, spec, bx), plate_mode=PLATE_MODE, claw_mode=CLAW_MODE)])
    grip = o["grip"]
    run_generators(world, [g_move_book(r, grip, slot_xy)])
    run_generators(world, [g_place_flat(r, grip, tb["top"], o)])
    run_generators(world, [g_home(r, 120)])
    return {"lifted": grip.rec["lifted"], "inserted": grip.rec["inserted"]}


def act_episode(world, r, policy, book, spec, init, slot_xy, steps: int) -> dict:
    """方策の行動を 50 Hz で効かせる。steps は制御の回数（デモの最長＋余裕。学習の meta にある）。"""
    for _ in range(steps):
        s, e = observe(r, book, spec, init, slot_xy, face_up=True)
        apply_action(r, policy.act(s, e))
        world.step(CTRL_EVERY)
    return {}


class Nudge:
    """World.monitors に入れる。取り出しを始めてから after_s 秒後に、本を水平に dxy[n, 2] だけずらす（外乱。--nudge）。"""

    def __init__(self, world, book, dxy: np.ndarray, after_s: float = NUDGE_AFTER_S):
        self.world, self.book, self.dxy = world, book, dxy
        self.at = world.steps + int(round(after_s / world.dt))
        self.moved_mm = None

    def __call__(self) -> None:
        if self.world.steps != self.at:
            return
        p0 = self.book.get_pos().cpu().numpy()
        p = p0.copy()
        p[:, :2] += self.dxy
        self.book.set_pos(p)
        self.moved_mm = np.linalg.norm(self.book.get_pos().cpu().numpy()[:, :2] - p0[:, :2], axis=1) * 1000


def nudge_dirs(seed: int, k: int, n: int, mm: float) -> np.ndarray:
    """k 冊目をずらす向き（環境ごと）。本体の乱数とは別の種から作り、方策によらず同じにする。"""
    a = np.random.default_rng([seed, 7, k]).uniform(0.0, 2 * math.pi, n)
    return np.stack([np.cos(a), np.sin(a)], 1) * (mm / 1000)


def placed_ok(book, spec, site, slot_xy) -> dict:
    from .pinch import book_frame

    tb = site["appraisal_table"]
    pb, mb = book_frame(book)
    t = spec.thickness_m
    on_table = (np.abs(pb[:, 2] - (tb["top"] + t / 2)) < 0.006) & (np.abs(mb[:, 2, 2]) > 0.97)
    err = np.linalg.norm(pb[:, :2] - slot_xy, axis=1)
    return {"on_table": on_table, "pos_err_mm": err * 1000, "ok": on_table & (err < POS_TOL_M)}


# ---------- 方策（学習済み ACT） ----------

class ActPolicy:
    """act_train が保存した checkpoint を読み、正規化して ACT に渡す。"""

    def __init__(self, ckpt: Path, device: str = "cuda", ensemble: float | None = None):
        import json

        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.torch = torch
        self.meta = json.loads((Path(ckpt) / "meta.json").read_text())
        st = np.load(Path(ckpt) / "stats.npz")
        self.stats = {k: torch.tensor(st[k], device=device) for k in st.files}
        self.p = ACTPolicy.from_pretrained(str(ckpt))
        if ensemble is not None:
            # 読み込んだ後で毎歩の予測を重ねる形に切り替える（ACTConfig.from_pretrained は保存した config の
            # type の欄を読めずに落ちる。LeRobot 0.4.4）
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            self.p.config.temporal_ensemble_coeff, self.p.config.n_action_steps = ensemble, 1
            self.p.temporal_ensembler = ACTTemporalEnsembler(ensemble, self.p.config.chunk_size)
        self.p.to(device).eval()
        self.device = device

    def reset(self) -> None:
        self.p.reset()

    def act(self, s: np.ndarray, e: np.ndarray) -> np.ndarray:
        t = self.torch
        st = self.stats
        with t.no_grad():
            b = {"observation.state": (t.tensor(s, device=self.device) - st["state_mean"]) / st["state_std"],
                 "observation.environment_state": (t.tensor(e, device=self.device) - st["env_mean"]) / st["env_std"]}
            a = self.p.select_action(b)
            a = a * st["action_std"] + st["action_mean"]
        return a.cpu().numpy()


# ---------- コマンド ----------

def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--n-envs", type=int, default=32)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--books", type=int, default=10, help="平積みの冊数（10 冊の一覧から seed で順を決める）")
    c.add_argument("--pile", default="", help="平積み（下から上の book_id）。省略すると seed で並べる")
    for p in (c,):
        p.add_argument("--site", type=Path, default=DEFAULT_SITE)
    e = sub.add_parser("eval")
    e.add_argument("--policy", choices=("scripted", "act"), required=True)
    e.add_argument("--ckpt", type=Path)
    e.add_argument("--ensemble", type=float, default=None, help="ACT の temporal ensembling の係数（省略で使わない）")
    e.add_argument("--n-envs", type=int, default=50)
    e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--pile", default=",".join(EVAL_PILE))
    e.add_argument("--books", type=int, default=4)
    e.add_argument("--video", action="store_true")
    e.add_argument("--render-env", type=int, default=0, help="動画に撮る環境の番号")
    e.add_argument("--jitter", type=float, nargs=2, metavar=("POS_M", "YAW_DEG"), default=list(EVAL_JITTER),
                   help="箱の中の本の初期位置のばらつき（既定は p2_line と同じ ±10 mm・±5°。デモ集めは 0.02 10）")
    e.add_argument("--nudge", type=float, default=0.0, metavar="MM",
                   help=f"取り出しを始めて {NUDGE_AFTER_S} s 後（手が本の上に着いたところ）に、本を水平に MM ずらす。"
                        "向きは種で決める（方策によらず同じ）")
    e.add_argument("--first-slot-only", action="store_true", help="置き場所を 1 つ目だけにする（p2_line --a-policy act と同じ）")
    e.add_argument("--site", type=Path, default=DEFAULT_SITE)
    m = sub.add_parser("compare")
    m.add_argument("run_a", type=Path, help="act_eval か act_collect の run のフォルダ")
    m.add_argument("run_b", type=Path)
    return ap.parse_args()


def outcomes(res: dict) -> dict:
    """result.json の per_book から、(本, 環境) ごとの成否を {(book_id, env): ok} にする。"""
    return {(b["book_id"], i): bool(ok) for b in res["per_book"] for i, ok in enumerate(b["ok"])}


def compare(ra: dict, rb: dict) -> dict:
    """同じ初期状態の 2 つの run（result.json）で、a だけ失敗した組・b だけ失敗した組・両方失敗した組を出す。
    種・平積み・環境の数・ばらつき・外乱がそろっていなければ止める（条件の違う run を比べない）。"""
    for k in ("pile_bottom_to_top", "n_envs", "jitter"):
        if ra.get(k) != rb.get(k):
            raise ValueError(f"{k} が違う: {ra.get(k)} / {rb.get(k)}")
    if ra["args"]["seed"] != rb["args"]["seed"]:
        raise ValueError(f"seed が違う: {ra['args']['seed']} / {rb['args']['seed']}")
    if bool(ra["args"].get("first_slot_only")) != bool(rb["args"].get("first_slot_only")):
        raise ValueError("first_slot_only が違う")
    if (ra.get("nudge") or None) != (rb.get("nudge") or None):
        raise ValueError(f"nudge が違う: {ra.get('nudge')} / {rb.get('nudge')}")
    pa, pb = outcomes(ra), outcomes(rb)
    common = sorted(set(pa) & set(pb), key=lambda c: (c[1], c[0]))
    return {"a": ra["run_id"], "b": rb["run_id"], "pairs": len(common),
            "a_ok": sum(pa[c] for c in common), "b_ok": sum(pb[c] for c in common),
            "only_a_failed": [list(c) for c in common if not pa[c] and pb[c]],
            "only_b_failed": [list(c) for c in common if pa[c] and not pb[c]],
            "both_failed": [list(c) for c in common if not pa[c] and not pb[c]]}


def choose_slots(rng, slots: np.ndarray, n: int, first_only: bool = False) -> np.ndarray:
    """環境ごとに置き場所を選ぶ（rng の使い方は方策と first_only によらず同じ）"""
    idx = rng.integers(0, len(slots), n)
    return slots[np.zeros(n, int) if first_only else idx]


def main() -> None:
    args = parse_args()
    if args.cmd == "compare":
        import json

        ra, rb = (json.loads((p / "result.json").read_text()) for p in (args.run_a, args.run_b))
        print(json.dumps(compare(ra, rb), ensure_ascii=False))
        return
    import genesis as gs

    trial = "act_collect" if args.cmd == "collect" else "act_eval"
    run_id, out, meta = new_run(trial, {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                extra={"genesis": gs.__version__})
    result: dict = {}
    error = None
    try:
        result = run(args, out)
    except BaseException as ex:
        error = ex
        result = {"error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
        raise
    finally:
        result["status"] = classify(bool(result.get("acceptance_ok", False)), error)
        save_result(out, meta, result)
        print(f"run_id={run_id} status={result['status']} out={out}")


def run(args, out: Path) -> dict:
    import genesis as gs

    site = load_site(args.site)
    catalog = load_catalog(DEFAULT_CATALOG)
    rng = np.random.default_rng(args.seed)
    if args.pile:
        pile_ids = [s for s in args.pile.split(",") if s]
    else:
        pile_ids = [str(b) for b in rng.permutation([b.book_id for b in catalog.books])[:args.books]]
    pile = [catalog.by_id(b) for b in pile_ids]
    n = args.n_envs
    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    t_build = time.time()
    render_env = getattr(args, "render_env", 0)
    if not 0 <= render_env < n:
        raise ValueError(f"--render-env は 0〜{n - 1}")
    scene, world, r, books, cam = build(n, out, site, pile, video=getattr(args, "video", False),
                                       batch_dofs=args.cmd == "eval", render_env=render_env)
    build_s = time.time() - t_build
    jitter = TRAIN_JITTER if args.cmd == "collect" else tuple(args.jitter)
    nudge_mm = getattr(args, "nudge", 0.0)
    stack(books, pile, site, n, rng, jitter)
    world.step(100)
    slots = np.array(site["appraisal_table"]["slots"])
    policy = None
    steps_act = None
    if args.cmd == "eval" and args.policy == "act":
        policy = ActPolicy(args.ckpt, ensemble=args.ensemble)
        steps_act = int(policy.meta["episode_ctrl_steps"])
    if cam is not None:
        cam.start_recording(save_to_filename=str(out / f"overview_env{render_env}.mp4"), fps=25)
        world.monitors.append(lambda: world.steps % 4 == 0 and cam.render())
    t0 = time.time()
    per_book = []
    n_books = min(args.books, len(pile))
    for k in range(n_books):
        idx = len(pile) - 1 - k
        book, spec = books[idx], pile[idx]
        slot_xy = choose_slots(rng, slots, n, first_only=getattr(args, "first_slot_only", False))
        init = initial(book, spec)
        rec = Recorder(r, book, spec, init, slot_xy)
        world.monitors.append(rec)
        nudge = None
        if nudge_mm > 0:
            nudge = Nudge(world, book, nudge_dirs(args.seed, k, n, nudge_mm))
            world.monitors.append(nudge)
        if policy is None:
            info = scripted_episode(world, r, book, spec, [(books[j], pile[j]) for j in range(idx)], site, slot_xy)
        else:
            policy.reset()
            info = act_episode(world, r, policy, book, spec, init, slot_xy, steps_act)
        world.monitors.remove(rec)
        if nudge is not None:
            world.monitors.remove(nudge)
        res = placed_ok(book, spec, site, slot_xy)
        lifted = rec.lift_max > 0.03
        row = {"book_id": spec.book_id, "ctrl_steps": len(rec.A), "lifted": lifted, **res,
               **{k2: v for k2, v in info.items()}}
        if nudge is not None:
            row["nudge_moved_mm"] = nudge.moved_mm
        per_book.append(row)
        print(f"[{k}] {spec.book_id} lifted {int(lifted.sum())}/{n} placed_ok {int(res['ok'].sum())}/{n} "
              f"wall {time.time() - t0:.0f}s", flush=True)
        if args.cmd == "collect":
            arr = rec.arrays()
            np.savez_compressed(out / f"demos_{k:02d}_{spec.book_id}.npz", **arr, success=res["ok"],
                                book_id=spec.book_id, size_m=np.array(spec.size_m))
        park(book, k, n, spec.thickness_m)
        world.step(20)
        # 次の本の前に、腕を待機姿勢に揃える（ACT が待機姿勢に戻りきらなくても、次の本の条件をそろえる）
        r.target[:] = np.tile(r.home, (n, 1))
        r.setup()
    wall = time.time() - t0
    if cam is not None:
        cam.stop_recording()
    ok = np.stack([b["ok"] for b in per_book])  # [冊, 環境]
    lifted = np.stack([b["lifted"] for b in per_book])
    res = {
        "criterion": "act_a_" + args.cmd,
        "pile_bottom_to_top": pile_ids,
        "jitter": {"pos_m": jitter[0], "yaw_deg": jitter[1]},
        "nudge": {"mm": nudge_mm, "after_s": NUDGE_AFTER_S} if nudge_mm > 0 else None,
        "render_env": render_env if cam is not None else None,
        "pos_tol_mm": POS_TOL_M * 1000,
        "n_envs": n,
        "per_book_ok": ok.sum(1), "per_book_lifted": lifted.sum(1),
        "episodes": int(ok.size), "episodes_ok": int(ok.sum()),
        "box_all_ok": int(ok.all(0).sum()),
        "sim_time_s": world.sim_time, "wall_build_s": build_s, "wall_run_s": wall,
        "per_book": per_book,
        "failed_pairs": [[b["book_id"], int(i)] for b in per_book for i in np.nonzero(~b["ok"])[0]],
    }
    if args.cmd == "collect":
        res["acceptance"] = "デモとして使える（置き場所に平らに置けた）エピソードが 1 つ以上ある"
        res["acceptance_ok"] = bool(ok.any())
    else:
        res["policy"] = args.policy
        res["ckpt"] = str(args.ckpt) if args.ckpt else None
        res["acceptance"] = "評価の記録。合否は成功率で読む（この実行自体は最後まで動けば succeeded）"
        res["acceptance_ok"] = True
        res["success_rate"] = float(ok.mean())
        res["stage_rates"] = {"lifted": float(lifted.mean()), "on_table": float(np.stack([b["on_table"] for b in per_book]).mean()),
                              "placed_ok": float(ok.mean())}
    return res


if __name__ == "__main__":
    main()
