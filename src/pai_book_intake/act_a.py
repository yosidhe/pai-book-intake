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
POS_TOL_M = 0.025  # 置き場所からのずれの許容（B6 の間隔 20〜25 mm より小さく）
PARK = (3.0, 3.0)  # 置き終えた本を退ける場所（次の本の邪魔をしない）


# ---------- シーン ----------

def build(n_envs: int, out: Path, site, pile_specs, video: bool = False, batch_dofs: bool = False):
    """batch_dofs: 力の上限を環境ごとに変える（ACT の評価）。評価はスクリプト版も同じ設定で作り、条件をそろえる。"""
    import genesis as gs

    from .book_assets import make_book_assets
    from .p2_line import CARD, GRAY, HAND, _box, _open_box
    from .pinch import PinchRobot, build_pinch_robot_xml
    from .tool import World

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=4),
                     rigid_options=gs.options.RigidOptions(enable_self_collision=False,
                                                           friction_cone=gs.friction_cone.elliptic,
                                                           constraint_timeconst=0.01, batch_dofs_info=batch_dofs),
                     vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
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
        cam = scene.add_camera(res=(960, 540), pos=(1.45, -0.75, 0.95), lookat=(0.45, 0.25, 0.1), fov=50, env_idx=0)
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


def observe(r, book, spec, init: dict, slot_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from .pinch import book_frame

    q = r.e.get_qpos().cpu().numpy()
    f = r.force_now[:, force_idx(r)]
    state = np.concatenate([q, f], 1)
    pb, mb = book_frame(book)
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
        s, e = observe(r, book, spec, init, slot_xy)
        apply_action(r, policy.act(s, e))
        world.step(CTRL_EVERY)
    return {}


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
    e.add_argument("--site", type=Path, default=DEFAULT_SITE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
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
    scene, world, r, books, cam = build(n, out, site, pile, video=getattr(args, "video", False),
                                       batch_dofs=args.cmd == "eval")
    build_s = time.time() - t_build
    jitter = TRAIN_JITTER if args.cmd == "collect" else EVAL_JITTER
    stack(books, pile, site, n, rng, jitter)
    world.step(100)
    slots = np.array(site["appraisal_table"]["slots"])
    policy = None
    steps_act = None
    if args.cmd == "eval" and args.policy == "act":
        policy = ActPolicy(args.ckpt, ensemble=args.ensemble)
        steps_act = int(policy.meta["episode_ctrl_steps"])
    if cam is not None:
        cam.start_recording(save_to_filename=str(out / "overview_env0.mp4"), fps=25)
        world.monitors.append(lambda: world.steps % 4 == 0 and cam.render())
    t0 = time.time()
    per_book = []
    n_books = min(args.books, len(pile))
    for k in range(n_books):
        idx = len(pile) - 1 - k
        book, spec = books[idx], pile[idx]
        slot_xy = slots[rng.integers(0, len(slots), n)]  # 環境ごとに置き場所を選ぶ（rng の使い方は方策によらず同じ）
        init = initial(book, spec)
        rec = Recorder(r, book, spec, init, slot_xy)
        world.monitors.append(rec)
        if policy is None:
            info = scripted_episode(world, r, book, spec, [(books[j], pile[j]) for j in range(idx)], site, slot_xy)
        else:
            policy.reset()
            info = act_episode(world, r, policy, book, spec, init, slot_xy, steps_act)
        world.monitors.remove(rec)
        res = placed_ok(book, spec, site, slot_xy)
        lifted = rec.lift_max > 0.03
        row = {"book_id": spec.book_id, "ctrl_steps": len(rec.A), "lifted": lifted, **res,
               **{k2: v for k2, v in info.items()}}
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
        "pos_tol_mm": POS_TOL_M * 1000,
        "n_envs": n,
        "per_book_ok": ok.sum(1), "per_book_lifted": lifted.sum(1),
        "episodes": int(ok.size), "episodes_ok": int(ok.sum()),
        "box_all_ok": int(ok.all(0).sum()),
        "sim_time_s": world.sim_time, "wall_build_s": build_s, "wall_run_s": wall,
        "per_book": per_book,
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
