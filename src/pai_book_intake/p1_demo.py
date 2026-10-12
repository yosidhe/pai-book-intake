"""P1 のデモ: 箱から本を取り出して受け渡し台に置き、バーコードを読み、本棚か廃棄箱へ仕分ける（スクリプト制御）。

手順（本 1 冊ごと。平積みの一番上から --books 冊）
  1. robot-a-pick  平積みの一番上を flat_pick で取り、受け渡し台へ置き方の標準どおりに置く → place.done
  2. inspect-flip  置かれた姿勢から自分の成功確率を推定し、0.80 未満なら A へ修正依頼。台の真上のカメラで
                   バーコードを読む（描画は環境 0 だけ。ほかの環境は正解の ISBN を使い、そのことを記録する）
  3. appraisal     ISBN から棚か廃棄かを決める（スタブ）。運搬許可（release.py）を出す
  4. robot-b-sort  許可が出た環境でだけ動く（許可の無い環境では B を止めたまま、本に触れない）。
                   台の上の姿勢から成功確率を推定し、0.80 未満なら inspect-flip へ修正依頼。
                   台から本を取り、許可の行き先（本棚のスロットか廃棄箱）へ置く → book.result
  5. observer      全メッセージを見て、止める条件を判定し、箱ごとの報告書を書く（台に残った本も載せる）
試験用: --withhold-appraisal BOOK_ID で、その本の査定を意図的に出さない。B がその本を持っていかないこと
        （本が台の上から動かない・B の手先が動かない）を受け入れの条件に加える
未実装（P1 の後半以降）: 修正依頼を受けた側のやり直し・裏返し・ざらざらの本の取り出し（試行3 の scoop）・立てた本

使い方: python -m pai_book_intake.p1_demo --n-envs 2 --books 2
出力: /workspace/outputs/p1_demo/<run_id>/（動画・bus.jsonl・観測者の報告書・result.json・バーコード画像）
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from . import layout as L
from .appraisal import appraise
from .books import DEFAULT_CATALOG, PLACEHOLDER_CATALOG, Catalog, grip_plan, load_catalog
from .bus import Bus
from .negotiation import UPSTREAM, BinnedSuccessEstimator, TaskTimer, should_request_fix
from .observer import Observer, ObserverConfig
from .release import ReleaseGate
from .runlog import classify, new_run, save_result
from .vision import decode_isbn

TRIAL = "p1_demo"
CARRY_Z = 0.62  # 運ぶときの手先の高さ（箱の壁・台・棚の背板より十分上）
BIN_RELEASE_H = 0.12  # この高さ（床から。廃棄箱の底板の上面は 0.03）で離す


def default_pile(catalog: Catalog):
    """既定の山。平積みは下から上の順で、上の 2 冊は滑らかな本（P1 の取り出し技能は滑らかな本だけ）。"""
    from .scene import PileConfig

    ids = {b.book_id for b in catalog.books}
    if {"book-03", "book-04", "book-07", "book-01", "book-05", "book-08"} <= ids:
        # 下から: book-07（ざらざら）→ book-04（滑らか・廃棄の正解）→ book-03（滑らか・棚の正解）
        return PileConfig(flat=tuple(catalog.by_id(k) for k in ("book-07", "book-04", "book-03")),
                          standing=tuple(catalog.by_id(k) for k in ("book-01", "book-05", "book-08")))
    if {"ph01", "ph02", "ph03", "ph04", "ph05", "ph06"} <= ids:
        return PileConfig(flat=tuple(catalog.by_id(k) for k in ("ph02", "ph03", "ph01")),
                          standing=tuple(catalog.by_id(k) for k in ("ph04", "ph05", "ph06")))
    return PileConfig(flat=catalog.books[:3], standing=catalog.books[3:6])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--books", type=int, default=2, help="平積みの上から何冊処理するか")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="books_v1.json")
    ap.add_argument("--placeholder", action="store_true", help="カタログの代わりに仮の本を使う")
    ap.add_argument("--barcode-res", type=int, nargs=2, default=[1920, 1440])
    ap.add_argument("--overview-res", type=int, nargs=2, default=[960, 540])
    ap.add_argument("--max-consecutive-failures", type=int, default=3)
    ap.add_argument("--barcode-unreadable-limit", type=int, default=3)
    ap.add_argument("--withhold-appraisal", action="append", default=[], metavar="BOOK_ID",
                    help="試験用: この本の査定を出さない（B が持っていかないことを確かめる）")
    ap.add_argument("--withhold-envs", type=int, nargs="*", default=None,
                    help="査定を出さない環境（省略すると全環境）")
    args = ap.parse_args()

    import genesis as gs

    def _arg(v):  # パスはリポジトリからの相対で記録する（絶対パスを残さない）
        if isinstance(v, Path):
            root = DEFAULT_CATALOG.parents[2]
            return str(v.resolve().relative_to(root)) if v.resolve().is_relative_to(root) else v.name
        return v

    run_id, out, meta = new_run(TRIAL, {k: _arg(v) for k, v in vars(args).items()},
                                extra={"genesis": gs.__version__})
    bus = Bus(run_id, out / "bus.jsonl")
    observer = Observer(bus, ObserverConfig(max_consecutive_failures=args.max_consecutive_failures,
                                            barcode_unreadable_limit=args.barcode_unreadable_limit,
                                            shelf_capacity=L.shelf_capacity()))
    ctx: dict = {"recording": False}
    result: dict = {}
    error: BaseException | None = None
    try:
        result = run(args, out, bus, observer, ctx)
    except BaseException as e:  # 状態 failed として記録してから、そのまま上げ直す
        error = e
        result = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        raise
    finally:
        if ctx["recording"]:
            ctx["line"].overview_cam.stop_recording()
        result["reports"] = [p.name for p in observer.write_reports(out, title=f"{TRIAL} {run_id}")]
        bus.close()
        result["status"] = classify(bool(result.get("acceptance_ok", False)), error)
        save_result(out, meta, result)
        print(f"run_id={run_id} status={result['status']} out={out}")


def run(args, out: Path, bus: Bus, obs: Observer, ctx: dict) -> dict:
    import genesis as gs

    from .scene import SceneConfig, build_line, reset_pile
    from .skills import book_pose, flat_pick, place_by_plunger

    catalog = PLACEHOLDER_CATALOG if args.placeholder else load_catalog(args.catalog)
    pile = default_pile(catalog)
    if args.books > len(pile.flat):
        raise ValueError(f"--books は平積みの冊数（{len(pile.flat)}）以下")
    n = args.n_envs
    rng = np.random.default_rng(args.seed)
    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    cfg = SceneConfig(n_envs=n, barcode_res=tuple(args.barcode_res), overview_res=tuple(args.overview_res))
    t_build = time.time()
    line = ctx["line"] = build_line(cfg, pile, out)
    build_s = time.time() - t_build
    world = line.world
    bus.set_sim_clock(lambda: world.sim_time)  # ここからバスの時刻はシミュレーションの時刻
    est = BinnedSuccessEstimator()
    timers = [TaskTimer() for _ in range(n)]
    ra, rb = line.robot_a, line.robot_b

    reset_pile(line, rng)
    world.step(100)  # 両方のロボットに待機姿勢を指令し続けながら落ち着かせる
    bus.publish("observer", "*", "note", "run.start",
                {"n_envs": n, "books": [b.spec.book_id for b in line.books], "handoff": "static_table",
                 "placement_standard": {"overhang_ratio": L.OVERHANG_RATIO, "overhang_min_m": L.OVERHANG_MIN,
                                        "overhang_max_m": L.OVERHANG_MAX, "tol_overhang_m": L.TOL_OVERHANG,
                                        "tol_yaw_deg": math.degrees(L.TOL_YAW), "tol_lateral_m": L.TOL_LATERAL}})
    line.overview_cam.start_recording(save_to_filename=str(out / "overview_env0.mp4"), fps=25)
    ctx["recording"] = True
    t_sim0 = time.time()

    alive = np.ones(n, bool)
    # 運搬許可は環境（箱）ごと。いまの受け渡し台は 1 冊しか置けない（スロット 1）
    gates = [ReleaseGate(bus, catalog, n_slots=1, shelf_capacity=L.shelf_capacity(), env=i) for i in range(n)]
    withhold_envs = set(range(n)) if args.withhold_envs is None else set(args.withhold_envs)
    withheld: list[dict] = []  # 試験用に査定を出さなかった本（環境ごと）。最後に、動いていないことを確かめる
    per_book = []
    flat = [b for b in line.books if b.role == "flat"]
    for k in range(args.books):
        body = flat[-1 - k]  # まだ残っている平積みの一番上
        spec = body.spec
        if body.spec.surface != "smooth":
            raise NotImplementedError(f"{spec.book_id}: ざらざらの本の取り出し（scoop）は P1 後半で本体に入れる")
        for i in range(n):
            if alive[i] and obs.is_stopped(i):
                alive[i] = False
            if alive[i] and gates[i].free_slot() is None:
                # 台の上の本に運搬許可が出ていない（台が空かない）。A は次の本を置けないので、この箱はここで待つ
                bus.publish("robot-a-pick", "observer", "event", "line.waiting",
                            {"env": int(i), "reason": "tray_occupied_without_release",
                             "on_tray": [it.book_id for it in gates[i].remaining()]})
                alive[i] = False
        if not alive.any():
            break
        rec = {"book_id": spec.book_id, "alive_at_start": alive.copy()}
        ra.freeze(~alive)  # 止まった箱では A も動かさない（台の上の本にぶつけない）

        # ---- 1. robot-a-pick ----
        t0 = world.sim_time
        plan = grip_plan(spec)  # つかむ前の判断は推定重さだけで決める
        bus.publish("robot-a-pick", "inspect-flip", "event", "pick.plan", {"book_id": spec.book_id, **plan})
        if not plan["suction_can_lift"]:
            raise NotImplementedError(f"{spec.book_id}: 推定重さが吸着の上限を超える（pinch での取り出しは未実装）")
        ra.set_grip_limit(plan["grip_limit_n"])
        rb.set_grip_limit(plan["grip_limit_n"])
        pick = flat_pick(ra, body)
        hp = L.handoff_pose(spec.width_m, spec.thickness_m)
        tgt = np.tile(hp.center_world, (n, 1))
        place = place_by_plunger(ra, body, tgt, np.full(n, hp.width_axis_yaw_world), np.full(n, L.TABLE_TOP), CARRY_Z)
        ra.go_home(120)
        world.step(30)
        ra.unfreeze()
        t_a = world.sim_time - t0
        c, yaw, up_z = book_pose(body)
        checks = [L.check_placement(c[i], yaw[i], *spec.size_m, up_z=up_z[i]) for i in range(n)]
        rec.update(pick=pick.per_env(), place=place.per_env(),
                   placement=[{**ch.features(), "within_tolerance": ch.within_tolerance,
                               "overhang_mm": ch.overhang_m * 1000} for ch in checks], t_a_s=t_a)
        for i in np.nonzero(alive)[0]:
            bus.publish("robot-a-pick", "inspect-flip", "event", "place.done",
                        {"env": int(i), "book_id": spec.book_id, "held": bool(pick.held[i]),
                         "placement": checks[i].features(), "within_tolerance": checks[i].within_tolerance,
                         "task_time_s": t_a})
            if not checks[i].on_table:
                _book_result(bus, "robot-a-pick", i, spec, None, "rejected", None, 0,
                             "pick_failed" if not pick.held[i] else "not_on_table")
                alive[i] = False
            else:
                gates[i].place(spec.book_id)

        # ---- 2. inspect-flip ----
        t1 = world.sim_time
        isbn_read = [None] * n
        for i in np.nonzero(alive)[0]:
            p = est.estimate(checks[i].features())
            if should_request_fix(p):
                bus.publish("inspect-flip", UPSTREAM["inspect-flip"], "feedback", "fix_request",
                            {"env": int(i), "book_id": spec.book_id, "p_success": p, "features": checks[i].features(),
                             "note": "やり直しは未実装（P1 後半）"})
        rgb = None
        if alive[0]:
            rgb, *_ = line.barcode_cam.render()
            rgb = np.asarray(rgb)
            Image.fromarray(rgb).save(out / f"barcode_{spec.book_id}_env0.png")
            isbn_read[0] = decode_isbn(rgb)
            bus.publish("inspect-flip", "appraisal", "event", "barcode.read",
                        {"env": 0, "book_id": spec.book_id, "ok": isbn_read[0] is not None, "isbn": isbn_read[0],
                         "correct": isbn_read[0] == spec.isbn13, "source": "camera",
                         "res": list(args.barcode_res)})
        for i in np.nonzero(alive)[0]:
            if i == 0:
                continue
            # 環境 0 以外は描画しない（CPU 描画が遅いため）。正解の ISBN を使い、そのことを記録する
            isbn_read[i] = spec.isbn13 if up_z[i] > 0 else None
            bus.publish("inspect-flip", "appraisal", "note", "barcode.unrendered",
                        {"env": int(i), "book_id": spec.book_id, "isbn": isbn_read[i], "source": "ground_truth"})
        for i in np.nonzero(alive)[0]:
            gates[i].on_scan(spec.book_id, isbn_read[i])
            if isbn_read[i] is None:  # 裏返しのやり直しは未実装。読めない本は人の判断待ちにする（運ばない）
                gates[i].on_escalated(spec.book_id, "barcode_unreadable")
        t_insp = world.sim_time - t1
        for i in np.nonzero(alive)[0]:
            if timers[i].record_and_judge("inspect-flip", t_insp):
                bus.publish("inspect-flip", "robot-a-pick", "feedback", "praise", {"env": int(i), "book_id": spec.book_id})

        # ---- 3. appraisal ----
        decisions = [None] * n
        for i in np.nonzero(alive)[0]:
            if spec.book_id in args.withhold_appraisal and i in withhold_envs:
                # 試験用: 査定を出さない（査定が終わっていない本を意図的に作る）
                bus.publish("appraisal", "observer", "note", "appraisal.withheld",
                            {"env": int(i), "book_id": spec.book_id, "purpose": "test_release_gate"})
                continue
            d = appraise(isbn_read[i], catalog)
            decisions[i] = d
            bus.publish("appraisal", "robot-b-sort", "event", "decision",
                        {"env": int(i), "book_id": spec.book_id, "destination": d.destination, "value_jpy": d.value_jpy,
                         "reason": d.reason})
            gates[i].on_appraisal(spec.book_id, d)

        # ---- 4. robot-b-sort（許可が出た環境だけ） ----
        cleared = np.zeros(n, bool)
        clearance = [None] * n
        for i in np.nonzero(alive)[0]:
            released = {it.book_id: it for it in gates[i].pickable()}
            if spec.book_id in released:
                cleared[i], clearance[i] = True, released[spec.book_id]
        t2 = world.sim_time
        c, yaw, up_z = book_pose(body)
        book_before = c.copy()
        b_grip_before, _ = rb.gripper_pose()
        checks_b = [L.check_placement(c[i], yaw[i], *spec.size_m, up_z=up_z[i]) for i in range(n)]
        for i in np.nonzero(cleared)[0]:
            p = est.estimate(checks_b[i].features())
            if should_request_fix(p):
                bus.publish("robot-b-sort", UPSTREAM["robot-b-sort"], "feedback", "fix_request",
                            {"env": int(i), "book_id": spec.book_id, "p_success": p,
                             "features": checks_b[i].features(), "note": "やり直しは未実装（P1 後半）"})
        dest = np.array([None] * n, dtype=object)
        for i in np.nonzero(cleared)[0]:
            dest[i] = gates[i].take(spec.book_id, clearance[i].clearance_id)  # 行き先は許可からだけ取る
        b_max_move = np.zeros(n)  # B の手先が、この工程の始めから最大でどれだけ動いたか（毎歩見る）
        book_max_move = np.zeros(n)  # 本が最大でどれだけ動いたか（毎歩見る）

        def track_b():
            g, _ = rb.gripper_pose()
            np.maximum(b_max_move, np.linalg.norm(g - b_grip_before, axis=1), out=b_max_move)
            np.maximum(book_max_move, np.linalg.norm(body.entity.get_pos().cpu().numpy() - book_before, axis=1),
                       out=book_max_move)

        world.monitors.append(track_b)
        pick_b = place_b = None
        slot_of = np.full(n, -1)
        if cleared.any():
            rb.freeze(~cleared)  # 許可の無い環境では B を止めたまま（本に触れない）
            # B の座標で見た台の上の位置は B の正面 0.6 m 付近。届く範囲が狭いので、上の中継点と持ち上げを低めにする
            pick_b = flat_pick(rb, body, above_dz=0.08, lift_dz=0.10)
            tgt = np.zeros((n, 3))
            surf = np.zeros(n)
            for i in range(n):
                if dest[i] == "shelf":
                    slot_of[i] = gates[i].shelf_used
                    tgt[i] = L.shelf_slot_center(gates[i].shelf_used, spec.thickness_m)
                    surf[i] = L.SHELF_BOARD_TOP
                else:  # 許可の行き先が廃棄箱（許可の無い環境は止めているので、ここの値は使われない）
                    tgt[i] = L.bin_center_world()
                    surf[i] = BIN_RELEASE_H
            axis = np.where(dest == "shelf", L.shelf_axis_yaw_world(), L.bin_axis_yaw_world())
            place_b = place_by_plunger(rb, body, tgt, axis, surf, CARRY_Z)
            rb.go_home(150)
            world.step(60)
            rb.unfreeze()
        else:
            world.step(60)
        world.monitors.remove(track_b)
        t_b = world.sim_time - t2
        c, yaw, up_z = book_pose(body)
        book_move = book_max_move.copy()
        delivered = np.array([bool(cleared[i]) and (L.on_shelf(c[i], spec.thickness_m) if dest[i] == "shelf"
                                                    else L.in_bin(c[i])) for i in range(n)])
        rec.update(pick_b=None if pick_b is None else pick_b.per_env(),
                   place_b=None if place_b is None else place_b.per_env(), cleared=cleared, destination=dest,
                   slot=slot_of, delivered=delivered, t_b_s=t_b, t_inspect_s=t_insp,
                   book_move_during_b_m=book_move, b_gripper_max_move_m=b_max_move,
                   isbn_read_env0=isbn_read[0], isbn_correct_env0=isbn_read[0] == spec.isbn13)
        for i in np.nonzero(alive)[0]:
            if not cleared[i]:
                it = gates[i].items[spec.book_id]
                if spec.book_id in args.withhold_appraisal and i in withhold_envs:
                    withheld.append({"env": int(i), "book_id": spec.book_id, "state": it.state,
                                     "book_move_mm": float(book_move[i] * 1000),
                                     "b_gripper_max_move_mm": float(b_max_move[i] * 1000),
                                     "still_on_table": bool(L.check_placement(c[i], yaw[i], *spec.size_m,
                                                                              up_z=up_z[i]).on_table)})
                continue  # 台に残す。book.result は出さない（処理は終わっていない）
            est.update(checks_b[i].features(), bool(pick_b.held[i]))
            if timers[i].record_and_judge("robot-b-sort", t_b):
                bus.publish("robot-b-sort", "inspect-flip", "feedback", "praise", {"env": int(i), "book_id": spec.book_id})
            d = decisions[i]
            ok = bool(delivered[i])
            gates[i].on_delivered(spec.book_id, ok)
            reason = "" if ok else ("pick_failed_at_table" if not pick_b.held[i] else f"not_in_{dest[i]}")
            _book_result(bus, "robot-b-sort", i, spec, isbn_read[i], "succeeded" if ok else "rejected", dest[i],
                         d.value_jpy, reason or d.reason)
            if not ok:
                alive[i] = False
        per_book.append(rec)

    line.overview_cam.stop_recording()
    ctx["recording"] = False
    wall_sim = time.time() - t_sim0
    for i, g in enumerate(gates):
        for it in g.remaining():
            bus.publish("appraisal", "observer", "event", "tray.remaining",
                        {"env": i, "book_id": it.book_id, "state": it.state, "reason": it.hold_reason})
    n_ok = [sum(1 for b in obs.box(i).books if b["outcome"] == "succeeded") for i in range(n)]
    # 受け入れ: 査定を出さなかった本以外は全部届いた。出さなかった本は、B が触れず（手先 5 mm 未満）台の上に残った
    processed = [f.spec.book_id for f in flat[len(flat) - args.books:]]
    n_withheld_expected = sum(1 for b in args.withhold_appraisal if b in processed) * len(withhold_envs)
    expected = [args.books - sum(1 for w in withheld if w["env"] == i) for i in range(n)]
    withheld_ok = (len(withheld) == n_withheld_expected
                   and all(w["still_on_table"] and w["book_move_mm"] < 5 and w["b_gripper_max_move_mm"] < 5
                           for w in withheld))
    acceptance_ok = all(n_ok[i] == expected[i] for i in range(n)) and withheld_ok
    result = {
        "criterion": "p1_scripted_pass",
        "acceptance": f"全環境で、査定を出した本は {args.books} 冊のうち全部、A が取り出して台に置き、B が許可の行き先へ届けた。"
                      "査定を出さなかった本（--withhold-appraisal）は B が触れず（手先の移動 5 mm 未満）、台の上に残った",
        "acceptance_ok": acceptance_ok,
        "books_succeeded_per_env": n_ok,
        "books_expected_per_env": expected,
        "withheld": withheld,
        "withheld_ok": withheld_ok,
        "left_on_tray": [[{"book_id": it.book_id, "state": it.state, "reason": it.hold_reason} for it in g.remaining()]
                         for g in gates],
        "stop_reasons": [obs.box(i).stop_reason for i in range(n)],
        "barcode_env0": [{"book_id": r["book_id"], "read": r["isbn_read_env0"], "correct": r["isbn_correct_env0"]}
                         for r in per_book],
        "sim_time_s": world.sim_time,
        "wall_build_s": build_s,
        "wall_run_s": wall_sim,
        "handoff": "static_table",
        "per_book": per_book,
    }
    return result


def _book_result(bus, from_task, i, spec, isbn, outcome, dest, value, reason) -> None:
    bus.publish(from_task, "observer", "event", "book.result",
                {"env": int(i), "book_id": spec.book_id, "title": spec.title, "isbn": isbn, "outcome": outcome,
                 "destination": None if dest is None else str(dest), "value_jpy": int(value), "reason": reason})


if __name__ == "__main__":
    main()
