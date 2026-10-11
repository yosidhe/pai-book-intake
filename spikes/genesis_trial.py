"""Genesis 試行: 箱から薄い本を吸着で取り出し、コンベアで運べるかを並列環境で確かめる。

合格条件（計画 v2）
  1. Docker 内の GPU で動く
  2. 同梱のアームモデルを読み込める
  3. 吸着で持ち上げ、80% 以上の環境で落とさない
  4. コンベアで運び、カメラ映像を mp4 に書き出せる
  5. 8GB VRAM で並列 16 環境以上が動く

使い方: python spikes/genesis_trial.py --n-envs 16
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch

import genesis as gs

BOOK_SIZE = (0.13, 0.19, 0.02)  # 文庫より少し大きい本（m）。質量は既定密度で約 0.5 kg
N_BOOKS = 3
BOX_CENTER = (0.55, -0.20)
BOX_INNER = (0.30, 0.34)
BOX_FLOOR_TOP = 0.02
BELT_X = 0.35
BELT_Y = (0.30, 0.95)
BELT_TOP = 0.10
BELT_SPEED = 0.15  # m/s
FINGER_REACH = 0.1034  # hand リンク原点から指先までの距離（指を閉じた状態）
DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)

KP = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
KV = [450, 450, 350, 350, 200, 200, 200, 10, 10]
FORCE_LO = [-87, -87, -87, -87, -12, -12, -12, -100, -100]


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)


def build_scene(n_envs: int, out_dir: Path):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        rigid_options=gs.options.RigidOptions(max_dynamic_constraints=8),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    gray = gs.surfaces.Default(color=(0.6, 0.6, 0.6))
    cardboard = gs.surfaces.Default(color=(0.72, 0.56, 0.38))

    # 段ボール（上が開いた箱）: 底と4枚の壁
    cx, cy = BOX_CENTER
    ix, iy = BOX_INNER
    wall_h, t = 0.16, 0.01
    parts = [
        ((ix + 2 * t, iy + 2 * t, BOX_FLOOR_TOP), (cx, cy, BOX_FLOOR_TOP / 2)),
        ((t, iy + 2 * t, wall_h), (cx - ix / 2 - t / 2, cy, wall_h / 2)),
        ((t, iy + 2 * t, wall_h), (cx + ix / 2 + t / 2, cy, wall_h / 2)),
        ((ix, t, wall_h), (cx, cy - iy / 2 - t / 2, wall_h / 2)),
        ((ix, t, wall_h), (cx, cy + iy / 2 + t / 2, wall_h / 2)),
    ]
    for size, pos in parts:
        scene.add_entity(gs.morphs.Box(size=size, pos=pos, fixed=True), surface=cardboard)

    # コンベア（ベルト面。搬送は本の速度を直接与えるキネマティック方式）
    belt_len = BELT_Y[1] - BELT_Y[0]
    scene.add_entity(
        gs.morphs.Box(size=(0.24, belt_len, BELT_TOP), pos=(BELT_X, sum(BELT_Y) / 2, BELT_TOP / 2), fixed=True),
        surface=gs.surfaces.Default(color=(0.15, 0.15, 0.15)),
    )

    books = []
    for i in range(N_BOOKS):
        books.append(
            scene.add_entity(
                gs.morphs.Box(size=BOOK_SIZE, pos=(cx, cy, BOX_FLOOR_TOP + BOOK_SIZE[2] * (i + 0.5) + 0.002)),
                surface=gs.surfaces.Default(color=(0.2 + 0.25 * i, 0.3, 0.7 - 0.2 * i)),
            )
        )
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    cam_wide = scene.add_camera(res=(960, 540), pos=(1.6, -1.2, 1.2), lookat=(0.45, 0.1, 0.15), fov=40)
    cam_scan = scene.add_camera(res=(320, 240), pos=(BELT_X, 0.80, 0.55), lookat=(BELT_X, 0.80, BELT_TOP), fov=35)

    scene.build(n_envs=n_envs)  # 環境は重ねて置く（描画は env 0 のみ）
    return scene, franka, books, cam_wide, cam_scan, gray


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/spike")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"run_id": run_id, "n_envs": args.n_envs, "seed": args.seed, "genesis": gs.__version__,
                    "torch": torch.__version__, "criteria": {}, "timings_s": {}}
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    mem_before = gpu_mem_used_mib()
    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    scene, franka, books, cam_wide, cam_scan, _ = build_scene(args.n_envs, out_dir)
    result["timings_s"]["build"] = round(time.perf_counter() - t0, 1)
    result["criteria"]["1_gpu"] = {"pass": gs.backend == gs.cuda or str(gs.backend).endswith("cuda"),
                                   "backend": str(gs.backend)}
    result["criteria"]["2_arm_loaded"] = {"pass": franka.n_dofs == 9, "n_dofs": franka.n_dofs}

    n = args.n_envs
    dofs = np.arange(9)
    franka.set_dofs_kp(np.array(KP), dofs)
    franka.set_dofs_kv(np.array(KV), dofs)
    franka.set_dofs_force_range(np.array(FORCE_LO), -np.array(FORCE_LO), dofs)
    home = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.785, 0.0, 0.0])
    franka.set_qpos(np.tile(home, (n, 1)))

    # 本の初期状態を環境ごとにランダム化（箱の中での位置ずれと回転）
    cx, cy = BOX_CENTER
    for i, b in enumerate(books):
        dx = rng.uniform(-0.04, 0.04, n)
        dy = rng.uniform(-0.03, 0.03, n)
        yaw = rng.uniform(-math.radians(20), math.radians(20), n)
        z = BOX_FLOOR_TOP + BOOK_SIZE[2] * (i + 0.5) + 0.003 * (i + 1)
        b.set_pos(np.stack([cx + dx, cy + dy, np.full(n, z)], axis=-1))
        b.set_quat(yaw_quat(yaw))
    franka.control_dofs_position(np.tile(home, (n, 1)))  # 待機中もホーム姿勢を保つ（与えないと崩れ落ちる）
    for _ in range(80):
        scene.step()

    hand = franka.get_link("hand")
    top = books[-1]
    cam_wide.start_recording(save_to_filename=str(out_dir / "wide.mp4"), fps=30)

    def qpos_now() -> torch.Tensor:
        return franka.get_qpos()

    def move(target_pos: np.ndarray, steps: int, conveyor: bool = False) -> None:
        q_goal = franka.inverse_kinematics(link=hand, pos=target_pos, quat=np.tile(DOWN_QUAT, (n, 1)))
        q_goal[:, 7:] = 0.0  # 指は閉じたまま（指先を吸着パッドに見立てる）
        q_start = qpos_now()
        for k in range(1, steps + 1):
            a = k / steps
            franka.control_dofs_position(q_start + (q_goal - q_start) * a)
            if conveyor:
                run_conveyor()
            scene.step()

    def run_conveyor() -> None:
        for b in books:
            p = b.get_pos().cpu().numpy()
            on_belt = (np.abs(p[:, 0] - BELT_X) < 0.13) & (p[:, 1] > BELT_Y[0]) & (p[:, 1] < BELT_Y[1]) \
                & (np.abs(p[:, 2] - (BELT_TOP + BOOK_SIZE[2] / 2)) < 0.02)
            idx = np.nonzero(on_belt)[0]
            if len(idx):
                v = np.zeros((len(idx), 6))
                v[:, 1] = BELT_SPEED
                b.set_dofs_velocity(v, envs_idx=idx)

    t_pick = time.perf_counter()
    book_pos0 = top.get_pos().cpu().numpy()
    book_top_z = book_pos0[:, 2] + BOOK_SIZE[2] / 2
    above = book_pos0.copy()
    above[:, 2] = book_top_z + FINGER_REACH + 0.15
    move(above, 120)
    contact = book_pos0.copy()
    contact[:, 2] = book_top_z + FINGER_REACH - 0.015  # 押し付ける（PD のたわみ約 12mm を見込む。接触すればそこで止まる）
    move(contact, 80)
    for _ in range(20):
        scene.step()

    # 吸着: 指先が本の上面中央付近に接していれば、その環境だけウェルド拘束を付ける
    hand_pos = hand.get_pos().cpu().numpy()
    book_now = top.get_pos().cpu().numpy()
    gap = hand_pos[:, 2] - FINGER_REACH - (book_now[:, 2] + BOOK_SIZE[2] / 2)
    xy_err = np.linalg.norm(hand_pos[:, :2] - book_now[:, :2], axis=1)
    # 距離ではなく、指先（吸着パッド見立て）と本の実際の接触で判定する
    finger_links = {franka.get_link('left_finger').idx, franka.get_link('right_finger').idx}
    ci = franka.get_contacts(with_entity=top)
    la, lb = ci['link_a'].cpu().numpy(), ci['link_b'].cpu().numpy()
    valid = ci['valid_mask'].cpu().numpy()
    touch = np.isin(la, list(finger_links)) | np.isin(lb, list(finger_links))
    in_contact = (touch & valid).any(axis=1)
    sealed = in_contact & (xy_err < 0.03)
    sealed_idx = np.nonzero(sealed)[0]
    if len(sealed_idx):
        scene.sim.rigid_solver.add_weld_constraint(top.base_link_idx, hand.idx, envs_idx=sealed_idx)

    lift = book_now.copy()
    lift[:, 2] = book_top_z + FINGER_REACH + 0.30
    move(lift, 120)
    lifted_z = top.get_pos().cpu().numpy()[:, 2]
    lifted = lifted_z > book_pos0[:, 2] + 0.20

    # コンベアの始点へ運ぶ
    via = np.tile([0.45, 0.20, BELT_TOP + BOOK_SIZE[2] + FINGER_REACH + 0.25], (n, 1))
    move(via, 120)
    place = np.tile([BELT_X, BELT_Y[0] + 0.12, BELT_TOP + BOOK_SIZE[2] + FINGER_REACH + 0.01], (n, 1))
    move(place, 120)
    carried = np.linalg.norm(top.get_pos().cpu().numpy()[:, :2] - place[:, :2], axis=1) < 0.05
    if len(sealed_idx):
        scene.sim.rigid_solver.delete_weld_constraint(top.base_link_idx, hand.idx, envs_idx=sealed_idx)
    result["timings_s"]["pick_place"] = round(time.perf_counter() - t_pick, 1)

    # 退避しながらコンベアで運ぶ
    t_conv = time.perf_counter()
    retreat = place.copy()
    retreat[:, 2] += 0.25
    move(retreat, 60, conveyor=True)
    y_start = top.get_pos().cpu().numpy()[:, 1]
    conv_steps = 450
    for _ in range(conv_steps):
        run_conveyor()
        scene.step()
    y_end = top.get_pos().cpu().numpy()[:, 1]
    conveyed = (y_end - y_start) > 0.25
    result["timings_s"]["conveyor"] = round(time.perf_counter() - t_conv, 1)
    cam_wide.stop_recording()

    # スキャナカメラ（コンベア上の真上）の描画速度を測る
    t_r = time.perf_counter()
    frames = []
    for _ in range(20):
        rgb, *_ = cam_scan.render()
        frames.append(rgb)
    render_ms = (time.perf_counter() - t_r) / 20 * 1000
    import imageio.v3 as iio
    iio.imwrite(out_dir / "scan_last.png", np.asarray(frames[-1]))

    mem_after = gpu_mem_used_mib()
    video = out_dir / "wide.mp4"
    ok_pick = lifted & carried
    result["per_env"] = {
        "sealed": sealed.tolist(), "lifted": lifted.tolist(), "carried": carried.tolist(),
        "in_contact": in_contact.tolist(), "conveyed": conveyed.tolist(), "gap_mm": np.round(gap * 1000, 1).tolist(),
        "xy_err_mm": np.round(xy_err * 1000, 1).tolist(),
    }
    result["criteria"]["3_suction_pick"] = {"pass": bool(ok_pick.mean() >= 0.8),
                                            "success_rate": round(float(ok_pick.mean()), 3)}
    result["criteria"]["4_conveyor_video"] = {
        "pass": bool(video.exists() and video.stat().st_size > 0 and conveyed[ok_pick].mean() >= 0.8 if ok_pick.any() else False),
        "conveyed_rate_of_picked": round(float(conveyed[ok_pick].mean()), 3) if ok_pick.any() else 0.0,
        "video_bytes": video.stat().st_size if video.exists() else 0,
        "scan_render_ms": round(render_ms, 1),
        "belt_speed_cmd_mps": BELT_SPEED,
        "belt_speed_effective_mps": round(float(np.median((y_end - y_start)[ok_pick]) / (conv_steps * 0.01)), 3) if ok_pick.any() else 0.0,
    }
    result["criteria"]["5_parallel_vram"] = {
        "pass": bool(n >= 16 and mem_after is not None and mem_after < 8000),
        "gpu_mem_used_mib_before": mem_before, "gpu_mem_used_mib_after": mem_after,
    }
    result["timings_s"]["total"] = round(time.perf_counter() - t0, 1)
    result["all_pass"] = all(c["pass"] for c in result["criteria"].values())
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
