"""試行2-9: 箱の中で縦に並んだ本の1冊を、吸盤で上へ引き出してからグリッパで挟んで抜き取る。

並び: 本を立てて（厚み方向 x）一列に詰め、両端を固定の仕切りで支える。対象は中ほどの1冊
手順: 指を開き吸盤を伸ばす → 対象の上端（天）に吸盤を当てる → 吸着 ON → 吸盤を引き込む（本が上へ出る）
      → 指を閉じて厚み方向を挟む → 腕を持ち上げて抜き取る → 1 秒保持
判定: 本の底が隣の本の上端より 0.03 m 以上高く、指が本に触れて挟んでいる

使い方: python spikes/trial2_standing.py --surface smooth --n-envs 16
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import genesis as gs
import trial2_common as C

BOOK = (0.02, 0.13, 0.19)  # 厚み(x)・幅(y)・高さ(z)
N_BOOKS = 7
TARGET = 3
ROW_C = (0.55, 0.0)
FLOOR = 0.02


def build(n_envs: int, surface: str, out):
    scene = C.make_scene(n_envs)
    scene.add_entity(gs.morphs.Plane())
    card = gs.surfaces.Default(color=(0.72, 0.56, 0.38))
    cx, cy = ROW_C
    scene.add_entity(gs.morphs.Box(size=(0.42, 0.30, FLOOR), pos=(cx, cy, FLOOR / 2), fixed=True), surface=card)
    half = (N_BOOKS * (BOOK[0] + 0.004)) / 2 + 0.01
    for sx in (-1, 1):  # 仕切り（箱の中の他の荷物や壁の代わり）
        scene.add_entity(gs.morphs.Box(size=(0.02, 0.20, 0.16), pos=(cx + sx * (half + 0.01), cy, FLOOR + 0.08),
                                       fixed=True), surface=card)
    s = C.SURFACES[surface]
    books = []
    for i in range(N_BOOKS):
        x = cx + (i - (N_BOOKS - 1) / 2) * (BOOK[0] + 0.004)
        color = s["color"] if i == TARGET else (0.55, 0.55, 0.62)
        books.append(scene.add_entity(
            gs.morphs.Box(size=BOOK, pos=(x, cy, FLOOR + BOOK[2] / 2 + 0.002)),
            material=gs.materials.Rigid(friction=s["friction"]),
            surface=gs.surfaces.Default(color=color),
        ))
    robot = scene.add_entity(gs.morphs.MJCF(file=str(C.build_robot_xml(out))))
    cam = scene.add_camera(res=(960, 540), pos=(1.3, -1.1, 0.9), lookat=(0.5, 0.0, 0.25), fov=45)
    scene.build(n_envs=n_envs)
    return scene, books, robot, cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", choices=list(C.SURFACES), default="smooth")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_id, out, meta = C.new_run("trial2_standing", args)
    rng = np.random.default_rng(args.seed)
    C.init_genesis(args.seed)
    scene, books, ent, cam = build(args.n_envs, args.surface, out)
    n = args.n_envs
    r = C.Robot(scene, ent, n)
    r.setup()

    # 初期状態のばらつき: 隙間（1〜4 mm）・列の位置・各本の小さな回転
    cx, cy = ROW_C
    gaps = rng.uniform(0.001, 0.004, n)
    shift_y = rng.uniform(-0.03, 0.03, n)
    for i, b in enumerate(books):
        x = cx + (i - (N_BOOKS - 1) / 2) * (BOOK[0] + gaps)
        b.set_pos(np.stack([x, cy + shift_y + rng.uniform(-0.01, 0.01, n), np.full(n, FLOOR + BOOK[2] / 2 + 0.002)], -1))
        b.set_quat(C.quat_z(rng.uniform(-math.radians(4), math.radians(4), n)))
    r.step(80)

    tgt = books[TARGET]
    r.suction = suction = C.Suction(r, tgt, C.SURFACES[args.surface]["suction_max_n"])
    cam.start_recording(save_to_filename=str(out / "env0.mp4"), fps=25)

    p0 = tgt.get_pos().cpu().numpy()
    yaw = C.yaw_of(tgt.get_quat().cpu().numpy())
    quat = C.down_quat_with_finger_axis(yaw)  # 指の開閉軸を本の厚み方向(x)に合わせる
    top_z = p0[:, 2] + BOOK[2] / 2
    neighbor_top = np.max([b.get_pos().cpu().numpy()[:, 2] for i, b in enumerate(books) if i != TARGET], axis=0) \
        + BOOK[2] / 2

    above = p0.copy()
    above[:, 2] = top_z + C.CUP_TIP_EXTENDED_Z + 0.10
    r.move_arm(above, quat, 120)
    touch = p0.copy()
    touch[:, 2] = top_z + C.CUP_TIP_EXTENDED_Z - 0.012
    r.move_arm(touch, quat, 80)
    r.step(15)
    _, cup_touch = r.touching(tgt)

    suction.on[:] = True
    r.step(10)
    r.move_joints(C.PLUNGER, np.zeros(n), 80)
    r.step(10)
    raised_by_suction = tgt.get_pos().cpu().numpy()[:, 2] - p0[:, 2]

    r.move_joints(C.FINGERS, np.zeros((n, 2)), 60)
    r.step(20)

    lift = p0.copy()
    lift[:, 2] = top_z + C.CUP_TIP_EXTENDED_Z + 0.25
    r.move_arm(lift, quat, 150)
    r.step(100)
    cam.stop_recording()

    pe = tgt.get_pos().cpu().numpy()
    bottom = pe[:, 2] - BOOK[2] / 2
    f_touch, _ = r.touching(tgt)
    out_ok = (bottom - neighbor_top > 0.03) & f_touch
    neighbors_disturbed = np.max([
        np.linalg.norm(b.get_pos().cpu().numpy()[:, :2] - b_p0, axis=1)
        for b, b_p0 in [(b, b.get_pos().cpu().numpy()[:, :2]) for i, b in enumerate(books) if i != TARGET]
    ], axis=0)
    result = {
        "criterion": "9_standing_pick",
        "surface": args.surface,
        "suction_max_n": C.SURFACES[args.surface]["suction_max_n"],
        "success_rate": float(out_ok.mean()),
        "pass": bool(out_ok.mean() >= 0.8) if args.surface == "smooth" else None,
        "suction_sealed_rate": float(suction.ever_sealed.mean()),
        "raised_by_suction_median_m": float(np.median(raised_by_suction)),
        "per_env": {"cup_touch": cup_touch, "sealed": suction.ever_sealed, "raised_by_suction_m": raised_by_suction,
                    "bottom_minus_neighbor_top_m": bottom - neighbor_top, "fingers_touch": f_touch,
                    "success": out_ok, "gap_m": gaps},
    }
    del neighbors_disturbed
    C.save_result(out, meta, result)


if __name__ == "__main__":
    main()
