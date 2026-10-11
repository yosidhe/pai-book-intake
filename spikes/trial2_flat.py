"""試行2-6: 平置きで積まれた本の一番上を、吸盤で浮かせてからグリッパで挟んで持ち上げる。

手順: 指を最大に開き、吸盤を伸ばす → 本の上面中央に吸盤を当てる → 吸着 ON → 吸盤を引き込む（本が指の間に入る）
      → 指を閉じて本の幅方向を挟む → 腕を持ち上げる → 1 秒保持
判定: 持ち上げ後、本が初期位置より 0.15 m 以上高く、指の間に挟まっている（指の開き ≒ 本の幅）

使い方: python spikes/trial2_flat.py --surface smooth --n-envs 16
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import genesis as gs
import trial2_common as C

BOOK = (0.13, 0.19, 0.02)  # 幅(x)・長さ(y)・厚み(z)
N_BOOKS = 3
BOX_C = (0.55, 0.0)
BOX_IN = (0.42, 0.42)
FLOOR = 0.02


def build(n_envs: int, surface: str, out, rounded: bool = False):
    scene = C.make_scene(n_envs)
    scene.add_entity(gs.morphs.Plane())
    card = gs.surfaces.Default(color=(0.72, 0.56, 0.38))
    cx, cy = BOX_C
    ix, iy = BOX_IN
    h, t = 0.16, 0.01
    for size, pos in [
        ((ix + 2 * t, iy + 2 * t, FLOOR), (cx, cy, FLOOR / 2)),
        ((t, iy + 2 * t, h), (cx - ix / 2 - t / 2, cy, h / 2)),
        ((t, iy + 2 * t, h), (cx + ix / 2 + t / 2, cy, h / 2)),
        ((ix, t, h), (cx, cy - iy / 2 - t / 2, h / 2)),
        ((ix, t, h), (cx, cy + iy / 2 + t / 2, h / 2)),
    ]:
        scene.add_entity(gs.morphs.Box(size=size, pos=pos, fixed=True), surface=card)
    s = C.SURFACES[surface]
    books = [
        scene.add_entity(
            gs.morphs.Box(size=BOOK, pos=(cx, cy, FLOOR + BOOK[2] * (i + 0.5) + 0.003 * (i + 1))),
            material=gs.materials.Rigid(friction=s["friction"]),
            surface=gs.surfaces.Default(color=tuple(min(1.0, c + 0.12 * i) for c in s["color"])),
        )
        for i in range(N_BOOKS)
    ]
    robot = scene.add_entity(gs.morphs.MJCF(file=str(C.build_robot_xml(out, rounded))))
    cam = scene.add_camera(res=(960, 540), pos=(1.5, -1.0, 1.0), lookat=(0.5, 0.0, 0.15), fov=40)
    # 接写（箱の手前の壁を透かさないよう、壁より高い位置から本の高さを斜めに見る）
    scene.cam_close = scene.add_camera(res=(960, 540), pos=(cx + 0.05, cy - 0.42, 0.30), lookat=(cx, cy, 0.07), fov=38)
    scene.build(n_envs=n_envs)
    return scene, books, robot, cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", choices=list(C.SURFACES), default="smooth")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suction-n", type=float, default=None, help="吸引力の上限を上書きする（N）")
    args = ap.parse_args()
    run_id, out, meta = C.new_run("trial2_flat", args)
    rng = np.random.default_rng(args.seed)
    C.init_genesis(args.seed)
    scene, books, ent, cam = build(args.n_envs, args.surface, out)
    n = args.n_envs
    r = C.Robot(scene, ent, n)
    r.setup()

    cx, cy = BOX_C
    for i, b in enumerate(books):
        b.set_pos(np.stack([cx + rng.uniform(-0.04, 0.04, n), cy + rng.uniform(-0.03, 0.03, n),
                            np.full(n, FLOOR + BOOK[2] * (i + 0.5) + 0.003 * (i + 1))], -1))
        b.set_quat(C.quat_z(rng.uniform(-math.radians(20), math.radians(20), n)))
    r.step(80)

    top = books[-1]
    suction_n = args.suction_n if args.suction_n is not None else C.SURFACES[args.surface]["suction_max_n"]
    r.suction = suction = C.Suction(r, top, suction_n)
    cam.start_recording(save_to_filename=str(out / "env0.mp4"), fps=25)

    p0 = top.get_pos().cpu().numpy()
    yaw = C.yaw_of(top.get_quat().cpu().numpy())
    quat = C.down_quat_with_finger_axis(yaw)  # 指の開閉軸を本の幅方向(x)に合わせる
    book_top = p0[:, 2] + BOOK[2] / 2

    above = p0.copy()
    above[:, 2] = book_top + C.CUP_TIP_EXTENDED_Z + 0.12
    r.move_arm(above, quat, 120)
    touch = p0.copy()
    touch[:, 2] = book_top + C.CUP_TIP_EXTENDED_Z - 0.012  # PD のたわみを見込んで押し付ける
    r.move_arm(touch, quat, 80)
    r.step(15)
    _, cup_touch = r.touching(top)

    suction.on[:] = True
    r.step(10)
    r.move_joints(C.PLUNGER, np.zeros(n), 80)  # 吸盤を引き込む → 本が指の間へ
    r.step(10)
    z_after_suction = top.get_pos().cpu().numpy()[:, 2]
    raised_by_suction = z_after_suction - p0[:, 2]

    r.move_joints(C.FINGERS, np.zeros((n, 2)), 60)  # 閉じる（本の幅で止まる）
    r.step(20)
    fingers_touch, _ = r.touching(top)

    lift = p0.copy()
    lift[:, 2] = book_top + C.CUP_TIP_EXTENDED_Z + 0.25
    r.move_arm(lift, quat, 120)
    r.step(100)  # 1 秒保持
    cam.stop_recording()

    z_end = top.get_pos().cpu().numpy()[:, 2]
    opening = r.finger_opening()
    fingers_touch_end, _ = r.touching(top)
    held = (z_end - p0[:, 2] > 0.15) & fingers_touch_end & (np.abs(opening - BOOK[0]) < 0.02)
    result = {
        "criterion": "6_flat_pick",
        "surface": args.surface,
        "suction_max_n": suction_n,
        "success_rate": float(held.mean()),
        "pass": bool(held.mean() >= 0.8) if args.surface == "smooth" and args.suction_n is None else None,
        "suction_sealed_rate": float(suction.ever_sealed.mean()),
        "raised_by_suction_median_m": float(np.median(raised_by_suction)),
        "per_env": {
            "cup_touch": cup_touch, "sealed": suction.ever_sealed, "raised_by_suction_m": raised_by_suction,
            "fingers_touch_after_close": fingers_touch, "opening_end_m": opening, "z_gain_m": z_end - p0[:, 2],
            "held": held,
        },
    }
    C.save_result(out, meta, result)


if __name__ == "__main__":
    main()
