"""試行2-7: 審査台の縁からはみ出した本の端を横から挟み、手首を 180° 回して裏返して置き直す。

配置: 審査台の手前の縁（ロボット側、-x 側）から本が 4〜6 cm はみ出している（前工程がそう置く想定）
手順: 指を少し開く → 台から離れた中継点を通って、手前（+x 方向）から差し込み、はみ出した端の上下を挟む → 持ち上げる
      → 手首（joint7）を 180° 回す → 台に下ろす → 指を開いて退避
判定: 本の上面が下を向き（本の z 軸の鉛直成分 < -0.9）、台の上に載っている

使い方: python spikes/trial2_flip.py --surface smooth --n-envs 16
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import genesis as gs
import trial2_common as C

BOOK = (0.13, 0.19, 0.02)  # 幅(x, はみ出す方向)・長さ(y)・厚み(z)
TABLE_TOP = 0.40
TABLE_EDGE_X = 0.72
TABLE = ((TABLE_EDGE_X, 1.05), (-0.30, 0.30))
JOINT7_LIMIT = 2.8973
# 手前から +x 方向へ差し込み、指の開閉軸を鉛直にする姿勢（列 = gripper の x, y, z 軸）
R_SIDE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def build(n_envs: int, surface: str, out):
    scene = C.make_scene(n_envs)
    scene.add_entity(gs.morphs.Plane())
    (x0, x1), (y0, y1) = TABLE
    scene.add_entity(gs.morphs.Box(size=(x1 - x0, y1 - y0, TABLE_TOP), pos=((x0 + x1) / 2, (y0 + y1) / 2, TABLE_TOP / 2),
                                   fixed=True), surface=gs.surfaces.Default(color=(0.85, 0.85, 0.82)))
    s = C.SURFACES[surface]
    book = scene.add_entity(
        gs.morphs.Box(size=BOOK, pos=(TABLE_EDGE_X + BOOK[0] / 2 - 0.05, 0.0, TABLE_TOP + BOOK[2] / 2 + 0.002)),
        material=gs.materials.Rigid(friction=s["friction"]),
        surface=gs.surfaces.Default(color=s["color"]),
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=str(C.build_robot_xml(out))))
    cam = scene.add_camera(res=(960, 540), pos=(0.9, -1.3, 0.8), lookat=(0.5, 0.0, 0.35), fov=40)
    scene.build(n_envs=n_envs)
    return scene, book, robot, cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", choices=list(C.SURFACES), default="smooth")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_id, out, meta = C.new_run("trial2_flip", args)
    rng = np.random.default_rng(args.seed)
    C.init_genesis(args.seed)
    scene, book, ent, cam = build(args.n_envs, args.surface, out)
    n = args.n_envs
    r = C.Robot(scene, ent, n)
    r.setup()

    overhang = rng.uniform(0.04, 0.06, n)
    bx = TABLE_EDGE_X + BOOK[0] / 2 - overhang
    by = rng.uniform(-0.05, 0.05, n)
    book.set_pos(np.stack([bx, by, np.full(n, TABLE_TOP + BOOK[2] / 2 + 0.002)], -1))
    book.set_quat(C.quat_z(rng.uniform(-math.radians(8), math.radians(8), n)))
    r.target[:, C.PLUNGER] = 0.0  # 吸盤は引っ込めておく
    r.target[:, C.FINGERS] = 0.045  # 指の間 9 cm（手先の傾きで指先が本の縁に当たらないように）
    r.step(80)
    cam.start_recording(save_to_filename=str(out / "env0.mp4"), fps=25)

    p0 = book.get_pos().cpu().numpy()
    # 差し込む向きを本の向き（yaw）に合わせる。指のパッドが本の縁と平行になる
    yaw = C.yaw_of(book.get_quat().cpu().numpy())
    quat = C.quat_mul(C.quat_z(yaw), np.tile(C.mat_to_quat(R_SIDE), (n, 1)))
    d = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(n)], -1)
    edge = p0 - d * (BOOK[0] / 2)
    overhang_along_d = (TABLE_EDGE_X - edge[:, 0]) / np.cos(yaw)
    # はみ出し量（カメラで測る想定）に合わせ、台の縁の 1.2 cm 手前まで差し込む。パッド幅 6 cm の端での縁のずれも差し引く
    insert = np.clip(overhang_along_d - 0.012 - 0.03 * np.abs(np.tan(yaw)), 0.02, 0.05)
    grip = edge + d * (insert - C.FINGERTIP_Z)[:, None]
    grip[:, 2] = TABLE_TOP + BOOK[2] / 2
    pre = grip - d * 0.10
    via = pre.copy()
    via[:, 2] += 0.15  # 台から離れた中継点
    high = via.copy()
    high[:, 2] += 0.10
    r.move_arm(high, C.down_quat_with_finger_axis(np.zeros(n)), 120)  # まず台の手前上空へ（下向きのまま）
    moved = lambda: np.linalg.norm(book.get_pos().cpu().numpy() - p0, axis=1) > 0.005  # noqa: E731
    moved_high = moved()
    r.move_linear(via, quat, 150)  # 直線で動かしながら横向きへ回す
    via_err = r.correct(via, quat)
    moved_via = moved()
    r.ik_err_max = np.zeros(n)
    r.move_linear(pre, quat, 80)
    pre_err = r.correct(pre, quat)
    ik_pre = r.ik_err_max.copy()
    r.ik_err_max = np.zeros(n)
    r.move_linear(grip, quat, 80)
    grip_err = r.correct(grip, quat)
    ik_grip = r.ik_err_max.copy()
    on_table_before_grip = np.abs(book.get_pos().cpu().numpy()[:, 2] - p0[:, 2]) < 0.005
    r.move_joints(C.FINGERS, np.zeros((n, 2)), 50)
    r.step(20)
    pinched, _ = r.touching(book)

    up = grip.copy()
    up[:, 2] += 0.16
    r.move_linear(up, quat, 100)
    lifted = book.get_pos().cpu().numpy()[:, 2] - p0[:, 2] > 0.10

    q7 = r.target[:, 6].copy()
    goal7 = np.where(q7 < 0, q7 + math.pi, q7 - math.pi)
    reachable = np.abs(goal7) <= JOINT7_LIMIT
    r.move_joints(6, np.clip(goal7, -JOINT7_LIMIT, JOINT7_LIMIT), 150)
    r.step(20)
    up_z_mid = C.quat_to_mat(book.get_quat().cpu().numpy())[:, 2, 2]

    q_flipped = r.gripper.get_quat().cpu().numpy()
    down = r.gripper.get_pos().cpu().numpy()
    down[:, 2] = TABLE_TOP + BOOK[2] / 2 + 0.006
    r.move_linear(down, q_flipped, 100)
    r.move_joints(C.FINGERS, np.full((n, 2), 0.045), 40)
    back = down.copy()
    back = back - d * 0.12
    r.move_linear(back, q_flipped, 80)
    r.step(80)
    cam.stop_recording()

    pe = book.get_pos().cpu().numpy()
    up_z = C.quat_to_mat(book.get_quat().cpu().numpy())[:, 2, 2]
    on_table = np.abs(pe[:, 2] - (TABLE_TOP + BOOK[2] / 2)) < 0.01
    flipped = (up_z < -0.9) & on_table
    result = {
        "criterion": "7_flip",
        "surface": args.surface,
        "success_rate": float(flipped.mean()),
        "pass": bool(flipped.mean() >= 0.8),
        "per_env": {"via_err_mm": via_err * 1000, "moved_after_high": moved_high, "moved_after_via": moved_via, "pre_err_mm": pre_err * 1000, "ik_pre_mm": ik_pre * 1000, "grip_err_mm": grip_err * 1000, "ik_grip_mm": ik_grip * 1000, "book_still_before_grip": on_table_before_grip, "pinched": pinched, "lifted": lifted, "up_z_after_rotate": up_z_mid, "joint7_reachable": reachable, "book_up_z": up_z, "on_table": on_table,
                    "overhang_m": overhang, "flipped": flipped},
    }
    C.save_result(out, meta, result)


if __name__ == "__main__":
    main()
