"""試行3: ざらざらの本（吸着だけでは持ち上がらない）を、グリッパで挟んで取り出す。吸引は補助。

方法（--method）
  pinch : 指を本の両脇まで下ろし、本の側面を挟む（本を傷つけないよう把持力を抑える）→ 吸引を落下防止に持ち上げる
  scoop : v2 ツール（右の指先に、丸く反った薄い爪）を使う。
          1. 吸盤を本の右寄りに当て、左の指で本の左の側面を押さえる（本が滑らないように）
          2. 吸盤を少し引き込み、本の右の端だけをめくる（本全体は持ち上がらない吸引力でも、端ならてこで上がる）
          3. 右の爪を本の底の高さに置き、上下に小刻みに動かしながら内側へ押し込んで、めくれた隙間に滑り込ませる
          4. 右の爪を上げて本を支え、左右で挟んで持ち上げる
  （吸引力を強めるだけの案は trial2_flat.py --suction-n で試す）

本は箱の中に3冊積み、一番上を取る。--jitter で下の本とのずれの大きさを変える。

使い方: python spikes/trial3_rough.py --method scoop --suction-n 4 --n-envs 16
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import trial2_common as C
import trial2_flat as F

JITTER = {"small": (0.01, 0.01, 5.0), "large": (0.04, 0.03, 20.0)}  # x, y (m), yaw (deg)
SCOOP_OFFSET = 0.035  # 吸盤（グリッパの中心）を本の中心から右へずらす量


def place_books(books, rng, n, jitter: str) -> None:
    jx, jy, jyaw = JITTER[jitter]
    cx, cy = F.BOX_C
    for i, b in enumerate(books):
        b.set_pos(np.stack([cx + rng.uniform(-jx, jx, n), cy + rng.uniform(-jy, jy, n),
                            np.full(n, F.FLOOR + F.BOOK[2] * (i + 0.5) + 0.003 * (i + 1))], -1))
        b.set_quat(C.quat_z(rng.uniform(-math.radians(jyaw), math.radians(jyaw), n)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["pinch", "scoop"], default="scoop")
    ap.add_argument("--surface", choices=list(C.SURFACES), default="rough")
    ap.add_argument("--jitter", choices=list(JITTER), default="small")
    ap.add_argument("--grip-n", type=float, default=25.0, help="把持力の上限（N）。本を傷つけないよう抑える")
    ap.add_argument("--suction-n", type=float, default=None, help="吸引力の上限を上書きする（N）")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_id, out, meta = C.new_run(f"trial3_rough_{args.method}", args)
    rng = np.random.default_rng(args.seed)
    C.init_genesis(args.seed)
    rounded = args.method == "scoop"
    scene, books, ent, cam = F.build(args.n_envs, args.surface, out, rounded=rounded)
    n = args.n_envs
    r = C.Robot(scene, ent, n)
    r.setup()
    force = C.FORCE.copy()
    force[[7, 8]] = args.grip_n
    ent.set_dofs_force_range(-force, force, np.arange(len(C.HOME)))

    place_books(books, rng, n, args.jitter)
    r.step(80)
    top = books[-1]
    suction_n = args.suction_n if args.suction_n is not None else C.SURFACES[args.surface]["suction_max_n"]
    r.suction = suction = C.Suction(r, top, suction_n)
    cam.start_recording(save_to_filename=str(out / "env0.mp4"), fps=25)
    scene.cam_close.start_recording(save_to_filename=str(out / "env0_close.mp4"), fps=25)

    p0 = top.get_pos().cpu().numpy()
    yaw = C.yaw_of(top.get_quat().cpu().numpy())
    quat = C.down_quat_with_finger_axis(yaw)  # 指の開閉軸を本の幅方向に合わせる
    ly = C.quat_to_mat(quat)[:, :, 1]  # gripper の y 軸（左の指が開く向き）の世界での向き
    bottom = p0[:, 2] - F.BOOK[2] / 2
    top_z = p0[:, 2] + F.BOOK[2] / 2
    half_w = F.BOOK[0] / 2

    g = p0.copy()
    if args.method == "pinch":
        g[:, 2] = bottom + F.BOOK[2] / 2 + C.FINGERTIP_Z  # 指先を本の厚みの中ほどに
    else:
        g = g - ly * SCOOP_OFFSET  # 吸盤を本の右寄りへ
        g[:, 2] = bottom + 0.001 + C.FINGERTIP_Z + C.LIP_T  # 右の爪の下面を本の底の高さ（下の本の上面）に
    plunger_q = np.clip(g[:, 2] - top_z - C.CUP_TIP_RETRACTED_Z + 0.004, 0, C.PLUNGER_STROKE)  # 吸盤が上面に軽く当たる

    r.target[:, C.PLUNGER] = plunger_q
    if args.method == "scoop":
        r.target[:, 7] = C.FINGER_OPEN  # 左は最大まで開く
        r.target[:, 8] = half_w - SCOOP_OFFSET + 0.030  # 右は本の右の縁の 3 cm 外側
    above = g.copy()
    above[:, 2] += 0.12
    r.move_arm(above, quat, 120)
    r.move_linear(g, quat, 100)
    pos_err = r.correct(g, quat)
    max_f = np.zeros(n)

    def hook() -> None:
        np.maximum(max_f, np.abs(ent.get_dofs_control_force([7, 8]).cpu().numpy()).max(axis=1), out=max_f)

    edge_raise = np.zeros(n)
    inserted = np.zeros(n, bool)
    if args.method == "pinch":
        suction.on[:] = True
        r.step(10)
        r.move_joints(C.FINGERS, np.zeros((n, 2)), 60, hook)
        r.step(20, hook)
    else:
        # 1. 左の指で本の左の側面を押さえる（3 mm 押し込む）
        r.move_joints(7, np.full(n, half_w + SCOOP_OFFSET - 0.003), 50, hook)
        # 2. 吸着して、吸盤を 1.5 cm 引き込む → 右の端がめくれる（吸引力が足りなければ外れる）
        suction.on[:] = True
        r.step(10, hook)
        r.move_joints(C.PLUNGER, plunger_q - 0.015, 60, hook)
        r.step(10, hook)
        bm = C.quat_to_mat(top.get_quat().cpu().numpy())
        bp = top.get_pos().cpu().numpy()
        tilt = np.degrees(np.arccos(np.clip(bm[:, 2, 2], -1, 1)))
        edge_raise = np.sin(np.radians(tilt)) * F.BOOK[0]  # 左の縁を支点に傾いたときの右の縁の持ち上がり
        # 3. 爪を上下に小刻みに動かしながら、内側へ押し込む（2 秒）
        q_r0 = r.target[:, 8].copy()
        q_r1 = np.full(n, half_w - SCOOP_OFFSET + 0.002)  # 指の本体が本の側面の 2 mm 手前まで → 爪は約 2.3 cm 入る
        steps = 200
        for k in range(1, steps + 1):
            r.target[:, 8] = q_r0 + (q_r1 - q_r0) * (k / steps)
            r.target[:, C.LIFT] = 0.003 * math.sin(2 * math.pi * k / 25)
            r.step(1, hook)
        # 爪の先端が本の下に入ったか（本の座標で判定）
        tip = ent.get_link("finger_r_tip")
        tip_w = tip.get_pos().cpu().numpy() + r.approach_axis() * (C.FINGER_LEN + C.LIP_T / 2) \
            + ly * C.LIP_LEN  # 爪の内側の端
        bm = C.quat_to_mat(top.get_quat().cpu().numpy())
        bp = top.get_pos().cpu().numpy()
        tl = np.einsum("nji,nj->ni", bm, tip_w - bp)
        inserted = (np.abs(tl[:, 0]) < half_w - 0.01) & (tl[:, 2] < -F.BOOK[2] / 2 + 0.006)
        # 4. 爪を上げて本を支え、左の指で挟む
        r.move_joints(C.LIFT, np.full(n, -0.008), 50, hook)
        r.move_joints(7, np.full(n, half_w + SCOOP_OFFSET - 0.010), 30, hook)
        r.step(20, hook)

    lift = g.copy()
    lift[:, 2] += 0.25
    r.move_linear(lift, quat, 120, hook=hook)
    r.step(100, hook)
    cam.stop_recording()
    scene.cam_close.stop_recording()

    z_gain = top.get_pos().cpu().numpy()[:, 2] - p0[:, 2]
    f_touch, _ = r.touching(top)
    held = (z_gain > 0.15) & f_touch
    result = {
        "criterion": f"rough_{args.method}",
        "surface": args.surface,
        "jitter": args.jitter,
        "grip_limit_n": args.grip_n,
        "suction_max_n": suction_n,
        "success_rate": float(held.mean()),
        "edge_raise_median_mm": float(np.median(edge_raise) * 1000) if args.method == "scoop" else None,
        "inserted_rate": float(inserted.mean()) if args.method == "scoop" else None,
        "max_finger_force_n_median": float(np.median(max_f)),
        "per_env": {"pos_err_mm": pos_err * 1000, "edge_raise_mm": edge_raise * 1000, "inserted": inserted,
                    "z_gain_m": z_gain, "fingers_touch": f_touch, "max_finger_force_n": max_f, "held": held},
    }
    C.save_result(out, meta, result)


if __name__ == "__main__":
    main()
