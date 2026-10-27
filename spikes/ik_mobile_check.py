"""台車に載せた腕で、Genesis の IK が台車の今の向きを使っているかを確かめる（使い捨て）。

台車の向きを変えてから、手を正面に水平に伸ばす姿勢の IK を解き、その関節角を入れて手の姿勢を測る。
"""

import math

import numpy as np

import genesis as gs
from pai_book_intake.geometry import quat_to_mat
from pai_book_intake.hand_design import PinchHand
from pai_book_intake.pinch import PinchRobot, build_pinch_robot_xml, mats_to_quat
from pai_book_intake.tool import World
from pathlib import Path

gs.init(backend=gs.gpu, logging_level="warning")
out = Path("/workspace/outputs/ik_mobile_check")
out.mkdir(parents=True, exist_ok=True)
d = PinchHand(name="P10")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
ent = scene.add_entity(gs.morphs.MJCF(file=str(build_pinch_robot_xml(out, d, mobile=True))))
scene.build(n_envs=1)
w = World(scene, 1, 0.01)
for yaw in (0.0, math.pi / 2, -math.pi / 2):
    r = PinchRobot(w, ent, "b", d, base_pose=(0.0, 0.0, yaw))
    r.setup()
    w.robots.clear()
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    x = np.array([0.0, 0.0, -1.0])
    z = fwd
    y = np.cross(z, x)
    m = np.stack([x, y, z], -1)[None]
    q = mats_to_quat(m)
    pos = (fwd * 0.5 + np.array([0, 0, 0.45]))[None]
    sol = r.ik(pos, q)
    full = r.e.get_qpos().cpu().numpy()
    full[:, r.arm] = sol
    r.e.set_qpos(full)
    mg = quat_to_mat(r.gripper.get_quat().cpu().numpy())
    err = math.degrees(math.acos(np.clip((np.trace(mg[0].T @ m[0]) - 1) / 2, -1, 1)))
    print(f"base_yaw={math.degrees(yaw):.0f} pos_err_mm={np.linalg.norm(r.gripper.get_pos().cpu().numpy() - pos) * 1000:.1f} "
          f"rot_err_deg={err:.1f} hand_z={np.round(mg[0][:, 2], 2)} want_z={np.round(z, 2)} base_q={np.round(full[0, r.base], 3)}")
