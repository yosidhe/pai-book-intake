"""摩擦の較正: 本を左右から一定の力で挟み、重さで滑り落ちるかを見て、シミュで実際に効いている摩擦係数を求める。

背景（試行8、2026-10-18）: ゴム μ1.0 で本の重さの 2 倍の摩擦が出るように挟んでも、手を上げると本が滑り、
接触の力から求めた摩擦は 0.5 前後だった。腕・爪・台を除いて、接触の設定（摩擦円錐・拘束の硬さ・noslip）だけで比べる。

形: 本（Box、重さ m、摩擦 μ_book）を、x の向きに滑る 2 枚の板（ゴム μ_pad。y・z は固定）で挟む。板には一定の力 F を
  内向きにかける（関節の力の指令）。重力で本が下へ滑る速さを測り、0.5 s の間に 1 mm 以上下がったら「滑った」。
  F を変えて、滑らない最小の F から μ_eff = m g / (2 F) を出す（摩擦が設定どおりなら μ_eff = max(μ_pad, μ_book)）。

使い方: python spikes/friction_calib.py --configs "pyr:ns5:tc0.01,pyr:ns5:tc0,ell:ns0:tc0.01"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs

PAD_XML = """<mujoco><worldbody>
  <body name="pads" pos="0 0 0.5">
    <body name="pad_l" pos="-{off} 0 0">
      <joint name="jl" type="slide" axis="1 0 0" range="-0.05 0.05" damping="5"/>
      <geom type="box" size="0.006 0.05 0.03" pos="-0.006 0 0" friction="{mu} 0.01 0.001" mass="0.1"/>
    </body>
    <body name="pad_r" pos="{off} 0 0">
      <joint name="jr" type="slide" axis="-1 0 0" range="-0.05 0.05" damping="5"/>
      <geom type="box" size="0.006 0.05 0.03" pos="0.006 0 0" friction="{mu} 0.01 0.001" mass="0.1"/>
    </body>
  </body>
</worldbody></mujoco>"""


def run_config(cfg: str, mass: float, mu_pad: float, mu_book: float, forces: list[float], out: Path) -> dict:
    cone, ns, tc = cfg.split(":")
    w, h, t = 0.182, 0.257, 0.045  # B5 の寸法（本の座標: x=横幅）
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
                     rigid_options=gs.options.RigidOptions(
                         enable_self_collision=False, noslip_iterations=int(ns[2:]),
                         friction_cone=gs.friction_cone.elliptic if cone == "ell" else gs.friction_cone.pyramidal,
                         constraint_timeconst=float(tc[2:]) or None, gravity=(0, 0, -9.81)),
                     show_viewer=False)
    xml = out / f"pads_{mu_pad}.xml"
    xml.write_text(PAD_XML.format(off=w / 2 + 0.002, mu=mu_pad))
    pads = scene.add_entity(gs.morphs.MJCF(file=str(xml)))
    book = scene.add_entity(gs.morphs.Box(size=(w, h, t), pos=(0, 0, 0.5)),
                            material=gs.materials.Rigid(rho=mass / (w * h * t), friction=mu_book))
    n = len(forces)
    scene.build(n_envs=n)
    dofs = [pads.get_joint("jl").dofs_idx_local[0], pads.get_joint("jr").dofs_idx_local[0]]
    f = np.repeat(np.array(forces)[:, None], 2, axis=1)
    pads.set_dofs_kp(np.zeros(2), dofs)
    for _ in range(30):  # 本を宙に保ったまま挟む力を立ち上げる（最初の 0.3 s は本を固定）
        pads.control_dofs_force(f, dofs)
        book.set_pos(np.tile([0, 0, 0.5], (n, 1)))
        book.set_dofs_velocity(np.zeros((n, 6)))
        scene.step()
    z0 = book.get_pos().cpu().numpy()[:, 2].copy()
    for _ in range(50):
        pads.control_dofs_force(f, dofs)
        scene.step()
    drop = (z0 - book.get_pos().cpu().numpy()[:, 2]) * 1000
    held = drop < 1.0
    weight = mass * 9.81
    f_hold = min((F for F, hd in zip(forces, held) if hd), default=None)
    return {"config": cfg, "mu_pad": mu_pad, "mu_book": mu_book, "forces_n": forces,
            "drop_mm_in_0.5s": [round(float(d), 2) for d in drop], "held": held.tolist(),
            "f_hold_min_n": f_hold, "mu_eff": round(weight / (2 * f_hold), 3) if f_hold else None,
            "mu_expected": max(mu_pad, mu_book)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="pyr:ns5:tc0.01,pyr:ns5:tc0,ell:ns0:tc0.01,ell:ns0:tc0")
    ap.add_argument("--mass", type=float, default=1.5)
    ap.add_argument("--mu-pad", type=float, default=1.0)
    ap.add_argument("--mu-book", type=float, default=0.4)
    args = ap.parse_args()
    weight = args.mass * 9.81
    # μ_eff が 0.2〜1.5 のどこにあっても分かるよう、F を重さ/(2 μ) の格子で並べる
    mus = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    forces = [round(weight / (2 * m), 2) for m in mus]
    out = Path("/workspace/outputs/friction_calib")
    out.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, logging_level="warning")
    res = []
    for cfg in args.configs.split(","):
        r = run_config(cfg, args.mass, args.mu_pad, args.mu_book, forces, out)
        print(json.dumps({k: r[k] for k in ("config", "mu_eff", "mu_expected", "f_hold_min_n")}))
        print("   drop", dict(zip([f"μ{m}" for m in mus], r["drop_mm_in_0.5s"])))
        res.append(r)
        gs.destroy()
        gs.init(backend=gs.gpu, logging_level="warning")
    (out / f"calib_pad{args.mu_pad}_book{args.mu_book}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
