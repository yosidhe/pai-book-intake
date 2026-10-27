"""ロボット B の手のリンクにある当たりの形と、その衝突ビットを表示する（使い捨て）。"""
from pathlib import Path

import genesis as gs
from pai_book_intake.hand_design import PinchHand
from pai_book_intake.pinch import build_pinch_robot_xml

gs.init(backend=gs.cpu, logging_level="warning")
out = Path("/workspace/outputs/geom_bits_check")
out.mkdir(parents=True, exist_ok=True)
scene = gs.Scene(show_viewer=False)
ent = scene.add_entity(gs.morphs.MJCF(file=str(build_pinch_robot_xml(out, PinchHand(name="P10"), mobile=True, palm_through_env=True))))
scene.build()
for name in ("link7", "gripper", "finger_l", "finger_r", "claw", "plate"):
    lk = ent.get_link(name)
    for g in lk.geoms:
        print(name, g.type, [round(float(v), 3) for v in g.data[:3]] if hasattr(g, "data") and g.data is not None else "",
              "contype", getattr(g, "contype", "?"), "conaff", getattr(g, "conaffinity", "?"))
