"""試行8 の結果を短く表示する（cells.csv と、環境ごとの主な測定値）。使い方: python spikes/trial8_peek.py <run_id>"""
import csv
import json
import sys
from pathlib import Path

base = Path("/workspace/outputs/trial8_pinch")
run = base / sys.argv[1] if len(sys.argv) > 1 else sorted(base.iterdir())[-1]
res = json.loads((run / "result.json").read_text())
r = res.get("result", res)
if "error" in r:
    print(r["error"]); print(r.get("traceback", "")[-3000:]); sys.exit()
for row in csv.DictReader(open(run / "cells.csv", encoding="utf-8")):
    print({k: row[k] for k in ("cell", "success", "success_raw", "implausible", "lift_gap_mm_med", "lift_tilt_deg_max",
                               "claw_shift_mm_max", "support_med", "withdraw_shift_mm_max", "release_rise_mm_max", "max_pen_mm", "lower_disp_mm_max", "claw_support_from_com_mm_med", "modes")})
keys = sys.argv[2].split(",") if len(sys.argv) > 2 else ["lift_gap_mm", "lift_tilt_deg", "claw_depth_mm", "inserted", "claw_shift_mm",
       "support_ratio_median", "lifted", "carried", "slip_in_carry_mm", "withdraw_shift_mm", "claw_left_under", "release_rise_mm",
       "fore_touch", "violated_edge_rest", "violated_penetration", "violated_airborne", "max_penetration_mm", "first_t_penetration"]
for k, c in r["cells"].items():
    pe = c["per_env"]
    print("==", k, c["checks_pass_n"])
    for key in keys:
        v = pe.get(key)
        if isinstance(v, list):
            v = [round(x, 2) if isinstance(x, float) else x for x in v]
        print(f"   {key}: {v}")
print("files:", sorted(p.name for p in run.iterdir()))
