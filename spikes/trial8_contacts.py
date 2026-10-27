"""試行8 の結果から、浮かせた直後の接触（環境 0）と浮いた高さを表示する。使い方: python spikes/trial8_contacts.py [run_id]"""
import json
import sys
from pathlib import Path

base = Path("/workspace/outputs/trial8_pinch")
run = base / sys.argv[1] if len(sys.argv) > 1 else sorted(base.iterdir())[-1]
r = json.loads((run / "result.json").read_text())
r = r.get("result", r)
print(run.name, r.get("physics"), r["book"]["surface"])
for k, c in r["cells"].items():
    pe = c["per_env"]
    print(k, "lift_gap_mm", [round(v, 2) for v in pe["lift_gap_mm"]])
    for x in pe.get("contacts_after_raise_env0", []):
        f = x["f_world_n"]
        print("   ", x["other"], x["loc_mm"], [round(v, 2) for v in f])
