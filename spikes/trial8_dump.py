"""試行8 の結果から、セルごとに指定のキー（環境ごとの値・接触の記録）を表示する。使い方: python spikes/trial8_dump.py <run_id|""> key1,key2"""
import json
import sys
from pathlib import Path

base = Path("/workspace/outputs/trial8_pinch")
run = base / sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else sorted(base.iterdir())[-1]
r = json.loads((run / "result.json").read_text())
r = r.get("result", r)
if "error" in r:
    print(r["error"], r.get("traceback", "")[-2000:]); sys.exit()
print(run.name, r.get("physics"), r["book"])
for k, c in r["cells"].items():
    print("==", k, "success", c["success_n"], c["failure_modes"])
    for key in sys.argv[2].split(","):
        v = c["per_env"].get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print("  ", key)
            for x in v:
                print("     ", x)
        else:
            if isinstance(v, list):
                v = [round(x, 2) if isinstance(x, float) else x for x in v]
            print("  ", key, v)
