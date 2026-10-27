"""p2_line の結果を短く見る（使い捨て）。使い方: python spikes/p2_peek.py <run_id>"""

import json
import sys

import numpy as np

r = json.load(open(f"/workspace/outputs/p2_line/{sys.argv[1]}/result.json"))
r = r.get("result", r)


def short(d, depth=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * depth + k + ":")
            short(v, depth + 1)
        elif isinstance(v, list) and len(str(v)) > 160:
            print("  " * depth + k, str(v)[:160])
        else:
            print("  " * depth + k, v)


for pb in r.get("per_book", []):
    print("=" * 20, pb["book_id"])
    short(pb)
print("succeeded", r.get("books_succeeded_per_env"), "expected", r.get("books_expected_per_env"),
      "shelved", r.get("shelved_upright_inside_per_env"))
for row in r.get("inventory", []):
    print("inv", {k: row[k] for k in ("env", "book_id", "x_lo_m", "x_hi_m", "upright", "inside", "tilt_deg", "spine_y_m", "note")})
if "error" in r:
    print(r["traceback"][-1500:])
scan = sys.argv[2] if len(sys.argv) > 2 else None
if scan:
    z = np.load(f"/workspace/outputs/p2_line/{sys.argv[1]}/{scan}")
    occ = z["occupied"]
    print("".join("#" if o else "." for o in occ))
