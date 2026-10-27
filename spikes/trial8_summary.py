"""試行8 の本試験をまとめる（試行5 の trial5_summary.py と同じ考え方）。

使い方: python spikes/trial8_summary.py --since 20261018T1415
  --since 以降の run_id のうち、n_envs が --n と同じものを集める。数は生の result.json から数え直す（cells.csv は使わない）
出力: 標準出力の表と /workspace/outputs/trial8_summary/<run_id>/（result.json・summary.csv）
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from pai_book_intake.runlog import OUTPUTS_ROOT, new_run, save_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", default="99999999T9999")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    base = OUTPUTS_ROOT / "trial8_pinch"
    runs = []
    for d in sorted(base.iterdir()):
        if not (args.since <= d.name <= args.until) or not (d / "result.json").exists():
            continue
        r = json.loads((d / "result.json").read_text())
        r = r.get("result", r)
        if "cells" not in r or r.get("n_envs") != args.n:
            continue
        runs.append((d.name, r))
    agg = defaultdict(lambda: defaultdict(lambda: np.zeros(2, int)))  # cell -> 区分 -> [成功, 試行]
    modes = defaultdict(lambda: defaultdict(int))
    metr = defaultdict(lambda: defaultdict(list))
    rows = []
    for rid, r in runs:
        b, sit = r["book"], r["situation"]
        for key, c in r["cells"].items():
            pe = c["per_env"]
            ok = np.array([m == "ok" for m in pe["mode"]])
            for part in ("all", sit, f"{sit}:{b['surface']}", b["book_id"]):
                agg[key][part] += [ok.sum(), len(ok)]
            for m in pe["mode"]:
                modes[key][m] += 1
            for k in ("support_ratio_median", "claw_support_from_com_mm", "lift_gap_mm", "lift_tilt_deg", "claw_shift_mm",
                      "withdraw_shift_mm", "release_rise_mm", "max_penetration_mm", "cycle_s"):
                v = pe.get(k)
                if v is not None:
                    metr[key][k] += list(np.atleast_1d(np.array(v, float)))
            rows.append({"run_id": rid, "book": b["book_id"], "format": b["format"], "surface": b["surface"], "situation": sit,
                         "cell": key, "success": int(ok.sum()), "n": len(ok),
                         "modes": ";".join(f"{m}={n}" for m, n in sorted(c["failure_modes"].items())),
                         "physics": json.dumps(r.get("physics"))})
    summary = {}
    print(f"runs={len(runs)}")
    print("cell | all | stack | flat | 失敗の内訳 | 正味の支え(中央値) | 支えの中心と重心の距離 mm(中央値) | 浮き mm | 抜くときのずれ max mm")
    for key in sorted(agg):
        a = agg[key]
        m = metr[key]
        fmt = lambda p: f"{a[p][0]}/{a[p][1]}" if p in a else "-"  # noqa: E731
        fail = {k: v for k, v in modes[key].items() if k != "ok"}
        summary[key] = {"success": {p: a[p].tolist() for p in a}, "failure_modes": fail,
                        **{f"{k}_median": float(np.nanmedian(v)) for k, v in m.items()},
                        **{f"{k}_max": float(np.nanmax(v)) for k, v in m.items()}}
        print(f"{key} | {fmt('all')} | {fmt('stack')} | {fmt('flat')} | {fail} | "
              f"{np.nanmedian(m['support_ratio_median']):.2f} | {np.nanmedian(m['claw_support_from_com_mm']):.0f} | "
              f"{np.nanmedian(m['lift_gap_mm']):.1f} | {np.nanmax(m['withdraw_shift_mm']):.1f}")
    run_id, out, meta = new_run("trial8_summary", vars(args))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    save_result(out, meta, {"runs": [rid for rid, _ in runs], "cells": summary, "status": "succeeded"})
    print(f"summary run_id={run_id}")


if __name__ == "__main__":
    main()
