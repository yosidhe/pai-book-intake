"""試行5 の集計: 指定した run の result.json を読み、セル（手の形 × 吸引）ごとの表と、手の形ごとの順位を作る。

使い方: python spikes/trial5_summary.py --tag coarse <run_id> <run_id> ...
出力: /workspace/outputs/trial5_summary/<run_id>/（sweep.csv・ranking.csv・result.json）
数字は一次データ（各 run の result.json）から読むだけで、ここでは作らない。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pai_book_intake.runlog import OUTPUTS_ROOT, new_run, save_result

SRC = OUTPUTS_ROOT / "trial5_slideunder"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    run_id, out, meta = new_run("trial5_summary", vars(args))
    rows = []
    for rid in args.runs:
        d = json.loads((SRC / rid / "result.json").read_text(encoding="utf-8"))
        if d["status"] == "failed":
            raise RuntimeError(f"{rid} は例外で止まっている: {d.get('error')}")
        for key, c in d["cells"].items():
            pe = c["per_env"]
            sup = pe.get("support_ratio_median") or []
            rows.append({
                "run_id": rid, "book": d["book"]["book_id"], "format": d["book"]["format"],
                "surface": d["book"]["surface"], "situation": d["situation"], "design": c["design"],
                "suction": "on" if c["suction_assist"] else "off", "n": d["n_envs"],
                "success": c["success_n"], "success_raw": c["success_raw_n"], "implausible": c["implausible_n"],
                "viol_any": c["violations_any_n"], **{f"viol_{k}": v for k, v in c["violations"].items()},
                "peak_link_force_med_n": round(c["peak_hand_force_n"]["median"], 1),
                "peak_link_force_max_n": round(c["peak_hand_force_n"]["max"], 1),
                "peak_total_force_max_n": round(c.get("peak_total_hand_force_n", {}).get("max", float("nan")), 1),
                "max_pen_mm": round(c["max_penetration_mm"], 2),
                "support_ratio_med": round(sorted(sup)[len(sup) // 2], 2) if sup else "",
                "cycle_s": round(c["cycle_s"], 2),
                "modes": ";".join(f"{k}={v}" for k, v in c["failure_modes"].items()),
            })
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        a = agg[(r["design"], r["suction"])]
        a["n"] += r["n"]
        a["success"] += r["success"]
        a["success_raw"] += r["success_raw"]
        a["implausible"] += r["implausible"]
        a["viol_any"] += r["viol_any"]
        a["peak_force_max"] = max(a["peak_force_max"], r["peak_link_force_max_n"])
        a["pen_max"] = max(a["pen_max"], r["max_pen_mm"])
    rank = sorted(({"design": k[0], "suction": k[1], **v, "rate": round(v["success"] / v["n"], 3)} for k, v in agg.items()),
                  key=lambda x: (-x["rate"], x["peak_force_max"]))
    with open(out / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rank[0]))
        w.writeheader()
        w.writerows(rank)
    for x in rank:
        print(f"{x['design']:8s} {x['suction']:3s} {x['success']:3d}/{x['n']:3d} raw={x['success_raw']} impl={x['implausible']} "
              f"viol={x['viol_any']} Fmax={x['peak_force_max']:.1f} pen={x['pen_max']:.2f}")
    save_result(out, meta, {"status": "succeeded", "tag": args.tag, "runs": args.runs, "ranking": rank,
                            "note": "表の数字は各 run の result.json から読んだもの"})
    print(f"run_id={run_id} out={out}")


if __name__ == "__main__":
    main()
