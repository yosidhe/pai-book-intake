"""試行5b の集計: 指定した run の result.json を読み、セル（手の形 × 吸引）ごとの表と、
（置き方 × 手の形 × 吸引）ごとの合計・失敗の内訳を作る。

使い方: python spikes/trial5b_summary.py --tag coarse <run_id> <run_id> ...
出力: /workspace/outputs/trial5b_summary/<run_id>/（sweep.csv・ranking.csv・result.json）
数字は一次データ（各 run の result.json）から読むだけで、ここでは作らない。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

from pai_book_intake.runlog import OUTPUTS_ROOT, new_run, save_result

SRC = OUTPUTS_ROOT / "trial5b_standing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    run_id, out, meta = new_run("trial5b_summary", vars(args))
    rows = []
    for rid in args.runs:
        d = json.loads((SRC / rid / "result.json").read_text(encoding="utf-8"))
        if d["status"] == "failed":
            raise RuntimeError(f"{rid} は例外で止まっている: {d.get('error')}")
        for key, c in d["cells"].items():
            rows.append({
                "run_id": rid, "book": d["book"]["book_id"], "format": d["book"]["format"],
                "surface": d["book"]["surface"], "situation": d["situation"], "design": c["design"],
                "suction": "on" if c["suction_assist"] else "off", "n": d["n_envs"],
                "success": c["success_n"], "success_raw": c["success_raw_n"], "implausible": c["implausible_n"],
                "lay_ok": c["lay_ok_n"], "viol_any": c["violations_any_n"],
                **{f"viol_{k}": v for k, v in c["violations"].items()},
                "peak_link_force_max_n": round(c["peak_hand_force_n"]["max"], 1),
                "max_pen_mm": round(c["max_penetration_mm"], 2),
                "lay_s": round(c["lay_s"], 2), "cycle_s": round(c["cycle_s"], 2),
                "modes": ";".join(f"{k}={v}" for k, v in c["failure_modes"].items()),
            })
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg: dict = defaultdict(lambda: defaultdict(float))
    modes: dict = defaultdict(Counter)
    for r in rows:
        k = (r["situation"], r["design"], r["suction"])
        a = agg[k]
        for f in ("n", "success", "success_raw", "implausible", "lay_ok", "viol_any", "viol_edge_rest", "viol_airborne",
                  "viol_penetration"):
            a[f] += r[f]
        a["peak_force_max"] = max(a["peak_force_max"], r["peak_link_force_max_n"])
        a["pen_max"] = max(a["pen_max"], r["max_pen_mm"])
        a["cycle_sum"] += r["cycle_s"] * r["n"]
        for m in r["modes"].split(";"):
            if m:
                name, v = m.rsplit("=", 1)
                modes[k][name] += int(v)
    rank = []
    for k, v in sorted(agg.items()):
        rank.append({"situation": k[0], "design": k[1], "suction": k[2],
                     **{f: int(v[f]) for f in ("n", "success", "success_raw", "implausible", "lay_ok", "viol_any",
                                               "viol_edge_rest", "viol_airborne", "viol_penetration")},
                     "rate": round(v["success"] / v["n"], 3), "peak_force_max_n": v["peak_force_max"],
                     "pen_max_mm": v["pen_max"], "cycle_mean_s": round(v["cycle_sum"] / v["n"], 2),
                     "modes": ";".join(f"{m}={c}" for m, c in modes[k].most_common())})
    with open(out / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rank[0]))
        w.writeheader()
        w.writerows(rank)
    for x in rank:
        print(f"{x['situation']:5s} {x['design']:4s} {x['suction']:3s} {x['success']:3d}/{x['n']:3d} raw={x['success_raw']} "
              f"impl={x['implausible']} lay_ok={x['lay_ok']} viol={x['viol_any']} Fmax={x['peak_force_max_n']:.1f} "
              f"pen={x['pen_max_mm']:.2f} cyc={x['cycle_mean_s']} | {x['modes']}")
    save_result(out, meta, {"status": "succeeded", "tag": args.tag, "runs": args.runs, "ranking": rank,
                            "note": "表の数字は各 run の result.json から読んだもの"})
    print(f"run_id={run_id} out={out}")


if __name__ == "__main__":
    main()
