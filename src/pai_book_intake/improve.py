"""実行の記録から、LLM に渡す「改善の材料」を作る（02_確認事項 B11。査定そのものには LLM を使わない）。

ねらい: 工程全体が定期的に改善されていく仕組みにする。毎回の実行の記録（修正依頼・停止の理由・工程ごとの成功率・
失敗の画像）を集め、LLM が「次に直すこと」と「次に学習させる場面の候補」をまとめる。

ここでやるのは数えることだけ（W10: 数えられることはコードで数える。LLM には数えさせない）
  - p2_line の run: 停止の理由・工程ごとの成功率（環境 0）・本ごとの失敗した工程・工程間のメッセージの件数・失敗の画像
  - act_a eval の run: 本ごとの成功数と、失敗の形（箱に残った・持ち上げたが台に無い・台の上でずれた）
LLM への頼み方（prompt）も一緒に書き出す。LLM の答えは「案」として扱い、採否は人が決める

使い方: python -m pai_book_intake.improve --runs /workspace/outputs/p2_line/<id> /workspace/outputs/act_eval/<id> ...
出力: /workspace/outputs/improve/<run_id>/digest.json・prompt.md
"""

from __future__ import annotations

import argparse
import collections
import json
import traceback
from pathlib import Path

from .runlog import classify, new_run, save_result

# act_a eval の失敗の形。箱から台までは 500 mm 以上あるので、それより離れていれば本は箱の側に残っている
LEFT_IN_BOX_MM = 400.0

PROMPT = """あなたは、中古書店の買取ライン（シミュレーション）の改善担当です。下の digest は、実行の記録をコードで数えたものです。
これだけを根拠に、次の 2 つを日本語で書いてください。

1. 次に直すこと（多くても 5 つ。効きそうな順）
   - それぞれ: 何が起きたか（digest のどの値か。run_id と book_id を書く）・考えられる原因・直し方の案・直ったかの確かめ方
2. 次に学習させる場面の候補（多くても 5 つ）
   - それぞれ: 場面（本・山の中の高さ・工程）・なぜ学習が効きそうか・デモの集め方の案

決まり
- digest に無い数値を作らない。分からないことは「記録からは分からない」と書く
- 原因は「仮説」と書く。確かめる手順を必ず付ける
- 実装されていない機能（not_implemented）を、できることとして書かない
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="p2_line・act_eval の run のフォルダ")
    return ap.parse_args()


def failed_stage(rec: dict) -> str | None:
    """環境 0 で、その本が最初につまずいた工程（stage_rates と同じ並び）。届いたなら None"""
    if rec.get("skipped"):
        return f"skipped:{rec['skipped']}"
    a = rec.get("a")
    if a is None:
        return None
    if not a["on_table"][0]:
        return "pick"
    if "isbn_read" in rec and rec["isbn_read"][0] is None:
        return "barcode"
    dest = (rec.get("destination") or [None])[0]
    if dest is None:
        return "release"
    if not (rec.get("b_pick") or {}).get("lifted", [True])[0]:
        return "transport"
    if dest == "shelf" and "shelf" in rec:
        if rec["shelf"].get("plan", [None])[0] is None:
            return "free_space"
        if not rec["shelf"]["ok"][0]:
            return "shelve"
    if dest == "discard" and "bin" in rec and not rec["bin"]["in_bin_flat"][0]:
        return "discard"
    if "delivered" in rec and not rec["delivered"][0]:
        return "delivered"
    return None


def digest_p2_line(d: Path) -> dict:
    r = json.loads((d / "result.json").read_text())
    topics: collections.Counter = collections.Counter()
    held: list = []
    face_down: list = []
    for line in (d / "bus.jsonl").read_text().splitlines():
        m = json.loads(line)
        topics[m["topic"]] += 1
        if m["topic"] == "run.start":
            face_down = list(m["payload"].get("face_down") or [])
        if m["topic"] == "tray.held":
            held.append(m.get("payload"))
    books = []
    for rec in r.get("per_book", []):
        st = failed_stage(rec)
        row = {"book_id": rec.get("book_id"), "failed_stage": st, "injected": rec.get("injected"),
               "face_down": rec.get("book_id") in face_down}  # 裏向き＝読めないのが正しい本
        img = d / f"barcode_{rec.get('book_id')}_env0.png"
        if st == "barcode" and img.exists():
            row["image"] = f"{d.name}/{img.name}"
        if "grippers_at_scan" in rec:
            row["grippers_at_scan"] = rec["grippers_at_scan"]
        books.append(row)
    return {"kind": "p2_line", "run_id": r["run_id"], "status": r.get("status"), "args": r.get("args"),
            "stop_reasons": r.get("stop_reasons"), "stopped_at_book": r.get("stopped_at_book"),
            "stage_rates_env0": r.get("stage_rates_env0"), "books": books,
            "messages": {k: topics[k] for k in ("fix_request", "fix.declined", "escalation", "tray.held", "stop",
                                                 "line.waiting", "line.stopped") if topics[k]},
            "held": held, "not_implemented": r.get("not_implemented")}


def digest_act_eval(d: Path) -> dict:
    r = json.loads((d / "result.json").read_text())
    books = []
    for b in r.get("per_book", []):
        ok, lifted, on_table = b["ok"], b["lifted"], b["on_table"]
        err = b["pos_err_mm"]
        modes = collections.Counter()
        for i in range(len(ok)):
            if ok[i]:
                continue
            if err[i] > LEFT_IN_BOX_MM:
                modes["dropped_after_lift" if lifted[i] else "left_in_box"] += 1
            elif not on_table[i]:
                modes["not_flat_on_table"] += 1
            else:
                modes["off_slot"] += 1
        books.append({"book_id": b["book_id"], "ok": int(sum(ok)), "n": len(ok), "failure_modes": dict(modes)})
    return {"kind": "act_eval", "run_id": r["run_id"], "status": r.get("status"), "args": r.get("args"),
            "pile_bottom_to_top": r.get("pile_bottom_to_top"), "jitter": r.get("jitter"), "books": books,
            "box_all_ok": r.get("box_all_ok"), "n_envs": r.get("n_envs")}


def main() -> None:
    args = parse_args()
    run_id, out, meta = new_run("improve", {"runs": [str(p) for p in args.runs]})
    result: dict = {}
    error = None
    try:
        runs = []
        for d in args.runs:
            trial = json.loads((d / "result.json").read_text()).get("trial")
            runs.append(digest_p2_line(d) if trial == "p2_line" else digest_act_eval(d))
        digest = {"source_runs": [r["run_id"] for r in runs], "runs": runs}
        (out / "digest.json").write_text(json.dumps(digest, ensure_ascii=False, indent=2))
        (out / "prompt.md").write_text(PROMPT + "\n## digest\n\n```json\n" + json.dumps(digest, ensure_ascii=False, indent=2)
                                       + "\n```\n")
        result = {"acceptance": "全部の run を読めて、digest と prompt を書いた", "acceptance_ok": True,
                  "source_runs": digest["source_runs"]}
    except BaseException as ex:
        error = ex
        result = {"error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
        raise
    finally:
        result["status"] = classify(bool(result.get("acceptance_ok", False)), error)
        save_result(out, meta, result)
        print(f"run_id={run_id} status={result['status']} out={out}")


if __name__ == "__main__":
    main()
