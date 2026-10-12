"""実行の記録（出力先・メタデータ・結果の状態）。Genesis に依存しない。

状態は 3 つを分けて持つ（成功の表示が正しさを意味しないため）
  succeeded: 最後まで動き、受け入れ基準も満たした
  rejected:  最後まで動いたが、受け入れ基準を満たさなかった
  failed:    例外で止まった
出力は /workspace/outputs（Docker の named volume。OneDrive の外）に置く。
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

OUTCOMES = ("succeeded", "rejected", "failed")
OUTPUTS_ROOT = Path("/workspace/outputs")


def classify(acceptance_ok: bool, error: BaseException | None = None) -> str:
    if error is not None:
        return "failed"
    return "succeeded" if acceptance_ok else "rejected"


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def new_run(trial: str, args: dict, root: Path = OUTPUTS_ROOT, extra: dict | None = None) -> tuple[str, Path, dict]:
    run_id = new_run_id()
    out = Path(root) / trial / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {"trial": trial, "run_id": run_id, "args": dict(args), "git_commit": git_commit(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {})}
    return run_id, out, meta


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # git が無い・リポジトリの外（記録上は「不明」として残す）


def json_default(o):
    if hasattr(o, "tolist"):
        v = o.tolist()
        return round(v, 4) if isinstance(v, float) else v
    raise TypeError(type(o))


def save_result(out: Path, meta: dict, result: dict) -> Path:
    meta = {**meta, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **result}
    if meta.get("status") not in OUTCOMES:
        raise ValueError(f"result の status は {OUTCOMES} のどれか: {meta.get('status')!r}")
    path = Path(out) / "result.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return path
