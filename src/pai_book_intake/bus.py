"""工程どうしのメッセージバス（同じプロセスの中の pub/sub）。

すべてのメッセージを、配る前に JSONL（1 行 1 件）へ追記する（観測を先に置く）。
1 行の形（記録の種類と書き込み先）:
  - type は feedback / event / sample / note のどれか
  - feedback（修正依頼・好評価）は「受け取る側（to_task）」の records に書く。それ以外は送った側（from_task）の records
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# inspect-scan は検品トレー案（inspection.py）の工程。P1 の流れ（p1_demo）はまだ inspect-flip を使う
TASKS = ("robot-a-pick", "inspect-flip", "inspect-scan", "robot-b-sort", "appraisal", "observer")
BROADCAST = "*"
RECORD_TYPES = ("feedback", "event", "sample", "note")


@dataclass(frozen=True)
class Message:
    run_id: str
    msg_id: str
    seq: int
    sim_time: float | None  # シミュレーションの時刻（秒）。シミュレーションの外なら None
    wall_time: str  # ISO 8601（UTC）
    from_task: str
    to_task: str  # TASKS のどれか、または "*"（全員）
    type: str  # RECORD_TYPES のどれか
    topic: str  # 何の知らせか（例: pick.done / fix_request / praise / stop）
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_storage_record(self) -> dict:
        """記録の形。{"task": 書き込み先のタスク, "record": 本文}。"""
        task = self.to_task if self.type == "feedback" else self.from_task
        return {
            "task": task,
            "record": {
                "type": self.type,
                "from_task": self.from_task,
                "run_id": self.run_id,
                "occurred_at": self.wall_time,
                "payload": {**self.payload, "_msg": {"msg_id": self.msg_id, "seq": self.seq, "topic": self.topic,
                                                     "to_task": self.to_task, "sim_time": self.sim_time}},
            },
        }


def _json_default(o):
    # numpy の配列・数値を JSON にする（bus は numpy に依存しない）
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"JSON にできない値: {type(o).__name__}")


Handler = Callable[[Message], None]


class Bus:
    """publish したメッセージは、まず JSONL に書き、次に購読者へ順に配る。

    購読者の中から publish された分は待ち行列に積み、今のメッセージを全員に配り終えてから配る（順序を保つ）。
    購読者の例外はそのまま呼び出し元へ上げる（握りつぶさない）。
    """

    def __init__(self, run_id: str, log_path: Path, sim_clock: Callable[[], float | None] | None = None):
        self.run_id = run_id
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", encoding="utf-8")
        self._clock = sim_clock or (lambda: None)
        self._subs: list[tuple[str | None, str | None, Handler]] = []
        self._queue: deque[Message] = deque()
        self._dispatching = False
        self._seq = 0
        self.history: list[Message] = []

    def set_sim_clock(self, sim_clock: Callable[[], float | None]) -> None:
        self._clock = sim_clock

    def subscribe(self, handler: Handler, to_task: str | None = None, topic: str | None = None) -> None:
        """to_task=None は宛先を問わず全部（オブザーバ用）。宛先を指定すると、その宛先と "*" 宛てを受け取る。"""
        if to_task is not None and to_task not in TASKS:
            raise ValueError(f"未知のタスク: {to_task!r}")
        self._subs.append((to_task, topic, handler))

    def publish(self, from_task: str, to_task: str, type: str, topic: str, payload: dict | None = None) -> Message:
        if from_task not in TASKS:
            raise ValueError(f"未知の送り手: {from_task!r}")
        if to_task != BROADCAST and to_task not in TASKS:
            raise ValueError(f"未知の宛先: {to_task!r}")
        if type not in RECORD_TYPES:
            raise ValueError(f"type は {RECORD_TYPES} のどれか: {type!r}")
        if not topic:
            raise ValueError("topic が空")
        self._seq += 1
        sim_t = self._clock()
        msg = Message(
            run_id=self.run_id, msg_id=uuid.uuid4().hex, seq=self._seq,
            sim_time=None if sim_t is None else round(float(sim_t), 4),
            wall_time=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            from_task=from_task, to_task=to_task, type=type, topic=topic, payload=dict(payload or {}),
        )
        # 先に記録する（配る途中で落ちても、何が送られたかは残る）
        line = json.dumps({**msg.to_dict(), "storage": msg.to_storage_record()}, ensure_ascii=False,
                          default=_json_default)
        self._log.write(line + "\n")
        self._log.flush()
        self.history.append(msg)
        self._queue.append(msg)
        if not self._dispatching:
            self._drain()
        return msg

    def _drain(self) -> None:
        self._dispatching = True
        try:
            while self._queue:
                msg = self._queue.popleft()
                for to_task, topic, handler in list(self._subs):
                    if to_task is not None and msg.to_task not in (to_task, BROADCAST):
                        continue
                    if topic is not None and msg.topic != topic:
                        continue
                    handler(msg)
        finally:
            self._dispatching = False

    def close(self) -> None:
        self._log.close()

    def __enter__(self) -> "Bus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
