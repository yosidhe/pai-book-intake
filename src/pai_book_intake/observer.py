"""オブザーバ: バスの全メッセージを見て、止める判断と箱ごとの報告書を書く。

止める条件（しきい値は ObserverConfig）
  - 本の処理の失敗（rejected / failed）が max_consecutive_failures 回続いた
  - 本棚が満杯になった（棚に置いた冊数 ≥ shelf_capacity）
  - バーコードが barcode_unreadable_limit 回続けて読めなかった（カメラや照明の故障を疑う）
止めるときは stop（type=event, 宛先 "*"）を 1 回だけ出す。

並列の環境（env）はそれぞれ別の箱として扱う。メッセージの payload の "env"（無ければ 0）で振り分ける。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .bus import Bus, Message
from .runlog import OUTCOMES

TASK = "observer"


@dataclass(frozen=True)
class ObserverConfig:
    max_consecutive_failures: int = 3
    barcode_unreadable_limit: int = 3
    shelf_capacity: int = 3

    def __post_init__(self) -> None:
        for k in ("max_consecutive_failures", "barcode_unreadable_limit", "shelf_capacity"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} は 1 以上")


@dataclass
class BoxState:
    books: list = field(default_factory=list)
    shelf: list = field(default_factory=list)
    discard: list = field(default_factory=list)
    consecutive_failures: int = 0
    consecutive_unreadable: int = 0
    barcode_reads: int = 0
    barcode_unreadable: int = 0
    feedback: Counter = field(default_factory=Counter)
    stopped: bool = False
    stop_reason: str | None = None


class Observer:
    def __init__(self, bus: Bus, config: ObserverConfig | None = None):
        self.bus = bus
        self.config = config or ObserverConfig()
        self.boxes: dict[int, BoxState] = {}
        bus.subscribe(self.on_message)  # 宛先を問わず全部

    def box(self, env: int) -> BoxState:
        return self.boxes.setdefault(int(env), BoxState())

    def is_stopped(self, env: int = 0) -> bool:
        return self.box(env).stopped

    def on_message(self, msg: Message) -> None:
        if msg.from_task == TASK:
            return
        env = int(msg.payload.get("env", 0))
        st = self.box(env)
        if msg.type == "feedback":
            st.feedback[(msg.topic, msg.from_task, msg.to_task)] += 1
        if msg.topic == "barcode.read":
            st.barcode_reads += 1
            if msg.payload["ok"]:
                st.consecutive_unreadable = 0
            else:
                st.barcode_unreadable += 1
                st.consecutive_unreadable += 1
                if st.consecutive_unreadable >= self.config.barcode_unreadable_limit:
                    self._stop(env, "barcode_unreadable", {"consecutive": st.consecutive_unreadable})
        elif msg.topic == "book.result":
            self._on_book_result(env, st, msg.payload)

    def _on_book_result(self, env: int, st: BoxState, p: dict) -> None:
        outcome = p["outcome"]
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome は {OUTCOMES} のどれか: {outcome!r}")
        st.books.append(dict(p))
        if outcome == "succeeded":
            st.consecutive_failures = 0
            if p.get("destination") == "shelf":
                st.shelf.append(dict(p))
            elif p.get("destination") == "discard":
                st.discard.append(dict(p))
        else:
            st.consecutive_failures += 1
        if st.consecutive_failures >= self.config.max_consecutive_failures:
            self._stop(env, "consecutive_failures", {"consecutive": st.consecutive_failures})
        elif len(st.shelf) >= self.config.shelf_capacity:
            self._stop(env, "shelf_full", {"shelf_count": len(st.shelf), "capacity": self.config.shelf_capacity})

    def _stop(self, env: int, reason: str, detail: dict) -> None:
        st = self.box(env)
        if st.stopped:
            return
        st.stopped, st.stop_reason = True, reason
        self.bus.publish(TASK, "*", "event", "stop", {"env": env, "reason": reason, **detail})

    # ---------- 報告書 ----------

    def report_markdown(self, env: int, title: str = "") -> str:
        st = self.box(env)
        n = Counter(b["outcome"] for b in st.books)
        value = sum(int(b.get("value_jpy") or 0) for b in st.shelf)
        lines = [
            f"# 箱の報告書{(' — ' + title) if title else ''}",
            "",
            f"- run_id: `{self.bus.run_id}`",
            f"- 箱（環境）: {env}",
            f"- 処理した本: {len(st.books)} 冊（成功 {n['succeeded']} / 不合格 {n['rejected']} / 異常 {n['failed']}）",
            f"- 停止: {'あり — ' + str(st.stop_reason) if st.stopped else 'なし'}",
            f"- バーコード: 読み取り {st.barcode_reads} 回、うち読めず {st.barcode_unreadable} 回",
            f"- 査定額の合計（架空の仮値。棚に置いた本のみ）: {value:,} 円",
            "",
            "## 本棚",
            "",
            *self._book_table(st.shelf),
            "",
            "## 廃棄箱",
            "",
            *self._book_table(st.discard),
            "",
            "## 不合格・異常",
            "",
            *self._book_table([b for b in st.books if b["outcome"] != "succeeded"], with_reason=True),
            "",
            "## 工程間のフィードバック",
            "",
        ]
        if st.feedback:
            lines += ["| 種類 | 送り手 → 宛先 | 件数 |", "|---|---|---:|"]
            for (topic, src, dst), c in sorted(st.feedback.items()):
                lines.append(f"| {topic} | {src} → {dst} | {c} |")
        else:
            lines.append("まだありません。修正依頼や好評価が出ると、ここに件数が出ます。")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _book_table(rows: list, with_reason: bool = False) -> list[str]:
        if not rows:
            return ["まだありません。"]
        head = "| book_id | 題名 | ISBN | 査定額（仮） |" + (" 結果 | 理由 |" if with_reason else "")
        sep = "|---|---|---|---:|" + ("---|---|" if with_reason else "")
        out = [head, sep]
        for b in rows:
            line = (f"| {b.get('book_id', '')} | {b.get('title', '')} | {b.get('isbn') or '（読めず）'} "
                    f"| {int(b.get('value_jpy') or 0):,} |")
            if with_reason:
                line += f" {b['outcome']} | {b.get('reason', '')} |"
            out.append(line)
        return out

    def write_reports(self, out_dir: Path, title: str = "") -> list[Path]:
        paths = []
        for env in sorted(self.boxes):
            p = Path(out_dir) / f"observer_report_env{env}.md"
            p.write_text(self.report_markdown(env, title), encoding="utf-8")
            paths.append(p)
        return paths
