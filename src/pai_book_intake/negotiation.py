"""工程どうしの交渉ルール（純粋関数）と、成功確率の推定器（今はスタブ）。

ルール
  修正依頼: 後工程は、受け取った状態での自分の成功確率の推定が 0.80 未満なら、前工程へ修正依頼を出す
           （0.80 ちょうどは出さない。0.7999 は出す）
  好評価:   後工程は、自分の作業時間が「自分の過去の作業時間の 平均 − 1σ」以下なら、前工程へ好評価を出す
           - σ は標本標準偏差（n−1 で割る）。過去の記録には今回の分を含めない
           - 過去の記録が PRAISE_MIN_HISTORY（5）件未満のあいだは好評価を出さない（ばらつきが測れないため）
           - 過去が全部同じ値（σ = 0）なら「平均以下」で好評価になる

工程の並び（前 → 後）: robot-a-pick → inspect-flip → robot-b-sort。appraisal は脇から判断だけ返す
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

FIX_REQUEST_THRESHOLD = 0.80
PRAISE_MIN_HISTORY = 5
PRAISE_K_SIGMA = 1.0

UPSTREAM = {"inspect-flip": "robot-a-pick", "robot-b-sort": "inspect-flip"}


def _check_probability(p: float) -> float:
    p = float(p)
    if math.isnan(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"成功確率は 0〜1: {p!r}")
    return p


def should_request_fix(p_success: float, threshold: float = FIX_REQUEST_THRESHOLD) -> bool:
    return _check_probability(p_success) < threshold


def praise_threshold(history: Sequence[float], min_history: int = PRAISE_MIN_HISTORY,
                     k_sigma: float = PRAISE_K_SIGMA) -> float | None:
    """好評価の基準時間（平均 − kσ）。記録が足りなければ None。"""
    if min_history < 2:
        raise ValueError("標本標準偏差には 2 件以上が要る")
    xs = [float(x) for x in history]
    if any(math.isnan(x) or x < 0 for x in xs):
        raise ValueError(f"作業時間は 0 以上の数: {xs!r}")
    if len(xs) < min_history:
        return None
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean - k_sigma * math.sqrt(var)


def should_praise(task_time: float, history: Sequence[float], min_history: int = PRAISE_MIN_HISTORY,
                  k_sigma: float = PRAISE_K_SIGMA) -> bool:
    th = praise_threshold(history, min_history, k_sigma)
    return th is not None and float(task_time) <= th


# ---------- 成功確率の推定 ----------

class SuccessEstimator(Protocol):
    def estimate(self, features: Mapping[str, float]) -> float: ...

    def update(self, features: Mapping[str, float], success: bool) -> None: ...


@dataclass
class BinnedSuccessEstimator:
    """受け取った本の姿勢の特徴量を粗い区分（ビン）に分け、区分ごとの成功率を返すスタブ。

    各ビンは (成功数, 試行数) を持ち、推定値は (成功 + prior_a) / (試行 + prior_a + prior_b)。
    事前分布（prior）は仮の値で、実測が溜まるほど実測に寄る。
    TODO(P2): 学習した推定器（姿勢・表面・判型を入力にした分類器）に置き換える。インターフェースは同じ
    """

    # 区分の境界: はみ出しの誤差(mm) と 向きの誤差(度)。置き方の標準の許容差（±10 mm, ±8°）を最初の境界にする
    overhang_edges_mm: tuple[float, ...] = (10.0, 20.0)
    yaw_edges_deg: tuple[float, ...] = (8.0, 15.0)
    # 区分ごとの事前の成功率（仮）。[はみ出しの区分][向きの区分]
    prior_rate: tuple[tuple[float, ...], ...] = ((0.95, 0.85, 0.50), (0.85, 0.80, 0.40), (0.50, 0.40, 0.20))
    prior_weight: float = 10.0
    counts: dict = field(default_factory=dict)

    @staticmethod
    def _bin(value: float, edges: Sequence[float]) -> int:
        for i, e in enumerate(edges):
            if value <= e:
                return i
        return len(edges)

    def key(self, features: Mapping[str, float]) -> tuple[int, int]:
        if not features.get("on_table", True):
            return (-1, -1)
        return (self._bin(abs(features["overhang_err_mm"]), self.overhang_edges_mm),
                self._bin(abs(features["yaw_err_deg"]), self.yaw_edges_deg))

    def estimate(self, features: Mapping[str, float]) -> float:
        k = self.key(features)
        if k == (-1, -1):
            return 0.0  # 台に載っていない本は後工程で扱えない
        rate = self.prior_rate[k[0]][k[1]]
        s, n = self.counts.get(k, (0, 0))
        return (s + rate * self.prior_weight) / (n + self.prior_weight)

    def update(self, features: Mapping[str, float], success: bool) -> None:
        k = self.key(features)
        s, n = self.counts.get(k, (0, 0))
        self.counts[k] = (s + int(bool(success)), n + 1)


@dataclass
class TaskTimer:
    """工程ごとの作業時間の記録（好評価の判断に使う）。"""

    history: dict = field(default_factory=dict)

    def record_and_judge(self, task: str, task_time: float) -> bool:
        """今回の時間で好評価に当たるかを判定してから、記録に足す。"""
        past = self.history.setdefault(task, [])
        praise = should_praise(task_time, past)
        past.append(float(task_time))
        return praise
