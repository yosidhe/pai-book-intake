"""物理的にありえるかの判定（Genesis に依存しない。シミュレーションの外から毎回の測定値を渡す）。

背景（試行4 の不採用、2026-10-12）: 本を変形しない剛体の箱として扱っているので、シミュレーションでは
現実の本ではありえない姿勢・動き（1 冊で長い辺で立つ、角で傾いたまま止まる）が「成功」に混ざる。
そこで、本が手にも壁にも支えられていない間の姿勢と接触を毎回見て、ありえない状態が続いたら失敗にする。

規則（本ごと・環境ごと）
  edge_rest   手にも壁にも触れておらず、台・床・ほかの本に触れていて（＝何かの上に載っていて）、
              傾き（本の厚みの軸と鉛直の角度。表裏は問わない）が 20° を超える状態が、一瞬でも出た
              （2026-10-12 ユーザー決定。当初は 0.3 s まで許していた。rest_tilt_s で続く時間の許容を変えられる）。
              現実の本は、支えなしに辺や角で立てない（ページが開き、表紙がしなって倒れる）。
              剛体の箱だけが、辺の上で釣り合ったり、辺に載ったまま滑ったり回ったりできる。
              壁に立てかけた本は数えない（現実にもある）。判定は測るたび（試行5 は 0.02 s ごと）
  airborne    何にも触れていない状態が 0.25 s を超えて続いた（0.25 s の自由落下は約 30 cm。
              この作業で本を 30 cm 落とすことはないので、取り落とした・宙に浮いたのどちらか）
  penetration 手（指・爪・吸盤）が本に 2 mm を超えてめり込んだ（指が本の中に入った＝現実ではありえない）
              ※ 接触の深さは物理エンジンの値。めり込みの深さが接触として出ない（すり抜け）場合に備え、
                呼び出し側で爪の先端が本の中にあるかも別に調べて inside を渡す
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("edge_rest", "airborne", "penetration")


@dataclass(frozen=True)
class Rules:
    rest_tilt_deg: float = 20.0
    rest_tilt_s: float = 0.0  # 続く時間の許容（0 = 一瞬でも違反）
    airborne_s: float = 0.25
    penetration_m: float = 0.002

    def describe(self) -> dict:
        return {
            "edge_rest": (f"手にも壁にも触れず何かの上に載ったまま、傾き > {self.rest_tilt_deg}° が"
                          + ("一瞬でも出る" if self.rest_tilt_s == 0 else f" {self.rest_tilt_s} s を超えて続く")),
            "airborne": f"何にも触れない状態が {self.airborne_s} s を超えて続く（取り落とし・宙に浮く）",
            "penetration": f"手が本に {self.penetration_m * 1000:.0f} mm を超えてめり込む、または爪の先端が本の中にある",
        }


def tilt_deg(up_z) -> np.ndarray:
    """本の厚みの軸（本の z 軸）の鉛直成分から、平置きからの傾き（度）。表裏は問わない（0〜90°）。"""
    return np.degrees(np.arccos(np.clip(np.abs(np.asarray(up_z, dtype=float)), 0.0, 1.0)))


class Monitor:
    """毎回の測定（形は (n_envs, n_books)）を update に渡す。違反は環境ごと・種類ごとに、最初の時刻と本の番号を残す。"""

    def __init__(self, n_envs: int, n_books: int, rules: Rules = Rules()):
        self.rules = rules
        self.n, self.m = n_envs, n_books
        self.rest_t = np.zeros((n_envs, n_books))
        self.air_t = np.zeros((n_envs, n_books))
        self.violated = {k: np.zeros(n_envs, bool) for k in KINDS}
        self.first_t = {k: np.full(n_envs, np.nan) for k in KINDS}
        self.first_book = {k: np.full(n_envs, -1) for k in KINDS}
        self.max_rest_tilt_deg = np.zeros(n_envs)  # 支えなしで載っている間の傾きの最大（参考）
        self.max_rest_tilt_t = np.full(n_envs, np.nan)  # その時刻と本の番号（どこで傾いたかを後で調べるため）
        self.max_rest_tilt_book = np.full(n_envs, -1)
        self.max_rest_s = np.zeros(n_envs)
        self.max_air_s = np.zeros(n_envs)
        self.max_penetration_m = np.zeros(n_envs)
        self.peak_hand_force_n = np.zeros(n_envs)

    def update(self, t: float, dt: float, up_z, hand, wall, support, penetration_m=None, inside=None,
               hand_force_n=None) -> None:
        """dt は前回の update からの時間。hand / wall / support は「手 / 壁 / 台・床・ほかの本に触れているか」。
        penetration_m は手と本の接触の深さの最大、inside は爪の先端が本の中にあるか、hand_force_n は手から本への力の合計。"""
        sh = (self.n, self.m)
        tilt = tilt_deg(np.broadcast_to(up_z, sh))
        hand, wall, support = (np.broadcast_to(np.asarray(a, bool), sh) for a in (hand, wall, support))
        resting = support & ~hand & ~wall & (tilt > self.rules.rest_tilt_deg)
        self.rest_t = np.where(resting, self.rest_t + dt, 0.0)
        air = ~(hand | wall | support)
        self.air_t = np.where(air, self.air_t + dt, 0.0)
        self.max_rest_s = np.maximum(self.max_rest_s, self.rest_t.max(axis=1))
        self.max_air_s = np.maximum(self.max_air_s, self.air_t.max(axis=1))
        free_tilt = np.where(support & ~hand & ~wall, tilt, 0.0)
        worse = free_tilt.max(axis=1) > self.max_rest_tilt_deg
        self.max_rest_tilt_t = np.where(worse, t, self.max_rest_tilt_t)
        self.max_rest_tilt_book = np.where(worse, free_tilt.argmax(axis=1), self.max_rest_tilt_book)
        self.max_rest_tilt_deg = np.maximum(self.max_rest_tilt_deg, free_tilt.max(axis=1))
        self._flag("edge_rest", t, self.rest_t > self.rules.rest_tilt_s)
        self._flag("airborne", t, self.air_t > self.rules.airborne_s)
        pen = np.zeros(sh) if penetration_m is None else np.broadcast_to(np.asarray(penetration_m, float), sh)
        ins = np.zeros(sh, bool) if inside is None else np.broadcast_to(np.asarray(inside, bool), sh)
        self.max_penetration_m = np.maximum(self.max_penetration_m, pen.max(axis=1))
        self._flag("penetration", t, (pen > self.rules.penetration_m) | ins)
        if hand_force_n is not None:
            f = np.broadcast_to(np.asarray(hand_force_n, float), sh)
            self.peak_hand_force_n = np.maximum(self.peak_hand_force_n, f.max(axis=1))

    def _flag(self, kind: str, t: float, hit: np.ndarray) -> None:
        env_hit = hit.any(axis=1)
        new = env_hit & ~self.violated[kind]
        if new.any():
            self.first_t[kind][new] = t
            self.first_book[kind][new] = np.argmax(hit[new], axis=1)
        self.violated[kind] |= env_hit

    @property
    def ok(self) -> np.ndarray:
        """環境ごとに、どの規則にも違反していないか。"""
        bad = np.zeros(self.n, bool)
        for k in KINDS:
            bad |= self.violated[k]
        return ~bad

    def per_env(self) -> dict:
        return {
            "plausible": self.ok,
            **{f"violated_{k}": self.violated[k] for k in KINDS},
            **{f"first_t_{k}": self.first_t[k] for k in KINDS},
            **{f"first_book_{k}": self.first_book[k] for k in KINDS},
            "max_unsupported_tilt_deg": self.max_rest_tilt_deg,
            "max_unsupported_tilt_t": self.max_rest_tilt_t,
            "max_unsupported_tilt_book": self.max_rest_tilt_book,
            "max_edge_rest_s": self.max_rest_s,
            "max_airborne_s": self.max_air_s,
            "max_penetration_mm": self.max_penetration_m * 1000,
            "peak_hand_force_n": self.peak_hand_force_n,
        }
