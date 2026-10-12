"""検品エリア（縁の低いトレー + 真上のカメラ）とロボット A のやりとり。Genesis に依存しない。

流れ（本 1 冊・環境 1 つごと）
  1. robot-a-pick が本をトレーに平らに置く
  2. inspect-scan が真上から撮ってバーコードを探し、結果を robot-a-pick へ送る（scan.result）
       {"barcode_visible": bool, "isbn": str|None, "barcode_yaw_deg": float|None, "attempt": int}
     attempt は 0 から数える（0 = 置いた直後、n = n 回目の裏返しの後）
  3. 見えなければ robot-a-pick へ裏返しを依頼する（feedback / fix_request {"action": "flip", "attempt": n}）。
     n は何回目の裏返しか（1 から）。robot-a-pick は裏返したら flip.done を返し、2 へ戻る
  4. 裏返しを MAX_FLIPS（2）回しても見えなければ、オブザーバへ上げる
     （event / escalation {"reason": "barcode_unreadable", ...}）。この本の検品はそこで終わり、人の判断を待つ

barcode_yaw_deg は、カメラ画像の中でのバーコードの読む向き（数字の左 → 右）の角度。
画像の右を 0°、上を 90° とする反時計回り（vision.read_isbn_with_yaw と同じ定義）。
世界の向きへ直すのは image_yaw_to_world（カメラの「上」の向きを渡す）。

メッセージの形は bus.py の約束どおり（修正依頼は feedback、それ以外は event）。
オブザーバの止める条件（barcode.read が続けて読めない＝カメラの故障を疑う）とは別の話なので、
ここでは barcode.read を出さない（同じ本の撮り直しで、箱ごと止まらないように）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .books import is_valid_isbn13
from .bus import Bus

TASK = "inspect-scan"
ROBOT = "robot-a-pick"
OBSERVER = "observer"
MAX_FLIPS = 2
ESCALATION_REASON = "barcode_unreadable"

# 次にすること（InspectionSession.state）
WAIT_SCAN = "wait_scan"  # 撮影待ち
WAIT_FLIP = "wait_flip"  # 裏返しを依頼して、終わるのを待っている
READ = "read"  # 読めた（終わり）
ESCALATED = "escalated"  # 読めないまま上げた（終わり）


@dataclass(frozen=True)
class ScanResult:
    barcode_visible: bool
    isbn: str | None
    barcode_yaw_deg: float | None
    attempt: int

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 0:
            raise ValueError(f"attempt は 0 以上の整数: {self.attempt!r}")
        if self.barcode_visible:
            if self.isbn is None or not is_valid_isbn13(self.isbn):
                raise ValueError(f"見えたのに ISBN が不正: {self.isbn!r}")
            if self.barcode_yaw_deg is None or not math.isfinite(self.barcode_yaw_deg):
                raise ValueError(f"見えたのに向きが無い: {self.barcode_yaw_deg!r}")
        elif self.isbn is not None or self.barcode_yaw_deg is not None:
            raise ValueError("見えていないのに ISBN か向きが入っている")

    def to_payload(self) -> dict:
        return {"barcode_visible": bool(self.barcode_visible), "isbn": self.isbn,
                "barcode_yaw_deg": None if self.barcode_yaw_deg is None else round(float(self.barcode_yaw_deg), 2),
                "attempt": int(self.attempt)}


def scan_result(isbn: str | None, yaw_deg: float | None, attempt: int) -> ScanResult:
    """読み取りの結果（isbn が None なら見えなかった）。向きは (-180, 180] に畳む。"""
    if isbn is None:
        return ScanResult(False, None, yaw_deg, attempt)  # 向きだけ入っていれば ScanResult が拒む
    if yaw_deg is None:
        raise ValueError("ISBN が読めたのに向きが無い（read_isbn_with_yaw の戻り値をそのまま渡す）")
    return ScanResult(True, isbn, wrap_deg(yaw_deg), attempt)


def wrap_deg(a: float) -> float:
    """角度を (-180, 180] に畳む。"""
    a = math.fmod(float(a), 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def image_yaw_to_world(yaw_img_deg: float, cam_up_world_yaw_deg: float) -> float:
    """真下を向いたカメラの画像の中の角度（右 0°・上 90°・反時計回り）を、世界の yaw（度）へ直す。
    カメラの「上」が世界の yaw = cam_up_world_yaw_deg を向くとき、画像の右は世界の yaw = 上 − 90°。"""
    return wrap_deg(cam_up_world_yaw_deg - 90.0 + float(yaw_img_deg))


@dataclass
class InspectionSession:
    """本 1 冊・環境 1 つぶんの検品。撮るたびに on_scan を呼ぶと、必要なメッセージを bus に出し、次の状態を返す。"""

    bus: Bus
    env: int
    book_id: str
    max_flips: int = MAX_FLIPS
    flips_requested: int = 0
    flips_done: int = 0
    state: str = WAIT_SCAN
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_flips < 0:
            raise ValueError("max_flips は 0 以上")

    @property
    def attempt(self) -> int:
        """次に撮るのは何回目か（0 = 置いた直後、n = n 回目の裏返しの後）。"""
        return self.flips_done

    def on_scan(self, isbn: str | None, yaw_img_deg: float | None, source: str = "camera") -> str:
        if self.state != WAIT_SCAN:
            raise RuntimeError(f"撮影を受け付けない状態: {self.state}（裏返しの完了を先に on_flip_done で知らせる）")
        res = scan_result(isbn, yaw_img_deg, self.attempt)
        self.history.append(res)
        self.bus.publish(TASK, ROBOT, "event", "scan.result",
                         {"env": int(self.env), "book_id": self.book_id, "source": source, **res.to_payload()})
        if res.barcode_visible:
            self.state = READ
        elif self.flips_requested < self.max_flips:
            self.flips_requested += 1
            self.bus.publish(TASK, ROBOT, "feedback", "fix_request",
                             {"env": int(self.env), "book_id": self.book_id, "action": "flip",
                              "attempt": self.flips_requested})
            self.state = WAIT_FLIP
        else:
            self.bus.publish(TASK, OBSERVER, "event", "escalation",
                             {"env": int(self.env), "book_id": self.book_id, "reason": ESCALATION_REASON,
                              "scans": len(self.history), "flips": self.flips_done})
            self.state = ESCALATED
        return self.state

    def on_flip_done(self, ok: bool = True, detail: dict | None = None) -> str:
        """robot-a-pick が裏返しを終えた（ok=False は動作が失敗した＝本が手から落ちたなど）。
        どちらでも撮り直す（裏返しの成否はカメラで確かめる。ロボットの「できた」は証拠にしない）。"""
        if self.state != WAIT_FLIP:
            raise RuntimeError(f"裏返しを依頼していない状態: {self.state}")
        self.flips_done += 1
        self.bus.publish(ROBOT, TASK, "event", "flip.done",
                         {"env": int(self.env), "book_id": self.book_id, "attempt": self.flips_done, "ok": bool(ok),
                          **(detail or {})})
        self.state = WAIT_SCAN
        return self.state

    @property
    def done(self) -> bool:
        return self.state in (READ, ESCALATED)
