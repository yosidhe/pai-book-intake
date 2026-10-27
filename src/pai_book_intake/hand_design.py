"""手の形（自作グリッパ）のパラメータと、その MJCF の生成。Genesis に依存しない（テストできるように）。

試行5（2026-10-12 のユーザー判断）: 本は吸引で持ち上げない。本の下に滑り込ませる手で下から支えて持ち上げる。
爪の長さを固定せず、手の形を複数試して最適な「手のひら」を探すため、形をパラメータにした。

座標（gripper リンク座標。z が手先方向。真下に向けると z が下）
  - 左の指は +y へ開き、内面は y = 0（指の座標）で -y を向く。右の指は -y へ開き、内面は +y を向く
  - 爪（lip）は指先（z = finger_len）のさらに先に厚み lip_t で付き、指の内側へ lip_len だけ張り出す。
    本の下に入れたとき、本の底は爪の上面（指先側の面 z = finger_len）に載る
  - 右の指には上下（z）に ±lift_stroke 動く関節がある（v2 と同じ）。左の指には無い（dof の並びを変えないため）

先端の形（tip）
  round  v2 と同じ。爪の板を反らせ（upturn_rad）、先端に丸棒（半径 lip_t/2）を 3.5 mm 浮かせて付ける
  bevel  下面は平ら、上面を先端に向けて削いだくさび（先端の厚み TIP_EDGE_T）。上面の削ぎ角が bevel_deg。
         upturn_rad > 0 なら根元を中心に先端を手のひら側へ反らせる
  square 下面も上面も平らな板（先端は角）

材料の仮定: 爪は SUS304 ステンレスの板（ヤング率 193 GPa・耐力 215 MPa）。シミュレーションでは剛体なので、
validate() で「本の重さを受けたときのたわみと応力」を片持ち梁で見積もり、剛体の仮定が大きく崩れる形を弾く。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

# v2 と共通の寸法（gripper リンク座標）
FINGER_BASE_Z = 0.04
FINGER_PLATE_T = 0.012  # 指の板の厚み（y）
CUP_TIP_RETRACTED_Z = 0.14
PLUNGER_STROKE = 0.08
PALM_MARGIN = 0.025  # 手のひらの箱は、指の最大の開きより片側 2.5 cm 広い（v2: 0.105 + 0.025 = 0.13）
ROUND_TIP_RAISE = 0.0035  # v2 の丸い先端の、爪の板の中心からの浮かせ量
LIP_FRICTION = "0.5 0.01 0.001"  # 爪（金属）の摩擦。v2 と同じ値（調整しない）
LIP_RGBA = "0.95 0.75 0.20 1"

# 材料（SUS304）と、強さの見積もりの条件
E_STEEL_PA = 193e9
YIELD_STEEL_PA = 215e6
DESIGN_BOOK_KG = 1.6  # 想定する最も重い本（カタログの B5 1.5 kg に余裕を持たせる）
DYNAMIC_FACTOR = 2.0  # 運ぶときの加速度・衝撃の上乗せ
MAX_TIP_DEFLECTION_M = 0.001  # 先端のたわみの上限（これを超えると剛体の仮定が崩れる）
STRESS_SAFETY = 2.0  # 耐力に対する安全率
TIP_EDGE_T = 0.0003  # くさびの先端の厚み（研いだ刃先 0.3 mm）
G = 9.81

TIPS = ("round", "bevel", "square")
LIPS = ("none", "right", "both")


@dataclass(frozen=True)
class HandDesign:
    name: str = "v2"
    lips: str = "right"  # 爪を付ける指: none / right / both
    lip_len_m: float = 0.025  # 指の内面から内側への張り出し
    lip_t_m: float = 0.003  # 爪の板の厚み
    lip_w_m: float = 0.06  # 爪の幅（指に沿った向き x。指のパッドより広くしてよい＝へら）
    tip: str = "round"
    bevel_deg: float = 0.0  # tip=bevel のときの上面の削ぎ角
    upturn_rad: float = 0.14  # 先端の反り（v2 は 0.14 rad）
    finger_pad_w_m: float = 0.06  # 指のパッドの幅（x）
    finger_len_m: float = 0.14
    finger_open_m: float = 0.105  # 指 1 本あたりの最大の開き
    lift_stroke_m: float = 0.03  # 右の指の上下の関節の可動域（±）
    note: str = ""

    def __post_init__(self) -> None:
        errs = self.problems()
        if errs:
            raise ValueError(f"手の形 {self.name!r} が不正: " + " / ".join(errs))

    # ---------- 検査 ----------

    def problems(self) -> list[str]:
        """形の不正・物理的に無理な点の一覧（空なら正しい）。"""
        e: list[str] = []
        if self.lips not in LIPS:
            e.append(f"lips は {LIPS} のどれか: {self.lips!r}")
        if self.tip not in TIPS:
            e.append(f"tip は {TIPS} のどれか: {self.tip!r}")
        if not 0.10 <= self.finger_len_m <= 0.20:
            e.append(f"finger_len_m は 0.10〜0.20: {self.finger_len_m}")
        if not 0.06 <= self.finger_open_m <= 0.16:
            e.append(f"finger_open_m は 0.06〜0.16: {self.finger_open_m}")
        if not 0.02 <= self.finger_pad_w_m <= 0.12:
            e.append(f"finger_pad_w_m は 0.02〜0.12: {self.finger_pad_w_m}")
        if not 0.0 <= self.lift_stroke_m <= 0.05:
            e.append(f"lift_stroke_m は 0〜0.05: {self.lift_stroke_m}")
        if self.lips == "none" or e:
            return e
        if not 0.005 <= self.lip_len_m <= 0.12:
            e.append(f"lip_len_m は 0.005〜0.12: {self.lip_len_m}")
        if not 0.0015 <= self.lip_t_m <= 0.005:
            e.append(f"lip_t_m は 1.5〜5 mm: {self.lip_t_m}")
        if not 0.02 <= self.lip_w_m <= 0.14:
            e.append(f"lip_w_m は 0.02〜0.14: {self.lip_w_m}")
        if not 0.0 <= self.upturn_rad <= 0.35:
            e.append(f"upturn_rad は 0〜0.35: {self.upturn_rad}")
        if self.tip == "bevel":
            if not 8.0 <= self.bevel_deg <= 45.0:
                e.append(f"bevel_deg は 8〜45°: {self.bevel_deg}")
            elif self.bevel_len_m > self.lip_len_m:
                e.append(f"削ぎの長さ {self.bevel_len_m * 1000:.1f} mm が爪の長さを超える")
        elif self.bevel_deg != 0.0:
            e.append("bevel_deg は tip=bevel のときだけ指定する")
        # 指を開き切ったとき、爪の先どうし（片側なら爪の先と左の指）の間に 2 cm 以上の隙間が要る
        reach = self.lip_len_m * (2 if self.lips == "both" else 1)
        if reach > 2 * self.finger_open_m - 0.02:
            e.append(f"爪の張り出しの合計 {reach * 1000:.0f} mm が指の開き {2 * self.finger_open_m * 1000:.0f} mm − 20 mm を超える")
        defl, stress = self.lip_strength()
        if defl > MAX_TIP_DEFLECTION_M:
            e.append(f"爪の先端のたわみ {defl * 1000:.2f} mm が上限 {MAX_TIP_DEFLECTION_M * 1000:.1f} mm を超える（剛体の仮定が崩れる）")
        if stress > YIELD_STEEL_PA / STRESS_SAFETY:
            e.append(f"爪の根元の応力 {stress / 1e6:.0f} MPa が耐力/安全率 {YIELD_STEEL_PA / STRESS_SAFETY / 1e6:.0f} MPa を超える")
        return e

    def lip_strength(self) -> tuple[float, float]:
        """(先端のたわみ m, 根元の曲げ応力 Pa)。想定の本の重さ × 動的係数を、爪の本数で分けて先端にかけた片持ち梁。"""
        if self.lips == "none":
            return 0.0, 0.0
        n = 2 if self.lips == "both" else 1
        p = DESIGN_BOOK_KG * G * DYNAMIC_FACTOR / n
        b, t, length = self.lip_w_m, self.lip_t_m, self.lip_len_m
        inertia = b * t ** 3 / 12
        return p * length ** 3 / (3 * E_STEEL_PA * inertia), p * length * (t / 2) / inertia

    # ---------- 寸法 ----------

    @property
    def fingertip_z(self) -> float:
        return FINGER_BASE_Z + self.finger_len_m

    @property
    def cup_tip_extended_z(self) -> float:
        return CUP_TIP_RETRACTED_Z + PLUNGER_STROKE

    @property
    def lip_bottom_z(self) -> float:
        """爪の下面（本の下に入れたとき、置く面に接する側）の gripper z。爪が無ければ指先。"""
        return self.fingertip_z + (self.lip_t_m if self.lips != "none" else 0.0)

    @property
    def lip_lowest_z(self) -> float:
        """爪のいちばん先（置く面の側）の gripper z。round は反らせた板の根元の角が最も出る。"""
        if self.lips == "none":
            return self.fingertip_z
        if self.tip == "round":
            a, half_l, half_t = self.upturn_rad, self.lip_len_m / 2, self.lip_t_m / 2
            return FINGER_BASE_Z + self.finger_len_m + half_t + half_l * math.sin(a) + half_t * math.cos(a)
        return self.lip_bottom_z  # くさび・平板は根元の下の角を中心に反らせるので、下面の根元が最も出る

    def lip_probe_local(self, sign: int, back_m: float = 0.004) -> tuple[float, float, float]:
        """爪の先端から back_m だけ根元寄りの、板の厚みの中央の点（指のリンク座標。右 sign=+1・左 -1）。
        本の下に入ったか・本の中にめり込んだかを、この点が本の箱のどこにあるかで調べる。"""
        y = max(self.lip_len_m - back_m, self.lip_len_m / 2)
        if self.tip == "round":
            yc, zc, a = self.lip_len_m / 2, self.finger_len_m + self.lip_t_m / 2, self.upturn_rad
            return 0.0, sign * y, zc - (y - yc) * math.sin(a)
        bot = self.finger_len_m + self.lip_t_m
        zm = self.finger_len_m + self.lip_t_m / 2
        if self.tip == "bevel" and y > self.lip_len_m - self.bevel_len_m:
            # 削いだ部分では、上面と下面の中央
            top = self.finger_len_m + (y - (self.lip_len_m - self.bevel_len_m)) * math.tan(math.radians(self.bevel_deg))
            zm = (top + bot) / 2
        c, s = math.cos(self.upturn_rad), math.sin(self.upturn_rad)
        return 0.0, sign * (y * c + (zm - bot) * s), bot - y * s + (zm - bot) * c

    @property
    def bevel_len_m(self) -> float:
        if self.tip != "bevel" or self.bevel_deg <= 0:
            return 0.0
        return (self.lip_t_m - TIP_EDGE_T) / math.tan(math.radians(self.bevel_deg))

    @property
    def lip_len_r(self) -> float:
        return self.lip_len_m if self.lips in ("right", "both") else 0.0

    @property
    def lip_len_l(self) -> float:
        return self.lip_len_m if self.lips == "both" else 0.0

    def required_span(self, book_w: float, margin: float) -> float:
        """本の横幅 book_w の両側に指を置き、爪の先も本の外（margin）に出しておくのに要る、左右の指の内面の間隔。"""
        return book_w + self.lip_len_r + self.lip_len_l + 2 * margin

    def fits(self, book_w: float, margin: float) -> bool:
        return self.required_span(book_w, margin) <= 2 * self.finger_open_m

    def params(self) -> dict:
        d = asdict(self)
        d.update(bevel_len_mm=round(self.bevel_len_m * 1000, 2),
                 tip_deflection_mm=round(self.lip_strength()[0] * 1000, 3),
                 root_stress_mpa=round(self.lip_strength()[1] / 1e6, 1))
        return d

    def with_(self, **kw) -> "HandDesign":
        return replace(self, **kw)


V1 = HandDesign(name="v1", lips="none", lip_len_m=0.025, lip_t_m=0.003, note="爪なし（試行2）")
V2 = HandDesign(name="v2", note="右の指先に 25 mm の丸い爪（試行3）")


# ---------- MJCF ----------

def _f(v: float) -> str:
    return f"{v:.6g}"


def _lip_prism(d: HandDesign, sign: int) -> list[tuple[float, float, float]]:
    """くさび・平板の爪の頂点（指の座標）。sign=+1 は右の指（+y へ張り出す）、-1 は左の指。"""
    fl, t, length, w = d.finger_len_m, d.lip_t_m, d.lip_len_m, d.lip_w_m
    top, bot = fl, fl + t  # top = 手のひら側の面（本の底が載る面）、bot = 置く面に接する面
    if d.tip == "bevel":
        yb = length - d.bevel_len_m
        section = [(0.0, top), (yb, top), (length, bot - TIP_EDGE_T), (length, bot), (0.0, bot)]
    else:
        section = [(0.0, top), (length, top), (length, bot), (0.0, bot)]
    # 反り: 根元の下の角 (0, bot) を中心に、先端を手のひら側（-z）へ回す
    c, s = math.cos(d.upturn_rad), math.sin(d.upturn_rad)
    out = []
    for x in (-w / 2, w / 2):
        for y, z in section:
            yr = y * c + (z - bot) * s
            zr = bot - y * s + (z - bot) * c
            out.append((x, sign * yr, zr))
    return out


def asset_xml(d: HandDesign) -> str:
    """<asset> に足すメッシュ（くさび・平板の爪）。round と爪なしでは空。"""
    if d.lips == "none" or d.tip == "round":
        return ""
    out = []
    for side, sign in (("r", +1), ("l", -1)):
        if side == "l" and d.lips != "both":
            continue
        verts = " ".join(f"{_f(x)} {_f(y)} {_f(z)}" for x, y, z in _lip_prism(d, sign))
        out.append(f'<mesh name="lip_{side}_{d.name}" vertex="{verts}"/>')
    return "\n".join(out)


def _lip_geoms(d: HandDesign, sign: int) -> str:
    if d.lips == "none" or (sign < 0 and d.lips != "both"):
        return ""
    if d.tip == "round":
        # v2 と同じ式（sign で左右を反転する）
        ll, lt, fl = d.lip_len_m, d.lip_t_m, d.finger_len_m
        yc, zc = sign * ll / 2, fl + lt / 2
        return (f'<geom type="box" size="{d.lip_w_m / 2} {ll / 2} {lt / 2}" pos="0 {yc} {zc}" euler="{-sign * d.upturn_rad} 0 0"'
                f' friction="{LIP_FRICTION}" rgba="{LIP_RGBA}"/>'
                f'<geom type="capsule" fromto="{-d.lip_w_m / 2} {sign * ll:.4f} {zc - ROUND_TIP_RAISE:.4f} {d.lip_w_m / 2} {sign * ll:.4f} {zc - ROUND_TIP_RAISE:.4f}"'
                f' size="{lt / 2}" friction="{LIP_FRICTION}" rgba="{LIP_RGBA}"/>')
    side = "r" if sign > 0 else "l"
    return f'<geom name="lip_{side}" type="mesh" mesh="lip_{side}_{d.name}" friction="{LIP_FRICTION}" rgba="{LIP_RGBA}"/>'


def gripper_xml(d: HandDesign) -> str:
    """自作グリッパの MJCF（attachment_site に付ける body）。dof の並びは設計によらず
    左指 / 右指 / 右指の上下 / プランジャ（腕の 7 つの後ろ）。"""
    fl, fo, pw = d.finger_len_m, d.finger_open_m, d.finger_pad_w_m / 2
    # v2 の文字列と同じ書式になるよう、v2 の値は v2 と同じ式で書く（tests で確かめる）
    left_lip = _lip_geoms(d, -1)  # 左の指の爪は内側（-y）へ張り出す。内面はどちらの指も y = 0
    right_lip = _lip_geoms(d, +1)
    return f"""
<body name="gripper" pos="0 0 0">
  <inertial mass="0.6" pos="0 0 0.03" diaginertia="0.002 0.002 0.001"/>
  <geom name="palm" type="box" size="0.03 {_f(fo + PALM_MARGIN)} 0.02" pos="0 0 0.02" rgba="0.30 0.30 0.32 1"/>
  <body name="finger_l" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.05" pos="0 0.006 0.07" diaginertia="1e-4 1e-4 1e-5"/>
    <joint name="finger_l_joint" type="slide" axis="0 1 0" range="0 {fo}" damping="5" armature="0.1"/>
    <geom type="box" size="{_f(pw)} 0.006 {fl / 2}" pos="0 0.006 {fl / 2}"
          friction="1.5 0.01 0.001" rgba="0.85 0.85 0.85 1"/>
    {left_lip}
  </body>
  <body name="finger_r" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.02" pos="0 0 0" diaginertia="1e-5 1e-5 1e-5"/>
    <joint name="finger_r_joint" type="slide" axis="0 -1 0" range="0 {fo}" damping="5" armature="0.1"/>
    <body name="finger_r_tip" pos="0 0 0">
      <inertial mass="0.05" pos="0 -0.006 0.07" diaginertia="1e-4 1e-4 1e-5"/>
      <joint name="finger_r_lift" type="slide" axis="0 0 1" range="{-d.lift_stroke_m} {d.lift_stroke_m}" damping="5" armature="0.1"/>
      <geom type="box" size="{_f(pw)} 0.006 {fl / 2}" pos="0 -0.006 {fl / 2}"
            friction="1.5 0.01 0.001" rgba="0.85 0.85 0.85 1"/>
      {right_lip}
    </body>
  </body>
  <body name="plunger" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.03" pos="0 0 0.05" diaginertia="2e-5 2e-5 1e-6"/>
    <joint name="plunger_joint" type="slide" axis="0 0 1" range="0 {PLUNGER_STROKE}" damping="5" armature="0.1"/>
    <geom type="cylinder" size="0.005 0.045" pos="0 0 0.045" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
    <geom name="cup" type="cylinder" size="0.016 0.005" pos="0 0 {CUP_TIP_RETRACTED_Z - FINGER_BASE_Z - 0.005}"
          friction="1.0 0.01 0.001" rgba="0.10 0.40 0.90 1"/>
  </body>
</body>
"""


def insert_into_robot_xml(robot_xml: str, d: "HandDesign | PinchHand") -> str:
    """panda_nohand.xml の文字列に手を取り付ける（attachment_site の直後に body、<asset> にメッシュ）。"""
    anchor = '<site name="attachment_site"/>'
    if anchor not in robot_xml:
        raise ValueError("attachment_site が無い（Genesis の版が想定と違う）")
    pinch = isinstance(d, PinchHand)
    out = robot_xml.replace(anchor, anchor + (pinch_gripper_xml(d) if pinch else gripper_xml(d)), 1)
    assets = pinch_asset_xml(d) if pinch else asset_xml(d)
    if assets:
        if "<asset>" not in out:
            raise ValueError("<asset> が無い（メッシュの爪を足せない）")
        out = out.replace("<asset>", "<asset>\n" + assets, 1)
    return out


def xml_file_name(d: "HandDesign | PinchHand") -> str:
    """生成する MJCF のファイル名。v1・v2 は従来の名前のまま。"""
    if d == V2:
        return "panda_book_gripper_v2.xml"
    if d == V1:
        return "panda_book_gripper.xml"
    return f"panda_book_gripper_{d.name}.xml"


# =====================================================================
# 試行8: 背の側で挟む手（PinchHand）。2026-10-17・18 のユーザー判断（Obsidian 02_確認事項 A0〜A5）
#
#   右の指（背の側）: 外側 2 本 = ゴムの面（pad）。真ん中 2 本 = 爪（claw）。爪は内側（+y）へ出す関節で動き、
#                    出す量がそのまま爪の長さになる（本ごとに中心を 10 mm 越える所まで）
#   左の指（小口の側）: ゴムを貼った壁。小口の側を下から支えたり、こじったりしない（コミックは小口が弱い）
#   板（plate）: 手のひらから下りて、本を上から押さえ、爪と挟む。手のひらに付けた横（y）と縦（z）の 2 つの関節で動かす。
#                (i) 爪の真上（背の側）に下ろす／(ii) 本の中央に下ろす、は制御だけの違い（実機の (ii) は小口の側の
#                「┐」形の腕だが、運動学的には同じ）
#   吸盤: 残すが、本を押さえて滑らせないためだけに使う（持ち上げにも、動かすにも使わない）
#
# ゴムで浮かせる仕組み: ゴムと爪が同じ指に固定されていると、手ごと上げても爪と本の位置関係は変わらず、爪は入らない。
#   そこで爪は右の指の中の上下の台（finger_r_tip、関節 finger_r_lift）に載せる。手を 5 mm 上げる間、台を 5 mm 下げ、
#   爪だけを置く面の高さに残す
#
# 爪の衝突: 爪は出し入れする板なので、引っ込めると板の後ろが指の外にはみ出す（実物は伸び縮みする構造で、はみ出さない）。
#   シミュでは爪を「本とだけ当たる」ようにして（contype/conaffinity のビット 2）、はみ出した部分が台・箱・床に
#   当たらないようにする。そのかわり台・箱・床は contype=conaffinity=1 で作ること（本は既定の 0xFFFF のまま）
#
# dof の並び（腕の後ろ。MJCF の木の順）: 左指 / 右指 / 爪の上下 / 爪の出し入れ / プランジャ / 板の横 / 板の縦。
#   tool.py の FINGERS・LIFT・PLUNGER の番号（v1・v2 の並び）とは違うので、名前から引く（pinch_dof_names）
# =====================================================================

RUBBER_RGBA = "0.15 0.15 0.15 1"
CLAW_RGBA = "0.95 0.75 0.20 1"
PLATE_RGBA = "0.25 0.45 0.70 1"
CLAW_RECESS = 0.003  # 引っ込めた爪の先と、右の指のゴムの面（内面）の隙間。1 mm では、ゴムが本に 0.85 mm めり込んだぶん爪の先が背の面にかかり、その摩擦で本が浮かなかった（試行8）
CLAW_BIT = 2  # 爪の衝突のビット（本だけがこのビットを持つ）
ENV_BIT = 1  # 台・箱・床の衝突のビット
PLATE_BASE_Z = FINGER_BASE_Z + 0.002  # 引っ込めた板の上面（gripper z）
PINCH_DOF_NAMES = ("finger_l_joint", "finger_r_joint", "finger_r_lift", "claw_joint", "plunger_joint",
                   "plate_y_joint", "plate_z_joint")


@dataclass(frozen=True)
class PinchHand:
    name: str = "P1"
    rubber_mu: float = 1.0  # ゴム（右の指の pad・左の壁・板の下面）の摩擦係数。シミュでは 2 面の大きい方が使われる
    span_m: float = 0.10  # 4 本の先の全体の幅（天地の向き x）。文庫（148 mm）でも本の中に収まる
    pad_w_m: float = 0.02  # ゴムの面 1 本の幅
    claw_w_m: float = 0.015  # 爪 1 本の幅
    claw_gap_m: float = 0.005  # ゴムの面と爪の隙間
    claw_t_m: float = 0.003  # 爪の厚み（3 mm。出す量 105 mm でたわみ 1 mm 以内）
    claw_bevel_deg: float = 20.0  # 爪の先の上面の削ぎ角（B15 と同じ）
    claw_stroke_m: float = 0.105  # 爪を出せる量（B5 の中心＋10 mm = 101 mm）
    finger_len_m: float = 0.16
    finger_open_m: float = 0.12  # 指 1 本あたりの最大の開き（B5 182 mm ＋ 両側の隙間）
    wall_t_m: float = 0.009  # 左の壁の厚み（板 6 mm ＋ ゴム 3 mm）
    plate_len_m: float = 0.03  # 板の長さ（本の幅の向き y）
    plate_w_m: float = 0.10  # 板の幅（天地の向き x）
    plate_t_m: float = 0.003
    lift_stroke_m: float = 0.03  # 爪の上下の台の可動域（±）
    note: str = ""

    def __post_init__(self) -> None:
        errs = self.problems()
        if errs:
            raise ValueError(f"手の形 {self.name!r} が不正: " + " / ".join(errs))

    def problems(self) -> list[str]:
        e: list[str] = []
        if not 0.3 <= self.rubber_mu <= 2.0:
            e.append(f"rubber_mu は 0.3〜2.0: {self.rubber_mu}")
        if not 0.06 <= self.span_m <= 0.14:
            e.append(f"span_m は 0.06〜0.14（文庫の天地 148 mm に収まる）: {self.span_m}")
        if self.claw_x_out > self.span_m / 2 - self.pad_w_m - self.claw_gap_m + 1e-9 or self.claw_x_in < 0.0:
            e.append("爪がゴムの面と重なる、または爪どうしが重なる（span・pad_w・claw_w・claw_gap を見直す）")
        if not 0.002 <= self.claw_t_m <= 0.005:
            e.append(f"claw_t_m は 2〜5 mm: {self.claw_t_m}")
        if not 8.0 <= self.claw_bevel_deg <= 45.0:
            e.append(f"claw_bevel_deg は 8〜45°: {self.claw_bevel_deg}")
        if not 0.01 <= self.claw_stroke_m <= 0.12:
            e.append(f"claw_stroke_m は 0.01〜0.12: {self.claw_stroke_m}")
        if not 0.10 <= self.finger_len_m <= 0.20:
            e.append(f"finger_len_m は 0.10〜0.20: {self.finger_len_m}")
        if not 0.06 <= self.finger_open_m <= 0.16:
            e.append(f"finger_open_m は 0.06〜0.16: {self.finger_open_m}")
        if self.claw_stroke_m + CLAW_RECESS > 2 * self.finger_open_m - 0.02:
            e.append("爪を出し切ると左の壁に届く")
        defl, stress = self.claw_strength()
        if defl > MAX_TIP_DEFLECTION_M:
            e.append(f"爪を出し切ったときの先端のたわみ {defl * 1000:.2f} mm が上限 {MAX_TIP_DEFLECTION_M * 1000:.1f} mm を超える")
        if stress > YIELD_STEEL_PA / STRESS_SAFETY:
            e.append(f"爪の根元の応力 {stress / 1e6:.0f} MPa が耐力/安全率 {YIELD_STEEL_PA / STRESS_SAFETY / 1e6:.0f} MPa を超える")
        return e

    def claw_strength(self, extend_m: float | None = None) -> tuple[float, float]:
        """(先端のたわみ m, 根元の曲げ応力 Pa)。爪 2 本で本の重さ × 動的係数を受け、荷重を先端にかけた片持ち梁（厳しめ）。"""
        length = self.claw_stroke_m if extend_m is None else extend_m
        p = DESIGN_BOOK_KG * G * DYNAMIC_FACTOR / 2
        inertia = self.claw_w_m * self.claw_t_m ** 3 / 12
        return p * length ** 3 / (3 * E_STEEL_PA * inertia), p * length * (self.claw_t_m / 2) / inertia

    # ---------- 寸法 ----------

    @property
    def pad_x(self) -> float:
        """ゴムの面の中心の x（±）。"""
        return self.span_m / 2 - self.pad_w_m / 2

    @property
    def claw_x_out(self) -> float:
        return self.span_m / 2 - self.pad_w_m - self.claw_gap_m

    @property
    def claw_x_in(self) -> float:
        return self.claw_x_out - self.claw_w_m

    @property
    def claw_x(self) -> float:
        """爪の中心の x（±）。"""
        return (self.claw_x_out + self.claw_x_in) / 2

    @property
    def fingertip_z(self) -> float:
        """指先（ゴムの面・壁・爪の下面）の gripper z。爪の上下の関節が 0 のとき爪の下面も同じ高さ。"""
        return FINGER_BASE_Z + self.finger_len_m

    @property
    def claw_body_len_m(self) -> float:
        return self.claw_stroke_m + 0.01

    @property
    def claw_bevel_len_m(self) -> float:
        return (self.claw_t_m - TIP_EDGE_T) / math.tan(math.radians(self.claw_bevel_deg))

    @property
    def plate_stroke_m(self) -> float:
        """板を下ろせる量（板の下面が指先の高さまで）。"""
        return self.fingertip_z - (PLATE_BASE_Z + self.plate_t_m)

    def claw_probe_local(self, sign: int, back_m: float = 0.004) -> tuple[float, float, float]:
        """爪（claw リンク座標）の先端から back_m 根元寄り、厚みの中央の点。sign は爪の左右（x の ±）。
        削いだ部分では上面と下面の中央にとる。"""
        y_tip = -CLAW_RECESS
        y = y_tip - back_m
        bot = self.finger_len_m
        top = bot - self.claw_t_m
        if back_m < self.claw_bevel_len_m:
            top = bot - TIP_EDGE_T - (self.claw_t_m - TIP_EDGE_T) * (back_m / self.claw_bevel_len_m)
        return sign * self.claw_x, y, (top + bot) / 2

    def required_open(self, book_w: float, margin_spine: float, margin_fore: float) -> float:
        return book_w + margin_spine + margin_fore

    def fits(self, book_w: float, margin_spine: float, margin_fore: float) -> bool:
        return self.required_open(book_w, margin_spine, margin_fore) <= 2 * self.finger_open_m

    def params(self) -> dict:
        d = asdict(self)
        defl, stress = self.claw_strength()
        d.update(claw_tip_deflection_at_stroke_mm=round(defl * 1000, 3), claw_root_stress_mpa=round(stress / 1e6, 1),
                 pad_x_mm=round(self.pad_x * 1000, 1), claw_x_mm=round(self.claw_x * 1000, 1))
        return d

    def with_(self, **kw) -> "PinchHand":
        return replace(self, **kw)


def _claw_prism(d: PinchHand, xc: float) -> list[tuple[float, float, float]]:
    """爪 1 本の頂点（claw リンク座標）。先端は +y、上面（手のひら側 -z）を先で削いだくさび。下面は指先と同じ高さ。"""
    y_tip = -CLAW_RECESS
    y0 = y_tip - d.claw_body_len_m
    yb = y_tip - d.claw_bevel_len_m
    bot = d.finger_len_m
    top = bot - d.claw_t_m
    section = [(y0, top), (yb, top), (y_tip, bot - TIP_EDGE_T), (y_tip, bot), (y0, bot)]
    hw = d.claw_w_m / 2
    return [(xc + sx * hw, y, z) for sx in (-1, 1) for y, z in section]


def pinch_asset_xml(d: PinchHand) -> str:
    out = []
    for tag, sign in (("a", -1), ("b", +1)):
        verts = " ".join(f"{_f(x)} {_f(y)} {_f(z)}" for x, y, z in _claw_prism(d, sign * d.claw_x))
        out.append(f'<mesh name="claw_{tag}_{d.name}" vertex="{verts}"/>')
    return "\n".join(out)


def pinch_gripper_xml(d: PinchHand) -> str:
    fl, fo = d.finger_len_m, d.finger_open_m
    rub = f'friction="{_f(d.rubber_mu)} 0.01 0.001"'
    claw_col = f'contype="{CLAW_BIT}" conaffinity="{CLAW_BIT}"'
    pads = "\n      ".join(
        f'<geom name="pad_{tag}" type="box" size="{_f(d.pad_w_m / 2)} 0.006 {_f(fl / 2)}" pos="{_f(sx * d.pad_x)} -0.006 {_f(fl / 2)}"'
        f' {rub} rgba="{RUBBER_RGBA}"/>' for tag, sx in (("a", -1), ("b", +1)))
    claws = "\n          ".join(
        f'<geom name="claw_{tag}" type="mesh" mesh="claw_{tag}_{d.name}" friction="{LIP_FRICTION}" {claw_col} rgba="{CLAW_RGBA}"/>'
        for tag in ("a", "b"))
    # 真ん中の 2 本の枠（見た目だけ。本に触れない）
    frame_h = fl - d.claw_t_m - 0.004
    frame = (f'<geom type="box" size="{_f(d.claw_x_out)} 0.003 {_f(frame_h / 2)}" pos="0 -0.009 {_f(frame_h / 2)}"'
             f' contype="0" conaffinity="0" rgba="0.6 0.6 0.6 1"/>')
    return f"""
<body name="gripper" pos="0 0 0">
  <inertial mass="0.7" pos="0 0 0.03" diaginertia="0.002 0.002 0.001"/>
  <geom name="palm" type="box" size="{_f(d.span_m / 2)} {_f(fo + PALM_MARGIN)} 0.02" pos="0 0 0.02" rgba="0.30 0.30 0.32 1"/>
  <body name="finger_l" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.08" pos="0 {_f(d.wall_t_m / 2)} 0.07" diaginertia="1e-4 1e-4 1e-5"/>
    <joint name="finger_l_joint" type="slide" axis="0 1 0" range="0 {_f(fo)}" damping="5" armature="0.1"/>
    <geom name="wall" type="box" size="{_f(d.span_m / 2)} {_f(d.wall_t_m / 2)} {_f(fl / 2)}" pos="0 {_f(d.wall_t_m / 2)} {_f(fl / 2)}"
          {rub} rgba="{RUBBER_RGBA}"/>
  </body>
  <body name="finger_r" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.08" pos="0 -0.006 0.07" diaginertia="1e-4 1e-4 1e-5"/>
    <joint name="finger_r_joint" type="slide" axis="0 -1 0" range="0 {_f(fo)}" damping="5" armature="0.1"/>
    {pads}
    {frame}
    <body name="finger_r_tip" pos="0 0 0">
      <inertial mass="0.02" pos="0 0 {_f(fl)}" diaginertia="1e-5 1e-5 1e-5"/>
      <joint name="finger_r_lift" type="slide" axis="0 0 1" range="{_f(-d.lift_stroke_m)} {_f(d.lift_stroke_m)}" damping="5" armature="0.1"/>
      <body name="claw" pos="0 0 0">
        <inertial mass="0.04" pos="0 {_f(-d.claw_body_len_m / 2)} {_f(fl)}" diaginertia="1e-5 1e-5 1e-5"/>
        <joint name="claw_joint" type="slide" axis="0 1 0" range="0 {_f(d.claw_stroke_m)}" damping="5" armature="0.1"/>
          {claws}
      </body>
    </body>
  </body>
  <body name="plunger" pos="0 0 {FINGER_BASE_Z}">
    <inertial mass="0.03" pos="0 0 0.05" diaginertia="2e-5 2e-5 1e-6"/>
    <joint name="plunger_joint" type="slide" axis="0 0 1" range="0 {PLUNGER_STROKE}" damping="5" armature="0.1"/>
    <geom type="cylinder" size="0.005 0.045" pos="0 0 0.045" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
    <geom name="cup" type="cylinder" size="0.016 0.005" pos="0 0 {CUP_TIP_RETRACTED_Z - FINGER_BASE_Z - 0.005}"
          friction="1.0 0.01 0.001" rgba="0.10 0.40 0.90 1"/>
  </body>
  <body name="plate_carriage" pos="0 0 0">
    <inertial mass="0.02" pos="0 0 {_f(PLATE_BASE_Z)}" diaginertia="1e-5 1e-5 1e-5"/>
    <joint name="plate_y_joint" type="slide" axis="0 1 0" range="{_f(-fo)} {_f(fo)}" damping="5" armature="0.1"/>
    <body name="plate" pos="0 0 0">
      <inertial mass="0.05" pos="0 0 {_f(PLATE_BASE_Z + d.plate_t_m / 2)}" diaginertia="1e-5 1e-5 1e-5"/>
      <joint name="plate_z_joint" type="slide" axis="0 0 1" range="0 {_f(d.plate_stroke_m)}" damping="5" armature="0.1"/>
      <geom name="plate" type="box" size="{_f(d.plate_w_m / 2)} {_f(d.plate_len_m / 2)} {_f(d.plate_t_m / 2)}"
            pos="0 0 {_f(PLATE_BASE_Z + d.plate_t_m / 2)}" {rub} rgba="{PLATE_RGBA}"/>
    </body>
  </body>
</body>
"""
