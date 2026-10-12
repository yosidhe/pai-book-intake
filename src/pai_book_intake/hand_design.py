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


def insert_into_robot_xml(robot_xml: str, d: HandDesign) -> str:
    """panda_nohand.xml の文字列に手を取り付ける（attachment_site の直後に body、<asset> にメッシュ）。"""
    anchor = '<site name="attachment_site"/>'
    if anchor not in robot_xml:
        raise ValueError("attachment_site が無い（Genesis の版が想定と違う）")
    out = robot_xml.replace(anchor, anchor + gripper_xml(d), 1)
    assets = asset_xml(d)
    if assets:
        if "<asset>" not in out:
            raise ValueError("<asset> が無い（メッシュの爪を足せない）")
        out = out.replace("<asset>", "<asset>\n" + assets, 1)
    return out


def xml_file_name(d: HandDesign) -> str:
    """生成する MJCF のファイル名。v1・v2 は従来の名前のまま。"""
    if d == V2:
        return "panda_book_gripper_v2.xml"
    if d == V1:
        return "panda_book_gripper.xml"
    return f"panda_book_gripper_{d.name}.xml"
