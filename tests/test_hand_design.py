"""手の形のパラメータ・検査・MJCF の生成（Genesis なし）。"""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pai_book_intake import hand_design as H
from pai_book_intake.hand_design import V1, V2, HandDesign


def _body(xml: str) -> ET.Element:
    return ET.fromstring(xml.strip())


def _finger(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.split()]


def test_v2_xml_matches_trial3_geometry():
    # 試行3〜4 の v2（右の指先に 25 mm・厚さ 3 mm の爪、0.14 rad 反らせた板と、3.5 mm 浮かせた丸い先端）
    root = _body(H.gripper_xml(V2))
    tip = _finger(root, "finger_r_tip")
    geoms = tip.findall("geom")
    assert len(geoms) == 3  # 指の板・爪の板・丸い先端
    plate, lip, cap = geoms
    assert _floats(plate.get("size")) == pytest.approx([0.03, 0.006, 0.07])
    assert _floats(lip.get("size")) == pytest.approx([0.03, 0.0125, 0.0015])
    assert _floats(lip.get("pos")) == pytest.approx([0.0, 0.0125, 0.1415])
    assert _floats(lip.get("euler")) == pytest.approx([-0.14, 0.0, 0.0])
    assert _floats(cap.get("fromto")) == pytest.approx([-0.03, 0.025, 0.138, 0.03, 0.025, 0.138])
    assert lip.get("friction") == "0.5 0.01 0.001"
    # 左の指は爪なし
    assert len(_finger(root, "finger_l").findall("geom")) == 1
    palm = [g for g in root.findall("geom") if g.get("name") == "palm"][0]
    assert _floats(palm.get("size")) == pytest.approx([0.03, 0.13, 0.02])
    assert H.asset_xml(V2) == ""


def test_v1_has_no_lip_and_joint_order_is_fixed():
    for d in (V1, V2, HandDesign(name="b", lips="both", tip="bevel", bevel_deg=20, upturn_rad=0.0, lip_t_m=0.002,
                                 lip_len_m=0.03, finger_open_m=0.13)):
        root = _body(H.gripper_xml(d))
        joints = [j.get("name") for j in root.iter("joint")]
        # dof の並び（腕の後ろ）: 左指 / 右指 / 右指の上下 / プランジャ。tool.py の FINGERS・LIFT・PLUNGER が前提にする
        assert joints == ["finger_l_joint", "finger_r_joint", "finger_r_lift", "plunger_joint"]
    assert len(_finger(_body(H.gripper_xml(V1)), "finger_r_tip").findall("geom")) == 1


def test_joint_ranges_follow_design():
    d = HandDesign(name="wide", finger_open_m=0.14, lift_stroke_m=0.02)
    root = _body(H.gripper_xml(d))
    rng = {j.get("name"): _floats(j.get("range")) for j in root.iter("joint")}
    assert rng["finger_l_joint"] == pytest.approx([0, 0.14])
    assert rng["finger_r_joint"] == pytest.approx([0, 0.14])
    assert rng["finger_r_lift"] == pytest.approx([-0.02, 0.02])
    palm = [g for g in root.findall("geom") if g.get("name") == "palm"][0]
    assert _floats(palm.get("size"))[1] == pytest.approx(0.14 + H.PALM_MARGIN)


def _mesh_verts(d: HandDesign, side: str) -> np.ndarray:
    for m in ET.fromstring("<asset>" + H.asset_xml(d) + "</asset>"):
        if m.get("name") == f"lip_{side}_{d.name}":
            return np.array(_floats(m.get("vertex"))).reshape(-1, 3)
    raise KeyError(side)


def test_bevel_lip_is_flat_bottom_wedge():
    d = HandDesign(name="w", tip="bevel", bevel_deg=20.0, upturn_rad=0.0, lip_len_m=0.04, lip_t_m=0.003)
    v = _mesh_verts(d, "r")
    fl = d.finger_len_m
    assert v[:, 0].min() == pytest.approx(-0.03) and v[:, 0].max() == pytest.approx(0.03)  # 幅 = lip_w
    assert v[:, 1].min() == pytest.approx(0.0) and v[:, 1].max() == pytest.approx(0.04)  # 右の指は +y へ
    assert v[:, 2].max() == pytest.approx(fl + 0.003)  # 下面（置く面の側）は平ら
    # 先端の厚みは刃先の厚みだけ
    tip = v[np.isclose(v[:, 1], 0.04)]
    assert tip[:, 2].max() - tip[:, 2].min() == pytest.approx(H.TIP_EDGE_T)
    # 削ぎの長さ = (厚み − 刃先) / tan(角)
    assert d.bevel_len_m == pytest.approx((0.003 - H.TIP_EDGE_T) / math.tan(math.radians(20)))
    assert H.asset_xml(d).count("<mesh") == 1


def test_both_lips_mirror_and_upturn_raises_tip():
    d = HandDesign(name="bb", lips="both", tip="square", upturn_rad=0.1, lip_len_m=0.03, lip_t_m=0.002,
                   finger_open_m=0.12)
    r, left = _mesh_verts(d, "r"), _mesh_verts(d, "l")
    assert np.allclose(r[:, [0, 2]], left[:, [0, 2]]) and np.allclose(r[:, 1], -left[:, 1])
    bot = d.finger_len_m + d.lip_t_m
    tip_bottom = r[np.argmax(r[:, 1]), 2]
    # 先端は手のひら側（-z）へ上がる: 根元の下の角を中心に回した分
    assert bot - tip_bottom == pytest.approx(r[:, 1].max() * math.tan(0.1), rel=1e-3)
    root = _body(H.gripper_xml(d))
    assert _finger(root, "finger_l").find("geom[@type='mesh']").get("mesh") == "lip_l_bb"


@pytest.mark.parametrize("kw, msg", [
    ({"lips": "left"}, "lips"),
    ({"tip": "spoon"}, "tip"),
    ({"lip_len_m": 0.13}, "lip_len_m"),
    ({"lip_t_m": 0.001}, "lip_t_m"),
    ({"tip": "bevel", "bevel_deg": 5.0}, "bevel_deg"),
    ({"tip": "round", "bevel_deg": 10.0}, "bevel_deg"),
    ({"tip": "bevel", "bevel_deg": 8.0, "lip_len_m": 0.01, "lip_t_m": 0.004}, "削ぎの長さ"),
    ({"lips": "both", "lip_len_m": 0.10}, "張り出し"),
    ({"lip_len_m": 0.12, "lip_t_m": 0.0015, "finger_open_m": 0.16, "upturn_rad": 0.0}, "たわみ"),
    ({"finger_open_m": 0.2}, "finger_open_m"),
])
def test_validation_rejects_bad_designs(kw, msg):
    with pytest.raises(ValueError, match=msg):
        HandDesign(name="bad", **kw)


def test_strength_estimate_cantilever():
    d = HandDesign(name="s", tip="square", upturn_rad=0.0, lip_len_m=0.08, lip_t_m=0.002, lip_w_m=0.06,
                   finger_open_m=0.105)
    defl, stress = d.lip_strength()
    p = H.DESIGN_BOOK_KG * H.G * H.DYNAMIC_FACTOR
    inertia = 0.06 * 0.002 ** 3 / 12
    assert defl == pytest.approx(p * 0.08 ** 3 / (3 * H.E_STEEL_PA * inertia))
    assert stress == pytest.approx(p * 0.08 * 0.001 / inertia)
    assert defl < H.MAX_TIP_DEFLECTION_M
    # 両方の指に付けると 1 枚あたりの荷重は半分
    d2 = d.with_(name="s2", lips="both", lip_len_m=0.04, finger_open_m=0.105)
    assert d2.lip_strength()[0] == pytest.approx(p / 2 * 0.04 ** 3 / (3 * H.E_STEEL_PA * inertia))


def test_span_and_fit():
    d = HandDesign(name="f", lips="both", tip="square", upturn_rad=0.0, lip_len_m=0.03, finger_open_m=0.12)
    assert d.required_span(0.105, 0.008) == pytest.approx(0.105 + 0.06 + 0.016)
    assert d.fits(0.148, 0.008)  # 148 + 60 + 16 = 224 ≤ 240
    assert not d.fits(0.182, 0.008)  # 258 > 240
    assert V2.required_span(0.182, 0.008) == pytest.approx(0.182 + 0.025 + 0.016)
    assert V1.lip_bottom_z == pytest.approx(V1.fingertip_z)
    assert V2.lip_bottom_z == pytest.approx(0.183)


def test_insert_into_robot_xml_adds_mesh_assets():
    base = '<mujoco><asset><mesh name="m" file="a.stl"/></asset><worldbody><body name="b">' \
           '<site name="attachment_site"/></body></worldbody></mujoco>'
    d = HandDesign(name="ins", tip="bevel", bevel_deg=20.0, upturn_rad=0.0)
    out = H.insert_into_robot_xml(base, d)
    root = ET.fromstring(out)
    assert [m.get("name") for m in root.find("asset")] == ["lip_r_ins", "m"]
    assert root.find(".//body[@name='gripper']") is not None
    with pytest.raises(ValueError, match="attachment_site"):
        H.insert_into_robot_xml("<mujoco/>", d)
    assert H.insert_into_robot_xml(base, V2).count("<mesh") == 1  # round は メッシュを足さない


def test_file_names_keep_legacy():
    assert H.xml_file_name(V2) == "panda_book_gripper_v2.xml"
    assert H.xml_file_name(V1) == "panda_book_gripper.xml"
    assert H.xml_file_name(HandDesign(name="x1", lip_len_m=0.04)) == "panda_book_gripper_x1.xml"


def test_lowest_point_and_probe():
    # v2: 反らせた板の根元の角がいちばん先に出る（指先 0.18 + 厚み 3 mm より 1.7 mm 先）
    assert V2.lip_lowest_z == pytest.approx(0.04 + 0.14 + 0.0015 + 0.0125 * math.sin(0.14) + 0.0015 * math.cos(0.14))
    d = HandDesign(name="p", tip="bevel", bevel_deg=20.0, upturn_rad=0.0, lip_len_m=0.04, lip_t_m=0.002)
    assert d.lip_lowest_z == pytest.approx(d.lip_bottom_z)
    x, y, z = d.lip_probe_local(+1)
    v = _mesh_verts(d, "r")
    # 探る点は爪の中（先端から 4 mm、厚みの範囲の内側）
    assert y == pytest.approx(0.036) and d.finger_len_m < z < d.finger_len_m + d.lip_t_m
    assert v[:, 1].max() > y
    assert d.lip_probe_local(-1)[1] == pytest.approx(-y)
    assert V1.lip_lowest_z == pytest.approx(V1.fingertip_z)


# ---------- 試行8: 背の側で挟む手（PinchHand） ----------

from pai_book_intake.hand_design import PinchHand  # noqa: E402


def _pinch_verts(d: PinchHand, tag: str) -> np.ndarray:
    for m in ET.fromstring("<asset>" + H.pinch_asset_xml(d) + "</asset>"):
        if m.get("name") == f"claw_{tag}_{d.name}":
            return np.array(_floats(m.get("vertex"))).reshape(-1, 3)
    raise KeyError(tag)


def test_pinch_joint_order_and_ranges():
    d = PinchHand()
    root = _body(H.pinch_gripper_xml(d))
    joints = {j.get("name"): _floats(j.get("range")) for j in root.iter("joint")}
    # MJCF の木の順（dof の並び）。trial8 は名前から番号を引くが、並びが変わったら気づけるように固定する
    assert list(joints) == list(H.PINCH_DOF_NAMES)
    assert joints["claw_joint"] == pytest.approx([0, 0.105])
    assert joints["finger_l_joint"] == pytest.approx([0, 0.12])
    assert joints["plate_y_joint"] == pytest.approx([-0.12, 0.12])
    assert joints["plate_z_joint"] == pytest.approx([0, d.plate_stroke_m])
    # 板を下ろし切ると、板の下面が指先の高さに来る
    assert H.PLATE_BASE_Z + d.plate_t_m + d.plate_stroke_m == pytest.approx(d.fingertip_z)


def test_pinch_claws_only_collide_with_books_and_sit_between_pads():
    d = PinchHand()
    root = _body(H.pinch_gripper_xml(d))
    claws = [g for g in root.iter("geom") if (g.get("name") or "").startswith("claw_")]
    assert len(claws) == 2
    for g in claws:
        assert g.get("contype") == str(H.CLAW_BIT) and g.get("conaffinity") == str(H.CLAW_BIT)
    # 台・箱・床のビット（1）と共通のビットを持たない
    assert H.CLAW_BIT & H.ENV_BIT == 0
    pads = {g.get("name"): g for g in root.iter("geom") if (g.get("name") or "").startswith("pad_")}
    assert set(pads) == {"pad_a", "pad_b"}
    for g in pads.values():
        assert _floats(g.get("friction"))[0] == pytest.approx(1.0)
    wall = [g for g in root.iter("geom") if g.get("name") == "wall"][0]
    assert _floats(wall.get("friction"))[0] == pytest.approx(1.0)  # 壁にも同じゴム（A2-3）
    # 爪はゴムの面の内側にあり、重ならない（x）
    for tag, sign in (("a", -1), ("b", +1)):
        v = _pinch_verts(d, tag)
        xs = np.abs(v[:, 0])
        assert xs.max() == pytest.approx(d.span_m / 2 - d.pad_w_m - d.claw_gap_m)
        assert xs.min() == pytest.approx(xs.max() - d.claw_w_m)
        assert np.sign(v[:, 0]).tolist() == [sign] * len(v)
        # 先端は右の指の内面の CLAW_RECESS 手前（引っ込めたとき CLAW_RECESS）、下面は指先と同じ高さ、厚みは 3 mm
        assert v[:, 1].max() == pytest.approx(-H.CLAW_RECESS)
        assert v[:, 2].max() == pytest.approx(d.finger_len_m)
        assert v[:, 2].max() - v[:, 2].min() == pytest.approx(d.claw_t_m)
        # 出し切っても根元は内面の後ろに残る
        assert v[:, 1].min() + d.claw_stroke_m < 0


def test_pinch_claw_strength_matches_cantilever_and_limits():
    d = PinchHand()
    p = H.DESIGN_BOOK_KG * H.G * H.DYNAMIC_FACTOR / 2
    inertia = d.claw_w_m * d.claw_t_m ** 3 / 12
    defl, stress = d.claw_strength()
    assert defl == pytest.approx(p * 0.105 ** 3 / (3 * H.E_STEEL_PA * inertia))
    assert defl < H.MAX_TIP_DEFLECTION_M  # 3 mm 厚・105 mm で 0.93 mm（02_確認事項 A2-2 の表）
    assert defl * 1000 == pytest.approx(0.93, abs=0.01)
    # 2 mm 厚では 100 mm 出すと 2.7 mm たわむので不可
    with pytest.raises(ValueError, match="たわみ"):
        PinchHand(name="thin", claw_t_m=0.002, claw_stroke_m=0.10)
    # 3 mm 厚でも 110 mm は 1.07 mm で不可
    with pytest.raises(ValueError, match="たわみ"):
        PinchHand(name="long", claw_stroke_m=0.11)


@pytest.mark.parametrize("kw, msg", [
    ({"rubber_mu": 0.1}, "rubber_mu"),
    ({"span_m": 0.20}, "span_m"),
    ({"claw_w_m": 0.03}, "重なる"),
    ({"claw_t_m": 0.001}, "claw_t_m"),
    ({"finger_open_m": 0.05}, "finger_open_m"),
])
def test_pinch_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        PinchHand(name="bad", **kw)


def test_pinch_probe_is_inside_claw_and_fit():
    d = PinchHand()
    x, y, z = d.claw_probe_local(+1)
    v = _pinch_verts(d, "b")
    assert v[:, 0].min() < x < v[:, 0].max()
    assert v[:, 1].min() < y < v[:, 1].max()
    # 探る点の高さが、その y での爪の上面と下面の間
    assert d.finger_len_m - d.claw_t_m < z < d.finger_len_m
    assert d.fits(0.182, 0.012, 0.012)  # B5 と両側 12 mm で 206 ≤ 240
    assert not d.fits(0.182, 0.04, 0.04)


def test_pinch_inserts_meshes_and_file_name():
    base = '<mujoco><asset><mesh name="m" file="a.stl"/></asset><worldbody><body name="b">' \
           '<site name="attachment_site"/></body></worldbody></mujoco>'
    d = PinchHand(name="P1")
    root = ET.fromstring(H.insert_into_robot_xml(base, d))
    assert [m.get("name") for m in root.find("asset")] == ["claw_a_P1", "claw_b_P1", "m"]
    assert root.find(".//body[@name='plate']") is not None
    assert H.xml_file_name(d) == "panda_book_gripper_P1.xml"
