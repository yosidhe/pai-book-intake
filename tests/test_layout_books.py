"""置き方の標準・配置・本の仕様・査定スタブ（Genesis なし）。"""

import json
import math

import numpy as np
import pytest

from pai_book_intake import layout as L
from pai_book_intake.appraisal import appraise
from pai_book_intake.books import (DEFAULT_CATALOG, PLACEHOLDER_BOOKS, PLACEHOLDER_CATALOG, BookSpec, Catalog, ean13,
                                   grip_plan, is_valid_isbn13, load_catalog)


@pytest.mark.parametrize("w, oh", [(0.130, 0.0455), (0.105, 0.03675), (0.080, 0.030), (0.148, 0.050), (0.21, 0.050)])
def test_overhang_rule(w, oh):
    assert L.overhang_for(w) == pytest.approx(oh)


@pytest.mark.parametrize("b", PLACEHOLDER_BOOKS, ids=lambda b: b.book_id)
def test_handoff_pose_is_standard_and_stable(b):
    pose = L.handoff_pose(b.width_m, b.thickness_m)
    # 標準どおりの位置・向きを測れば誤差 0 で許容内
    chk = L.check_placement(pose.center_world, pose.width_axis_yaw_world, b.width_m, b.height_m, b.thickness_m)
    assert chk.within_tolerance and chk.on_table
    assert chk.overhang_m == pytest.approx(pose.overhang_m)
    # 重心は台の上に 15 mm 以上残る
    assert b.width_m / 2 - pose.overhang_m >= 0.015 - 1e-9
    # 横幅の向きは B の向き（台の縁に直交）
    assert pose.width_axis_yaw_world == pytest.approx(L.ROBOT_B_YAW)


def test_placement_tolerance_boundaries():
    b = PLACEHOLDER_BOOKS[0]
    pose = L.handoff_pose(b.width_m, b.thickness_m)
    toward_b = np.array([0.0, 1.0, 0.0])  # B は +y 側。B の方へずらすとはみ出しが増える
    ok = L.check_placement(pose.center_world + toward_b * 0.0099, pose.width_axis_yaw_world, *b.size_m)
    ng = L.check_placement(pose.center_world + toward_b * 0.0101, pose.width_axis_yaw_world, *b.size_m)
    assert ok.within_tolerance and not ng.within_tolerance
    # 向きの誤差は ±180° 同値（軸に向きはない）
    flipped = L.check_placement(pose.center_world, pose.width_axis_yaw_world + math.pi, *b.size_m)
    assert abs(flipped.yaw_err_rad) < 1e-9
    tilted = L.check_placement(pose.center_world, pose.width_axis_yaw_world + math.radians(9), *b.size_m)
    assert not tilted.within_tolerance
    lifted = L.check_placement(pose.center_world + np.array([0, 0, 0.05]), pose.width_axis_yaw_world, *b.size_m)
    assert not lifted.on_table


def test_layout_has_no_overlaps_between_stations():
    # 台・本棚・廃棄箱・箱が床の上で重ならない（外形の矩形で確かめる）
    def rect_from_local(frame, xs, ys):
        pts = np.array([frame.to_world_xy((x, y)) for x in xs for y in ys])
        return pts.min(0), pts.max(0)

    table = rect_from_local(L.FRAME_B, (L.TABLE_EDGE_B, L.TABLE_EDGE_B + L.TABLE_DEPTH),
                            (-L.TABLE_HALF_W, L.TABLE_HALF_W))
    shelf = rect_from_local(L.FRAME_B, (L.SHELF_SLOT_X[0] - L.SHELF_SLOT_HALF, L.SHELF_SLOT_X[-1] + L.SHELF_SLOT_HALF),
                            (L.SHELF_DEPTH_C - L.SHELF_DEPTH / 2, L.SHELF_DEPTH_C + L.SHELF_DEPTH / 2))
    bin_ = rect_from_local(L.FRAME_B, (L.BIN_C[0] - L.BIN_IN[0] / 2, L.BIN_C[0] + L.BIN_IN[0] / 2),
                           (L.BIN_C[1] - L.BIN_IN[1] / 2, L.BIN_C[1] + L.BIN_IN[1] / 2))
    box = (np.array(L.BOX_C) - 0.22, np.array(L.BOX_C) + 0.22)
    rects = [table, shelf, bin_, box]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            (a0, a1), (b0, b1) = rects[i], rects[j]
            assert (a1 < b0).any() or (b1 < a0).any(), (i, j)


def test_shelf_and_bin_checks():
    t = 0.02
    for s in range(L.shelf_capacity()):
        assert L.on_shelf(L.shelf_slot_center(s, t), t)
    c = L.bin_center_world()
    assert L.in_bin(c + np.array([0, 0, 0.03]))
    assert not L.in_bin(c + np.array([0.5, 0, 0.03]))


def test_ean13_and_isbn():
    assert ean13("978400000001") == "9784000000017"
    assert is_valid_isbn13("9784000000017")
    assert not is_valid_isbn13("9784000000018")
    with pytest.raises(ValueError):
        ean13("12345")


def test_book_spec_validation():
    b = PLACEHOLDER_BOOKS[0]
    assert b.size_m == (0.130, 0.190, 0.020)
    assert b.density == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        BookSpec(**{**b.to_dict(), "isbn13": "9784000000018"})
    with pytest.raises(ValueError):
        BookSpec(**{**b.to_dict(), "surface": "glossy"})
    with pytest.raises(ValueError):
        BookSpec(**{**b.to_dict(), "width_m": 0.0})
    with pytest.raises(ValueError):
        BookSpec(**{**b.to_dict(), "condition": "good"})


def test_load_real_catalog_maps_nested_fields():
    cat = load_catalog(DEFAULT_CATALOG)
    raw = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    assert len(cat.books) == len(raw["books"]) == 10
    assert cat.shelf_threshold_yen == raw["decision_rule"]["shelf_threshold_yen"]
    for b, r in zip(cat.books, raw["books"]):
        ph, ap = r["physical"], r["appraisal"]
        assert b.book_id == r["id"] and b.isbn13 == r["isbn13"] and b.title == r["bib"]["title"]
        assert b.width_m == pytest.approx(ph["dims_mm"]["w"] / 1000)
        assert b.height_m == pytest.approx(ph["dims_mm"]["h"] / 1000)
        assert b.thickness_m == pytest.approx(ph["dims_mm"]["t"] / 1000)
        assert b.weight_kg == pytest.approx(ph["weight_g_true"] / 1000)  # 物理は本当の重さ
        assert b.weight_estimated_kg == pytest.approx(ph["weight_g_estimated"] / 1000)
        assert b.surface == ph["surface"] and b.condition == ap["condition"]
        # 査定スタブはカタログの規則と正解を再現する
        d = appraise(b.isbn13, cat)
        assert d.destination == ap["expected_decision"] and d.value_jpy == ap["buyback_yen"]


def test_catalog_rule_mismatch_is_rejected(tmp_path):
    raw = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    raw["books"][0]["appraisal"]["expected_decision"] = (
        "shelf" if raw["books"][0]["appraisal"]["expected_decision"] == "discard" else "discard")
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_catalog(p)


def test_grip_plan_uses_estimated_weight():
    cat = load_catalog(DEFAULT_CATALOG)
    heavy = cat.by_id("book-10")  # 約 1.5 kg: 滑らかな本の吸着上限 12 N を超える
    plan = grip_plan(heavy)
    assert plan["est_weight_n"] == pytest.approx(heavy.weight_estimated_kg * 9.81, abs=1e-3)
    assert plan["suction_can_lift"] is False
    assert grip_plan(cat.by_id("book-03"))["suction_can_lift"] is True
    assert grip_plan(cat.by_id("book-01"))["grip_limit_n"] == 25.0  # rough


def test_appraisal_stub_boundaries():
    cat = PLACEHOLDER_CATALOG
    assert appraise(None, cat).reason == "isbn_unread"
    assert appraise("9780000000002", cat).destination == "discard"
    d = appraise(cat.by_id("ph01").isbn13, cat)
    assert d.destination == "shelf" and d.value_jpy == 160 and d.book_id == "ph01"
    edge = Catalog((BookSpec(**{**cat.by_id("ph01").to_dict(), "buyback_yen": 50}),), 50)
    below = Catalog((BookSpec(**{**cat.by_id("ph01").to_dict(), "buyback_yen": 49}),), 50)
    assert appraise(edge.books[0].isbn13, edge).destination == "shelf"  # 50 円ちょうどは棚
    assert appraise(below.books[0].isbn13, below).destination == "discard"


def test_down_quat_branch_picks_wrist_angle_near_hint():
    from pai_book_intake.geometry import down_quat_branch, down_quat_with_finger_axis, wrist_psi

    for axis in (0.0, 1.2, -2.5):
        for hint in (-3.0, 0.4, 2.9):
            q = down_quat_branch(np.array(axis), np.array(hint))
            psi = float(wrist_psi(q))
            # 指の開閉軸は同じ（±180° 同値）で、手首の角度は hint から 90° 以内
            ref = float(wrist_psi(down_quat_with_finger_axis(np.array(axis))))
            assert abs(float(np.sin(psi - ref))) < 1e-9
            assert abs((psi - hint + math.pi) % (2 * math.pi) - math.pi) <= math.pi / 2 + 1e-9
