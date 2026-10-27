import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pai_book_intake.inventory import Inventory, Placement
from pai_book_intake.keeper import (F_BULB, F_T, MAXA_DEG, REST_DEG, KeeperSpec, choose_flat_slot, choose_slot, choose_stack,
                                    finger_tip_rest, flat_book_center_y, flat_fore_edge_max_y, free_spans, keeper_mjcf,
                                    occupancy_from_points, spans_from_books)
from pai_book_intake.site import load_site


def test_site_loads_and_shelf_is_far_from_line():
    site = load_site()
    s = site.shelf()
    assert s.shelf_id == "shelf-01"
    bx, by, _ = site["robot_b"]["station"]
    fx, fy, fyaw = s.front_pose()
    assert math.hypot(fx - bx, fy - by) > 2.0  # 本棚はラインから離れている（台車で移動する）
    # 正面の位置から棚の中央を向く
    c = s.to_world([s.width / 2, s.depth / 2, 0.0])
    assert abs(math.atan2(c[1] - fy, c[0] - fx) - fyaw) < 1e-6


def test_shelf_frame_roundtrip():
    s = load_site().shelf()
    p = np.array([[0.1, 0.05, 0.02], [0.39, 0.18, 0.2]])
    assert np.allclose(s.to_shelf(s.to_world(p)), p)


def test_keeper_matches_scad_and_b4():
    k = KeeperSpec()
    assert k.n_fingers == 64 and abs(k.width - 0.400) < 1e-9 and abs(k.depth - 0.190) < 1e-9  # B4-1・B4-2
    lo, hi = k.usable_x
    assert k.finger_x(0) - F_T / 2 >= lo and k.finger_x(63) + F_T / 2 <= hi
    root = ET.fromstring(keeper_mjcf(k))
    joints = root.findall(".//joint")
    assert len(joints) == 64
    assert joints[0].get("range") == f"{REST_DEG} {MAXA_DEG}"
    # 指は環境の当たり（ENV_BIT）だけ。爪（CLAW_BIT=2）とは当たらない
    assert all(g.get("contype") in ("1", "0") for g in root.findall(".//geom"))


def test_fingers_over_book():
    k = KeeperSpec()
    f = k.fingers_over(0.100, 0.130)  # 30 mm の本
    assert 4 <= len(f) <= 6
    assert all(abs(k.finger_x(i) - 0.115) < 0.02 for i in f)


def test_occupancy_and_free_spans_from_points():
    k = KeeperSpec()
    xs = np.arange(0.0, k.width, 0.001)
    y = np.full_like(xs, k.back_face_y)  # 後ろの壁
    book = (xs > 0.300) & (xs < 0.330)
    y[book] = 0.02  # 背表紙（手前）
    pts = np.stack([xs, y, np.full_like(xs, 0.03)], -1)
    left, occ = occupancy_from_points(pts, k)
    spans = free_spans(left, occ, 0.002, k.usable_x)
    assert len(spans) == 2
    assert abs(spans[0][1] - 0.300) < 0.003 and abs(spans[1][0] - 0.330) < 0.003


def test_unseen_bins_count_as_occupied():
    k = KeeperSpec()
    left, occ = occupancy_from_points(np.zeros((0, 3)), k)
    assert occ.all()
    assert free_spans(left, occ, 0.002, k.usable_x) == []


def test_choose_slot_packs_against_neighbour_on_the_far_side_of_palm():
    k = KeeperSpec()
    spans = spans_from_books([(0.300, 0.330)], k)
    s = choose_slot(spans, 0.020, palm_side=-1, hand_clear=0.17)
    assert s is not None and abs(s[1] - (0.300 - 0.003)) < 1e-9  # 手のひらが左 → 右の本に寄せる
    s2 = choose_slot(spans, 0.020, palm_side=+1, hand_clear=0.17)
    assert s2 is not None and abs(s2[0] - (0.004 + 0.003)) < 1e-9  # 手のひらが右 → 左端に寄せる
    assert choose_slot([(0.0, 0.1)], 0.02, -1, 0.17) is None  # 手が入らない


def test_inventory_records_and_queries(tmp_path):
    inv = Inventory(tmp_path / "inv.sqlite")
    inv.add(Placement(run_id="r1", env=0, shelf_id="shelf-01", book_id="book-03", x_lo_m=0.2, x_hi_m=0.22,
                      fingers=[30, 31], free_space_source="depth_camera", placed_sim_s=12.5, upright=True, inside=True))
    inv.add(Placement(run_id="r1", env=1, shelf_id="shelf-01", book_id="book-03", x_lo_m=0.1, x_hi_m=0.12,
                      fingers=[15], free_space_source="ground_truth", placed_sim_s=12.5))
    rows = inv.on_shelf("shelf-01", env=0)
    assert len(rows) == 1 and rows[0]["fingers"] == [30, 31] and rows[0]["upright"] == 1
    assert len(inv.on_shelf("shelf-01")) == 2
    inv.close()


def test_keeper_rejects_too_many_fingers():
    with pytest.raises(ValueError):
        keeper_mjcf(KeeperSpec(width=0.2, n_fingers=64))


def test_resting_finger_tip_is_forward_and_below_the_pivot():
    k = KeeperSpec()
    y, z = finger_tip_rest(k)
    # 前下がり −35°: 先は軸より手前で下。指の長さ（約 64 mm）より軸から遠くはない
    assert y < k.piv_y and z < k.piv_z
    assert 0.05 < math.hypot(k.piv_y - y, k.piv_z - z) < 0.064
    # 平らな本（厚み 45 mm まで）の上面は、休んでいる指の先より下
    assert z - F_BULB > 0.003 + 0.045


def test_flat_fore_edge_stops_in_front_of_fingers_and_b5_centre_stays_on_shelf():
    k = KeeperSpec()
    y_max = flat_fore_edge_max_y(k, wall_t=0.009)
    assert y_max + 0.009 < finger_tip_rest(k)[0] - F_BULB  # 手の壁の奥の面が指の先より手前
    yc = flat_book_center_y(k, 0.182, 0.009)  # B5 の幅
    assert yc is not None and yc >= 0.015
    assert flat_book_center_y(k, 0.250, 0.009) is None  # 幅が広すぎると重心が棚から出る


def test_choose_flat_slot_and_stack():
    assert choose_flat_slot([(0.004, 0.396)], 0.148) == (0.014, 0.162)
    assert choose_flat_slot([(0.170, 0.396)], 0.257) is None  # 226 mm < 257 + 20
    xr = (0.004, 0.396)
    piles = [{"x_lo": 0.016, "x_hi": 0.164, "top_z": 0.019, "n": 1}, {"x_lo": 0.2, "x_hi": 0.3, "top_z": 0.05, "n": 2}]
    j, xc = choose_stack(piles, 0.257, 0.12 - 0.045, xr)
    assert j == 0  # 上の本が長いほう
    assert xc - 0.257 / 2 >= xr[0] + 0.005 - 1e-9  # 棚の左端からはみ出さないようにずらす
    assert 0.016 + 0.015 <= xc <= 0.164 - 0.015  # ずらしても重心は下の本に載る
    assert choose_stack(piles, 0.257, 0.010, xr) is None  # どちらも高すぎる
    assert choose_stack([{"x_lo": 0.004, "x_hi": 0.06, "top_z": 0.02, "n": 1}], 0.257, 0.1, xr) is None  # 端の小さい本には載らない


def test_occupancy_counts_overhanging_spine_with_y_min():
    k = KeeperSpec()
    xs = np.arange(0.0, k.width, 0.001)
    y = np.full_like(xs, k.back_face_y)
    y[(xs > 0.1) & (xs < 0.2)] = -0.07  # 平らな本の背が棚の手前の縁から出ている
    pts = np.stack([xs, y, np.full_like(xs, 0.01)], -1)
    _, occ_default = occupancy_from_points(pts, k, z_band=(0.005, 0.05))
    left, occ = occupancy_from_points(pts, k, z_band=(0.005, 0.05), y_min=-0.10)
    spans = free_spans(left, occ, 0.002, k.usable_x)
    assert len(spans) == 2 and abs(spans[0][1] - 0.1) < 0.003
    assert occ_default[(left > 0.11) & (left < 0.19)].all()  # 既定でも「見えない＝埋まっている」なので安全側


def test_inventory_records_flat_stack(tmp_path):
    inv = Inventory(tmp_path / "inv.sqlite")
    inv.add(Placement(run_id="r1", env=0, shelf_id="shelf-01", book_id="book-10", x_lo_m=0.02, x_hi_m=0.28, fingers=[],
                      free_space_source="inventory", placed_sim_s=3.0, upright=False, inside=True, pose="flat",
                      stack_level=1, top_z_m=0.066, spine_front=True))
    r = inv.on_shelf("shelf-01")[0]
    assert r["pose"] == "flat" and r["stack_level"] == 1 and r["spine_front"] == 1 and r["upright"] == 0
    inv.close()
