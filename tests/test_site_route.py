"""B の台車の道が、廃棄箱・保留棚・査定台に近づきすぎないか（site.py。02_確認事項 B15）。"""

import math

import numpy as np
import pytest

from pai_book_intake.site import (ROUTE_MARGIN, b_route, cart_corners, load_site, obstacle_rects, polygon_distance,
                                  rect_corners, route_clearance)


def test_polygon_distance():
    a = rect_corners((0.0, 0.0), (1.0, 1.0))
    assert polygon_distance(a, rect_corners((2.0, 0.0), (1.0, 1.0))) == pytest.approx(1.0)
    assert polygon_distance(a, rect_corners((0.5, 0.5), (1.0, 1.0))) == 0.0  # 重なる
    assert polygon_distance(a, rect_corners((2.0, 2.0), (1.0, 1.0))) == pytest.approx(math.sqrt(2))  # 角どうし
    assert polygon_distance(a, rect_corners((1.6, 0.0), (1.0, 1.0), yaw=math.pi / 4)) == pytest.approx(
        1.6 - 0.5 - math.sqrt(2) / 2)


def test_cart_footprint_matches_the_chassis():
    """台車の床板は、台車の中心から前へ 0.25 m・後ろへ 0.35 m・左右 0.24 m（pinch.CHASSIS_XML）"""
    c = cart_corners((0.0, 0.0, 0.0))
    assert c[:, 0].min() == pytest.approx(-0.35) and c[:, 0].max() == pytest.approx(0.25)
    assert np.abs(c[:, 1]).max() == pytest.approx(0.24)
    c = cart_corners((0.0, 0.0, -math.pi / 2))  # 待機位置の向き（-y を向く）: 前は -y
    assert c[:, 1].min() == pytest.approx(-0.25) and c[:, 1].max() == pytest.approx(0.35)


def test_site_has_hold_table_reachable_like_the_bin():
    site = load_site()
    st = np.array(site["robot_b"]["station"][:2])
    d_bin = np.linalg.norm(np.array(site["bin"]["center"]) - st)
    d_hold = np.linalg.norm(np.array(site["hold_table"]["center"]) - st)
    assert d_hold == pytest.approx(d_bin, abs=0.02)  # 廃棄箱と同じく、台車を動かさずに腕で届く距離


@pytest.mark.parametrize("leg", ["to_shelf", "to_station"])
def test_b_routes_keep_the_margin_from_every_obstacle(leg):
    """行きも帰りも、台車の床板は廃棄箱・保留棚・査定台から ROUTE_MARGIN 以上離れる（待機位置に止まった所も含む）"""
    site = load_site()
    rbs = site["robot_b"]
    st, front = tuple(rbs["station"]), site.shelf().front_pose()
    route = b_route(rbs, st, front) if leg == "to_shelf" else b_route(rbs, front, st)
    obs = obstacle_rects(site)
    assert set(obs) == {"appraisal_table", "bin", "hold_table"}
    for name, (d, pose) in route_clearance(route, obs).items():
        assert d >= ROUTE_MARGIN - 1e-6, f"{leg}: {name} まで {d:.3f} m（{pose}）"


def test_route_turns_only_at_the_approach_point():
    site = load_site()
    rbs = site["robot_b"]
    route = b_route(rbs, tuple(rbs["station"]), site.shelf().front_pose())
    assert len(route) == 4
    assert route[1][:2] == route[2][:2] == tuple(rbs["station_approach"])
    assert route[0][2] == route[1][2] and route[2][2] == route[3][2]  # 向きを変えるのは手前の点だけ


def test_the_old_straight_route_hits_the_bin():
    """直す前の道（待機位置から本棚の正面へまっすぐ。向きを変えながら）は、台車の床板が廃棄箱に重なる。
    判定が当たりを見つけること"""
    site = load_site()
    route = [tuple(site["robot_b"]["station"]), site.shelf().front_pose()]
    assert route_clearance(route, obstacle_rects(site))["bin"][0] == 0.0
