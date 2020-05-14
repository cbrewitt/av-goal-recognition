import pytest
import numpy as np
from lanelet2 import geometry
from lanelet2.core import LaneletMap, Lanelet, getId, LineString3d, Point3d, Point2d,BasicPoint2d
from av_goal_recognition.lanelet_helpers import LaneletHelpers


def tuple_list_to_ls(tuple_list):
    return LineString3d(getId(), [Point3d(getId(), x, y) for x, y in tuple_list])


def get_test_map():
    map = LaneletMap()
    lanelet = get_test_lanelet_straight()
    map.add(lanelet)
    return map


def get_test_lanelet_straight():
    left_bound = tuple_list_to_ls([(0, 1), (1, 1)])
    right_bound = tuple_list_to_ls([(0, 0), (1, 0)])
    lanelet = Lanelet(getId(), left_bound, right_bound)
    return lanelet


def get_following_lanelets():
    left_points = [Point3d(getId(), x, y) for x, y in [(0, 1), (1, 1), (2, 2)]]
    right_points = [Point3d(getId(), x, y) for x, y in [(0, 0), (1, 0), (2, 1)]]
    lanelet_1_left_bound = LineString3d(getId(), left_points[:2])
    lanelet_1_right_bound = LineString3d(getId(), right_points[:2])
    lanelet_1 = Lanelet(getId(), lanelet_1_left_bound, lanelet_1_right_bound)
    lanelet_2_left_bound = LineString3d(getId(), left_points[1:])
    lanelet_2_right_bound = LineString3d(getId(), right_points[1:])
    lanelet_2 = Lanelet(getId(), lanelet_2_left_bound, lanelet_2_right_bound)
    return lanelet_1, lanelet_2


def get_adjacent_lanelets():
    leftmost_bound = tuple_list_to_ls([(0, 2), (1, 2), (2, 2)])
    mid_bound = tuple_list_to_ls([(0, 1), (1, 1), (2, 2)])
    rightmost_bound = tuple_list_to_ls([(0, 0), (1, 0), (2, 1)])
    left_lanelet = Lanelet(getId(), leftmost_bound, mid_bound)
    right_lanelet = Lanelet(getId(), mid_bound, rightmost_bound)
    return left_lanelet, right_lanelet


def get_test_lanelet_curved():
    left_bound = tuple_list_to_ls([(0, 1), (1, 1), (2, 2)])
    right_bound = tuple_list_to_ls([(0, 0), (1, 0), (2, 1)])
    lanelet = Lanelet(getId(), left_bound, right_bound)
    return lanelet


def test_dist_along():
    lanelet = get_test_lanelet_straight()
    point = BasicPoint2d(0.5, 0.75)
    assert LaneletHelpers.dist_along(lanelet, point) == pytest.approx(0.5)


def test_heading_at_center_straight():
    lanelet = get_test_lanelet_straight()
    point = BasicPoint2d(0.5, 0.75)
    assert LaneletHelpers.heading_at(lanelet, point) == pytest.approx(0)


def test_heading_at_center_start():
    lanelet = get_test_lanelet_straight()
    point = BasicPoint2d(0., 0.75)
    assert LaneletHelpers.heading_at(lanelet, point) == pytest.approx(0)


def test_heading_at_center_end():
    lanelet = get_test_lanelet_straight()
    point = BasicPoint2d(1., 0.75)
    assert LaneletHelpers.heading_at(lanelet, point) == pytest.approx(0)


def test_heading_at_center_curved():
    lanelet = get_test_lanelet_curved()
    point = BasicPoint2d(1.5, 0.75)
    assert LaneletHelpers.heading_at(lanelet, point) == pytest.approx(np.pi/4)


def test_following_lanelets_true():
    lanelet_1, lanelet_2 = get_following_lanelets()
    assert LaneletHelpers.follows(lanelet_2, lanelet_1)


def test_following_lanelets_false():
    lanelet_1, lanelet_2 = get_following_lanelets()
    assert not LaneletHelpers.follows(lanelet_1, lanelet_2)


def test_left_of_true():
    left_lanelet, right_lanelet = get_adjacent_lanelets()
    assert LaneletHelpers.left_of(left_lanelet, right_lanelet)


def test_left_of_false():
    left_lanelet, right_lanelet = get_adjacent_lanelets()
    assert not LaneletHelpers.left_of(right_lanelet, left_lanelet)
