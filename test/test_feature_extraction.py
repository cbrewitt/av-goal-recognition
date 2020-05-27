import numpy as np
import pytest
from lanelet2.core import LaneletMap, Lanelet, getId, LineString3d, Point3d, Point2d, BasicPoint2d
from av_goal_recognition.goal_recognition import FeatureExtractor, AgentState
from test.lanelet_test_helpers import get_test_lanelet_straight, get_test_lanelet_curved, get_test_map


def get_feature_extractor():
    lanelet_map = get_test_map()
    return FeatureExtractor(lanelet_map)


def test_angle_in_lane_straight():
    state = AgentState(0, 0.5, 0.75, 0, 0, np.pi/4, 0, 0, 0, 0, 0, 0)
    lanelet = get_test_lanelet_straight()
    assert FeatureExtractor.angle_in_lane(state, lanelet) == pytest.approx(np.pi/4)


def test_angle_in_lane_curved():
    state = AgentState(0, 1.5, 1.0, 0, 0, np.pi/2, 0, 0, 0, 0, 0, 0)
    lanelet = get_test_lanelet_curved()
    assert FeatureExtractor.angle_in_lane(state, lanelet) == pytest.approx(np.pi/4)


def test_reachable_goals():
    feature_extractor = get_feature_extractor()
    goals = [(3.5, 0.5), (3.0, 2.5)]
    start_lanelet = feature_extractor.lanelet_map.laneletLayer.get(1)
    reachable_goals = feature_extractor.reachable_goals(start_lanelet, goals)
    assert list(reachable_goals.keys()) == [0, 1]
