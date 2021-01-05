import numpy as np

from core.scenario import Frame, AgentState
from test_feature_extraction import get_feature_extractor

from igp2.maneuver import FollowLane, ManeuverConfig


def test_follow_lane_path():
    feature_extractor = get_feature_extractor()
    frames = [Frame(0)]
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frames[0].add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 1,
                             'final_lanelet_id': 3})
    maneuver = FollowLane(feature_extractor, config)
    path = maneuver.get_route(feature_extractor).shortestPath()
    assert [l.id for l in path] == [1, 3]


def test_follow_lane_points_turn():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.5, 2.9),
                             'initial_lanelet_id': 2,
                             'final_lanelet_id': 5})
    maneuver = FollowLane(feature_extractor, config)
    lanelet_path = [feature_extractor.lanelet_map.laneletLayer.get(i) for i in [2, 5]]
    points = maneuver.get_points(0, frame, lanelet_path)
    expected_points = [(0.1, 1.5), (2, 1.5), (2.5, 2), (3.5, 2.9)]
    assert points == expected_points


def test_follow_lane_points_straight():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 1,
                             'final_lanelet_id': 3})
    maneuver = FollowLane(feature_extractor, config)
    lanelet_path = [feature_extractor.lanelet_map.laneletLayer.get(i) for i in [1, 3]]
    points = maneuver.get_points(0, frame, lanelet_path)
    expected_points = [(0.1, 0.5), (2, 0.5), (3.9, 0.5)]
    assert points == expected_points


def test_follow_lane_get_path():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.5, 2.9),
                             'initial_lanelet_id': 2,
                             'final_lanelet_id': 5})
    maneuver = FollowLane(feature_extractor, config)
    lanelet_path = [feature_extractor.lanelet_map.laneletLayer.get(i) for i in [2, 5]]
    points = maneuver.get_points(0, frame, lanelet_path)
    path = maneuver.get_path(0, frame, points)
    assert np.allclose(path[0], (0.1, 1.5))
    assert np.allclose(path[-1], (3.5, 2.9))


def test_get_velocity_straight():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 1,
                             'final_lanelet_id': 3})
    maneuver = FollowLane(feature_extractor, config)
    route = maneuver.get_route(feature_extractor)
    path = np.array([[0.1, 0.5], [2.1, 0.5], [3.9, 0.5]])
    velocity = maneuver.get_velocity(path, 0, frame, feature_extractor, route)
    assert np.all(velocity == np.array([10, 10, 10]))


def test_get_velocity_vehicle_ahead():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    state2 = AgentState(0, 2.1, 0.5, 0, 0, 0, 0, 0, 5, 0, 0, 0)
    frame.add_agent_state(1, state2)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 1,
                             'final_lanelet_id': 3})
    maneuver = FollowLane(feature_extractor, config)
    route = maneuver.get_route(feature_extractor)
    path = np.array([[0.1, 0.5], [2.1, 0.5], [3.9, 0.5]])
    velocity = maneuver.get_velocity(path, 0, frame, feature_extractor, route)
    assert np.all(velocity == np.array([5, 5, 5]))
