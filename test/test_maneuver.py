import numpy as np
import pytest

from core.scenario import Frame, AgentState
from test_feature_extraction import get_feature_extractor

from igp2.maneuver import FollowLane, ManeuverConfig, SwitchLane, GiveWay, Maneuver


def test_follow_lane_path():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 1,
                             'final_lanelet_id': 3})
    maneuver = FollowLane(0, frame, feature_extractor, config)
    path = maneuver.get_route(feature_extractor).shortestPath()
    assert [l.id for l in path] == [1, 3]


def test_follow_lane_points_turn():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.4, 2.9),
                             'initial_lanelet_id': 2,
                             'final_lanelet_id': 5})
    maneuver = FollowLane(0, frame, feature_extractor, config)
    lanelet_path = [feature_extractor.lanelet_map.laneletLayer.get(i) for i in [2, 5]]
    points = maneuver.get_points(0, frame, lanelet_path)
    expected_points = [(0.1, 1.5), (2, 1.5), (2.5, 2), (3.4, 2.9)]
    assert np.allclose(points, expected_points)


def test_follow_lane_points_straight():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 1,
                             'final_lanelet_id': 3})
    maneuver = FollowLane(0, frame, feature_extractor, config)
    lanelet_path = [feature_extractor.lanelet_map.laneletLayer.get(i) for i in [1, 3]]
    points = maneuver.get_points(0, frame, lanelet_path)
    expected_points = [(0.1, 0.5), (2, 0.5), (3.9, 0.5)]
    assert np.allclose(points, expected_points)


def test_follow_lane_get_path():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.4, 2.9),
                             'initial_lanelet_id': 2,
                             'final_lanelet_id': 5})
    maneuver = FollowLane(0, frame, feature_extractor, config)
    lanelet_path = [feature_extractor.lanelet_map.laneletLayer.get(i) for i in [2, 5]]
    points = maneuver.get_points(0, frame, lanelet_path)
    path = maneuver.get_path(0, frame, points)
    assert np.allclose(path[0], (0.1, 1.5))
    assert np.allclose(path[-1], (3.4, 2.9))


def test_get_velocity_straight():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 1,
                             'final_lanelet_id': 3})
    maneuver = FollowLane(0, frame, feature_extractor, config)
    route = maneuver.get_route(feature_extractor)
    path = np.array([[0.1, 0.5], [2.1, 0.5], [3.9, 0.5]])
    lanelet_path = route.shortestPath()
    velocity = maneuver.get_velocity(path, 0, frame, feature_extractor, lanelet_path)
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
    maneuver = FollowLane(0, frame, feature_extractor, config)
    route = maneuver.get_route(feature_extractor)
    path = np.array([[0.1, 0.5], [2.1, 0.5], [3.9, 0.5]])
    lanelet_path = route.shortestPath()
    velocity = maneuver.get_velocity(path, 0, frame, feature_extractor, lanelet_path)
    assert np.all(velocity == np.array([5, 5, 5]))


def test_follow_lane_terminal_state():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 1,
                             'final_lanelet_id': 3})
    maneuver = FollowLane(0, frame, feature_extractor, config)
    terminal_state = maneuver.terminal_state()
    assert terminal_state.x == pytest.approx(3.9)
    assert terminal_state.y == pytest.approx(0.5)
    assert terminal_state.heading == pytest.approx(0)
    assert terminal_state.v_x == pytest.approx(10)
    assert terminal_state.v_y == pytest.approx(0)


def test_switch_lane_path():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 2,
                             'final_lanelet_id': 3})
    maneuver = SwitchLane(0, frame, feature_extractor, config)
    assert np.allclose(maneuver.path[-1], [3.9, 0.5])
    assert np.allclose(maneuver.path[0], [0.1, 1.5])
    assert maneuver.path.shape[0] == 4
    initial_heading = np.arctan2(*(maneuver.path[1] - maneuver.path[0])[::-1])
    final_heading = np.arctan2(*(maneuver.path[-1] - maneuver.path[-2])[::-1])
    assert initial_heading == pytest.approx(0, abs=np.pi/6)
    assert final_heading == pytest.approx(0, abs=np.pi/6)


def test_switch_lane_velocity():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    state2 = AgentState(0, 2.1, 0.5, 0, 0, 0, 0, 0, 5, 0, 0, 0)
    frame.add_agent_state(1, state2)
    frame.add_agent_state(0, state)
    config = ManeuverConfig({'termination_point': (3.9, 0.5),
                             'initial_lanelet_id': 2,
                             'final_lanelet_id': 3})
    maneuver = SwitchLane(0, frame, feature_extractor, config)
    assert np.all(maneuver.velocity <= 5)


def test_cautious_cost_deceleration():
    path = np.array([[0, 0], [0, 1], [0, 2]])
    velocity = GiveWay.get_const_deceleration_vel(10, 2, path)
    assert np.all(velocity == [10, 6, 2])


def test_trajectory_times():
    path = np.array([[0, 0], [0, 1], [0, 2]])
    velocity = np.array([10, 6, 2])
    time = Maneuver.trajectory_times(path, velocity)
    assert np.allclose(time, [0, 1/8, 3/8])


def test_time_until_clear():
    ego_time_to_junction = 5
    times_to_junction = [3, 7, 11, 12]
    assert GiveWay.get_time_until_clear(ego_time_to_junction, times_to_junction) == 7


def test_add_stopping_points():
    path = np.array([[0, 0], [0, 1], [0, 2]])
    new_path = GiveWay.add_stop_points(path)
    assert np.allclose(new_path, np.array([[0, 0], [0, 1], [0, 1.7], [0, 1.9], [0, 2]]))


def test_add_stop_velocity():
    path = np.array([[0, 0], [0, 1], [0, 1.7], [0, 1.9], [0, 2]])
    velocity = [1, 1, 1]
    stop_velocity = GiveWay.add_stop_velocity(path, velocity, 0.466)
    assert stop_velocity == pytest.approx(0.5)
