import numpy as np

from core.scenario import Frame, AgentState
from igp2.macro_action import ContinueLane
from igp2.maneuver import FollowLane
from test_feature_extraction import get_feature_extractor


def test_continue_lane_applicable():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    goal = (3.9, 0.5)
    assert ContinueLane.applicable(0, frame, feature_extractor, goal)


def test_continue_lane_not_applicable():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    goal = (3.4, 2.9)
    assert not ContinueLane.applicable(0, frame, feature_extractor, goal)


def test_continue_lane():
    feature_extractor = get_feature_extractor()
    frame = Frame(0)
    state = AgentState(0, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    goal = (3.9, 0.5)
    macro_action = ContinueLane(0, frame, feature_extractor, goal)
    assert len(macro_action.maneuvers) == 1
    assert type(macro_action.maneuvers[0]) == FollowLane
    assert np.allclose(macro_action.maneuvers[0].path[0], [0.1, 0.5])
    assert np.allclose(macro_action.maneuvers[0].path[-1], goal)
