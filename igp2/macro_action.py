from abc import ABC, abstractmethod

from core.feature_extraction import FeatureExtractor
from core.scenario import Frame
from igp2.maneuver import ManeuverConfig, FollowLane


class MacroAction(ABC):

    def __init__(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor, goal):
        self.maneuvers = self.get_maneuvers(agent_id, frame, feature_extractor, goal)

    @abstractmethod
    def applicable(self, frame: Frame, feature_extractor: FeatureExtractor, goal):
        raise NotImplementedError

    @abstractmethod
    def get_maneuvers(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor, goal):
        raise NotImplementedError


class ContinueLane(MacroAction):
    """ Continue in the current lane until the goal is reached/lane end """

    def get_maneuvers(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor, goal):
        state = frame.agents[agent_id]
        route = feature_extractor.get_goal_routes(state, [goal])[0]
        lanelet_path = route.shortestPath()
        initial_lanelet = lanelet_path[0]
        final_lanelet = lanelet_path[-1]
        man_config = ManeuverConfig({'termination_point': goal,
                                     'initial_lanelet_id': initial_lanelet.id,
                                     'final_lanelet_id': final_lanelet.id})
        maneuvers = [FollowLane(agent_id, frame, feature_extractor, man_config)]
        return maneuvers

    def applicable(self, frame: Frame, feature_extractor: FeatureExtractor, goal):
        raise NotImplementedError