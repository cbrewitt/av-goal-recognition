from abc import ABC, abstractmethod

from core.feature_extraction import FeatureExtractor


class ManeuverConfig:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    @property
    def termination_point(self):
        return self.config_dict.get('termination_point')

    @property
    def initial_lanelet(self):
        return self.config_dict.get('initial_lanelet')

    @property
    def final_lanelet(self):
        return self.config_dict.get('final_lanelet')


class Maneuver(ABC):

    def __init__(self, feature_extractor: FeatureExtractor, man_config: ManeuverConfig):
        self.man_config = man_config

    @classmethod
    @abstractmethod
    def applicable(cls, agent_id, frames, feature_extractor: FeatureExtractor):
        raise NotImplementedError

    @abstractmethod
    def trajectory(self, frames, feature_extractor: FeatureExtractor):
        raise NotImplementedError


class FollowLane(Maneuver):

    def __init__(self, feature_extractor: FeatureExtractor, man_config):
        super().__init__(feature_extractor, man_config)

    @classmethod
    def applicable(cls, agent_id, frames, feature_extractor: FeatureExtractor):
        return feature_extractor.get_current_lanelet(frames[-1]) is not None

    def trajectory(self, frames, feature_extractor: FeatureExtractor):
        route = feature_extractor.routing_graph.getRoute(self.man_config.initial_lanelet,
                                                         self.man_config.final_lanelet)
        path = route.shortestPath()
        points = []
        pass
        #TODO include current lanelet with trajectory?
        #TODO simulate forward other vehilcles base on cvel lane follow?
