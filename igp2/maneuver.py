from abc import ABC, abstractmethod

from core.feature_extraction import FeatureExtractor


class ManeuverConfig:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    @property
    def termination_point(self):
        return self.config_dict.get('termination_point', None)

    @property
    def initial_lanelet_id(self):
        return self.config_dict.get('initial_lanelet_id', None)

    @property
    def final_lanelet_id(self):
        return self.config_dict.get('final_lanelet_id', None)


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
        initial_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.man_config.initial_lanelet_id)
        final_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.man_config.final_lanelet_id)
        route = feature_extractor.routing_graph.getRoute(initial_lanelet, final_lanelet)
        path = route.shortestPath() # TODO do not allow lane changes
        final_point = path[-1].centerline[-1]
        points = [(p.x, p.y) for l in path for p in list(l.centerline)[:-1]] \
                 + [(final_point.x, final_point.y)]
        x, y = list(zip(*points))

        # TODO represent as linestring and project start/end points
        #TODO scipy.interpolate.CubicSpline can take deriv values at end

        pass
        #TODO include current lanelet with trajectory?
        #TODO simulate forward other vehilcles base on cvel lane follow?
