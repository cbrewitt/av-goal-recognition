from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline

from shapely.geometry import LineString, Point
from shapely.ops import split

from core.feature_extraction import FeatureExtractor
from core.scenario import Frame
from igp2.util import get_curvature


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

    POINT_SPACING = 1
    MAX_SPEED = 10
    MIN_SPEED = 3

    def __init__(self, feature_extractor: FeatureExtractor, man_config: ManeuverConfig):
        self.man_config = man_config

    @classmethod
    @abstractmethod
    def applicable(cls, agent_id, frames, feature_extractor: FeatureExtractor):
        raise NotImplementedError

    @abstractmethod
    def get_points(self, agent_id, frames, feature_extractor: FeatureExtractor):
        raise NotImplementedError

    @classmethod
    def get_curvature_velocity(cls, path):
        c = np.abs(get_curvature(path))
        v = np.maximum(cls.MIN_SPEED, cls.MAX_SPEED * (1 - 3 * np.abs(c)))
        return v


class FollowLane(Maneuver):

    def __init__(self, feature_extractor: FeatureExtractor, man_config):
        super().__init__(feature_extractor, man_config)

    @classmethod
    def applicable(cls, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        return feature_extractor.get_current_lanelet(frame) is not None

    def get_points(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):

        lanelet_path = self.get_lanelet_path(feature_extractor)

        final_point = lanelet_path[-1].centerline[-1]
        lane_points = [(p.x, p.y) for l in lanelet_path for p in list(l.centerline)[:-1]] \
            + [(final_point.x, final_point.y)]
        lane_ls = LineString(lane_points)

        current_point = frame.agents[agent_id].shapely_point
        termination_point = self.man_config.termination_point
        lat_dist = lane_ls.distance(current_point)
        current_lon = lane_ls.project(current_point)
        termination_lon = lane_ls.project(Point(termination_point))
        margin = self.POINT_SPACING + 2 * lat_dist

        assert current_lon < lane_ls.length - margin, 'current point is past end of lane'
        assert current_lon < termination_lon, 'current point is past the termination point'

        # trim out points we have passed
        first_ls_point = None
        final_ls_point = None
        for coord in lane_ls.coords:
            point = Point(coord)
            point_lon = lane_ls.project(point)
            if point_lon > current_lon + margin and first_ls_point is None:
                first_ls_point = point
            if first_ls_point is not None:
                if point_lon + self.POINT_SPACING > termination_lon:
                    break
                else:
                    final_ls_point = point

        if first_ls_point is None or final_ls_point is None:
            raise Exception('Could not find first/final point')

        if final_ls_point == first_ls_point:
            trimmed_points = first_ls_point
        else:
            # trim out points before first point
            if first_ls_point == Point(lane_ls.coords[-1]):
                # handle case where first point is final point
                following_points = first_ls_point
            else:
                following_points = split(lane_ls, first_ls_point)[-1]
            # trim out points after final point
            trimmed_points = split(following_points, final_ls_point)[0]

        all_points = list(current_point.coords) + list(trimmed_points.coords) + [termination_point]
        return all_points

    def get_lanelet_path(self, feature_extractor: FeatureExtractor):
        initial_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.man_config.initial_lanelet_id)
        final_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.man_config.final_lanelet_id)
        route = feature_extractor.routing_graph.getRoute(initial_lanelet, final_lanelet)
        assert route is not None, 'no route found from lanelet {} to {}'.format(
            initial_lanelet.id, final_lanelet.id)
        path = route.shortestPath() # TODO do not allow lane changes
        return path

    def get_path(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        # generate and array of evenly spaced points
        points = self.get_points(agent_id, frame, feature_extractor)
        points = np.array(points)
        heading = frame.agents[agent_id].heading
        initial_direction = np.array([np.cos(heading), np.sin(heading)])

        final_direction = np.diff(points[-2:], axis=0).flatten()
        final_direction = final_direction / np.linalg.norm(final_direction)

        t = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))

        cs_x = CubicSpline(t, points[:, 0], bc_type=((1, initial_direction[0]), (1, final_direction[0])))
        cs_y = CubicSpline(t, points[:, 1], bc_type=((1, initial_direction[1]), (1, final_direction[1])))
        num_points = int(t[-1] / self.POINT_SPACING)
        ts = np.linspace(0, t[-1], num_points)
        path = np.vstack((cs_x(ts), cs_y(ts))).T
        return path

    def get_trajectory(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        path = self.get_path(agent_id, frame, feature_extractor)
        velocity = self.get_curvature_velocity(path)
        return path, velocity



