from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from lanelet2.core import BasicPoint2d
from shapely.geometry import LineString, Point
from shapely.ops import split

from core.feature_extraction import FeatureExtractor
from core.lanelet_helpers import LaneletHelpers
from core.scenario import Frame, AgentState
from igp2.util import get_curvature


class ManeuverConfig:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    @property
    def type(self):
        return self.config_dict.get('type')

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

    def __init__(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor, config: ManeuverConfig):
        self.config = config
        self.path, self.velocity = self.get_trajectory(agent_id, frame, feature_extractor)

    @abstractmethod
    def get_trajectory(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        raise NotImplementedError

    @classmethod
    def get_curvature_velocity(cls, path):
        c = np.abs(get_curvature(path))
        v = np.maximum(cls.MIN_SPEED, cls.MAX_SPEED * (1 - 3 * np.abs(c)))
        return v

    def terminal_state(self):
        frame_id = None
        x, y = self.path[-1]
        direction = self.path[-1] - self.path[-2]
        direction = direction / np.linalg.norm(direction)
        heading = np.arctan2(direction[1], direction[0])
        v_lon = self.velocity[-1]
        v_x, v_y = direction * v_lon
        v_lat = 0
        a_x = 0
        a_y = 0
        a_lon = 0
        a_lat = 0
        return AgentState(frame_id, x, y, v_x, v_y, heading, a_x, a_y,
                          v_lon, v_lat, a_lon, a_lat)


class FollowLane(Maneuver):

    def get_points(self, agent_id: int, frame: Frame, lanelet_path):

        final_point = lanelet_path[-1].centerline[-1]
        lane_points = [(p.x, p.y) for l in lanelet_path for p in list(l.centerline)[:-1]] \
            + [(final_point.x, final_point.y)]
        lane_ls = LineString(lane_points)

        current_point = frame.agents[agent_id].shapely_point
        termination_lon = lane_ls.project(Point(self.config.termination_point))
        termination_point = lane_ls.interpolate(termination_lon).coords[0]
        lat_dist = lane_ls.distance(current_point)
        current_lon = lane_ls.project(current_point)

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

    def get_route(self, feature_extractor: FeatureExtractor):
        initial_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.config.initial_lanelet_id)
        final_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.config.final_lanelet_id)
        route = feature_extractor.routing_graph.getRoute(initial_lanelet, final_lanelet, withLaneChanges=False)
        assert route is not None, 'no route found from lanelet {} to {}'.format(
            initial_lanelet.id, final_lanelet.id)
        return route

    def get_path(self, agent_id: int, frame: Frame, points):
        points = np.array(points)
        heading = frame.agents[agent_id].heading
        initial_direction = np.array([np.cos(heading), np.sin(heading)])
        final_direction = np.diff(points[-2:], axis=0).flatten()
        final_direction = final_direction / np.linalg.norm(final_direction)
        t = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        cs = CubicSpline(t, points, bc_type=((1, initial_direction), (1, final_direction)))
        num_points = int(t[-1] / self.POINT_SPACING)
        ts = np.linspace(0, t[-1], num_points)
        path = cs(ts)
        return path

    def get_velocity(self, path, agent_id, frame, feature_extractor, route):
        velocity = self.get_curvature_velocity(path)
        vehicle_in_front_id, vehicle_in_front_dist = feature_extractor.vehicle_in_front(
            frame.agents[agent_id], route, frame)
        if vehicle_in_front_id is not None and vehicle_in_front_dist < 15:
            max_vel = frame.agents[vehicle_in_front_id].v_lon  # TODO what if this is zero?
            velocity = np.minimum(velocity, max_vel)
        return velocity

    def get_trajectory(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        route = self.get_route(feature_extractor)
        lanelet_path = route.shortestPath()
        points = self.get_points(agent_id, frame, lanelet_path)
        path = self.get_path(agent_id, frame, points)
        velocity = self.get_velocity(path, agent_id, frame, feature_extractor, route)
        return path, velocity


class Turn(FollowLane):
    pass


class SwitchLane(Maneuver):
    TARGET_SITCH_LENGTH = 20
    MIN_SWITCH_LENGTH = 5

    def get_path(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        target_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.config.final_lanelet_id)

        initial_state = frame.agents[agent_id]
        initial_point = np.array(initial_state.tuple_point)
        target_point = self.config.termination_point
        dist = np.linalg.norm(target_point - initial_point)
        initial_direction = np.array([np.cos(initial_state.heading),
                                      np.sin(initial_state.heading)])
        target_direction = LaneletHelpers.direction_at(target_lanelet, BasicPoint2d(*target_point))

        """
        Fit 2d cubic curve given boundary conditions at t=0 and t=1
        boundary == A @ coeff
        A = array([[0, 0, 0, 1],
                   [1, 1, 1, 1],
                   [0, 0, 1, 0],
                   [3, 2, 1, 0]])
        transform = np.linalg.inv(A)
        coeff = transform @ boundary
        """

        transform = np.array([[ 2., -2.,  1.,  1.],
                              [-3.,  3., -2., -1.],
                              [ 0.,  0.,  1.,  0.],
                              [ 1.,  0.,  0.,  0.]])
        boundary = np.vstack([initial_point,
                             target_point,
                             initial_direction * dist,
                             target_direction * dist])
        coeff = transform @ boundary

        # evaluate points on cubic curve
        num_points = max(2, int(dist / self.POINT_SPACING) + 1)
        t = np.linspace(0, 1, num_points)
        powers = np.vstack([t ** 3, t ** 2, t ** 1, t ** 0])
        points = powers.T @ coeff
        return points

    def get_trajectory(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        path = self.get_path(agent_id, frame, feature_extractor)
        velocity = self.get_curvature_velocity(path)  # TODO - take into account vehicle in front
        return path, velocity







