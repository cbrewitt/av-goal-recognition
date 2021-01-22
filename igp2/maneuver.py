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

    @property
    def exit_lanelet_id(self):
        return self.config_dict.get('exit_lanelet_id', None)


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

    def get_velocity(self, path, agent_id, frame, feature_extractor, lanelet_path):
        velocity = self.get_curvature_velocity(path)
        vehicle_in_front_id, vehicle_in_front_dist = feature_extractor.vehicle_in_front(
            frame.agents[agent_id], lanelet_path, frame)
        if vehicle_in_front_id is not None and vehicle_in_front_dist < 15:
            max_vel = frame.agents[vehicle_in_front_id].v_lon  # TODO what if this is zero?
            assert max_vel > 1e-4
            velocity = np.minimum(velocity, max_vel)
        return velocity

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

    @staticmethod
    def trajectory_times(path, velocity):
        # assume constant acceleration between points on path
        v_avg = (velocity[:-1] + velocity[1:])/2
        s = np.linalg.norm(np.diff(path, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(s / v_avg)])
        return t


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

    def get_trajectory(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        route = self.get_route(feature_extractor)
        lanelet_path = route.shortestPath()
        points = self.get_points(agent_id, frame, lanelet_path)
        path = self.get_path(agent_id, frame, points)
        velocity = self.get_velocity(path, agent_id, frame, feature_extractor, lanelet_path)
        return path, velocity


class Turn(FollowLane):
    pass


class SwitchLane(FollowLane):
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
        Fit cubic curve given boundary conditions at t=0 and t=1
        boundary == A @ coeff
        coeff = inv(A) @ boundary
        A = array([[0, 0, 0, 1],
                   [1, 1, 1, 1],
                   [0, 0, 1, 0],
                   [3, 2, 1, 0]])
        transform = np.linalg.inv(A)
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
        powers = np.power(t.reshape((-1, 1)), np.arange(3, -1, -1))
        points = powers @ coeff
        return points

    def get_lanelet_path(self, feature_extractor: FeatureExtractor):
        target_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.config.final_lanelet_id)
        initial_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.config.initial_lanelet_id)

        current_lanelet = target_lanelet
        path = [current_lanelet]

        while not LaneletHelpers.adjacent(initial_lanelet, current_lanelet):
            previous_lanelets = feature_extractor.routing_graph.previous(current_lanelet)
            if len(previous_lanelets) == 1:
                current_lanelet = previous_lanelets[0]
                path.insert(0, current_lanelet)
            else:
                break
        return path

    def get_trajectory(self, agent_id: int, frame: Frame, feature_extractor: FeatureExtractor):
        path = self.get_path(agent_id, frame, feature_extractor)
        lanelet_path = self.get_lanelet_path(feature_extractor)
        velocity = self.get_velocity(path, agent_id, frame, feature_extractor, lanelet_path)
        return path, velocity


class GiveWay(FollowLane):
    MAX_ONCOMING_VEHICLE_DIST = 50
    GAP_TIME = 3

    def get_velocity(self, path, agent_id, frame, feature_extractor, lanelet_path):
        state = frame.agents[agent_id]

        velocity = self.get_const_deceleration_vel(state.v_lon, 2, path)
        ego_time_to_junction = self.trajectory_times(path, velocity)[0]

        times_to_juction = self.get_times_to_junction(frame, feature_extractor, ego_time_to_junction)
        time_until_clear = self.get_time_until_clear(ego_time_to_junction, times_to_juction)
        stop_time = time_until_clear - ego_time_to_junction
        if stop_time > 0:
            # insert waiting points
            path = self.add_stop_points(path)
            velocity = self.add_stop_velocity(path, velocity, stop_time)

            pass

    @staticmethod
    def add_stop_points(path):
        p_start = path[-2, None]
        p_end = path[-1, None]
        diff = p_end - p_start
        p_stop_frac = np.array([[0.7, 0.9]]).T
        p_stop = p_start + p_stop_frac @ diff
        new_path = np.concatenate([path[:-1], p_stop, p_end])
        return new_path

    @staticmethod
    def add_stop_velocity(path, velocity, stop_time):
        final_section = path[-4:]
        s = np.linalg.norm(np.diff(final_section, axis=0), axis=1)
        v1, v2 = velocity[-2:]
        t = stop_time + 2 * np.sum(s) / (v1 + v2)
        A = np.array([[t, 0, 0, 0],
                      [t * (v1 + v2), -2, 1, -2],
                      [-v1 * v2 * t, -2 * v2, -v1 - v2, -2 * v1],
                      [0, 0, -v1 * v2, 0]])

        coeff = A @ np.concatenate([[1], s]).T
        r = np.roots(coeff)
        stop_vel = np.max(r.real[np.abs(r.imag < 1e-5)])
        import pdb;pdb.set_trace()
        return stop_vel

    def get_times_to_junction(self, frame, feature_extractor, ego_time_to_junction):
        exit_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.config.exit_lanelet_id)
        entry_lanelet = feature_extractor.lanelet_map.laneletLayer.get(self.config.final_lanelet_id)
        route = feature_extractor.routing_graph.shortestPath(entry_lanelet, exit_lanelet, withLaneChanges=False)
        oncoming_vehicles = feature_extractor.oncoming_vehicles(route, frame, max_dist=self.MAX_ONCOMING_VEHICLE_DIST)

        time_to_junction = []
        for agent, dist in oncoming_vehicles.values():
            # check is the vehicle stopped
            time = dist/agent.v_lon
            if agent.v_lon > 1 and time > ego_time_to_junction:
                time_to_junction.append(time)

        return time_to_junction

    @classmethod
    def get_time_until_clear(cls, ego_time_to_junction, times_to_junction):
        if len(times_to_junction) == 0:
            return 0.
        times_to_junction = np.array(times_to_junction)
        times_to_junction = times_to_junction[times_to_junction >= ego_time_to_junction]
        times_to_junction = np.concatenate([[ego_time_to_junction], times_to_junction, [np.inf]])
        gaps = np.diff(times_to_junction)
        first_long_gap = np.argmax(gaps >= cls.GAP_TIME)
        return times_to_junction[first_long_gap]

    @staticmethod
    def get_const_deceleration_vel(initial_vel, final_vel, path):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        velocity = initial_vel + s / s[-1] * (final_vel - initial_vel)
        return velocity

