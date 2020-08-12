import numpy as np
import lanelet2
from lanelet2.core import BasicPoint2d, BoundingBox2d
from lanelet2 import geometry

from av_goal_recognition.lanelet_helpers import LaneletHelpers
from av_goal_recognition.scenario import Frame


class FeatureExtractor:

    feature_names = ['path_to_goal_length',
                     'in_correct_lane',
                     'speed',
                     'acceleration',
                     'angle_in_lane',
                     'angle_to_goal',
                     'vehicle_in_front_dist',
                     'vehicle_in_front_speed',
                     'oncoming_vehicle_dist']

    def __init__(self, lanelet_map):
        self.lanelet_map = lanelet_map
        self.traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
        self.routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, self.traffic_rules)

    def extract(self, agent_id, frames, goal, route=None):
        """Extracts a vector of features

        Args:
            tracks:
            track_id:
            frame:
            lanelet_map:
            goal:

        Returns:
            list: [in_correct_lane, distance_to_goal, speed, acceleration]

        """

        current_frame = frames[-1]
        current_state = current_frame.agents[agent_id]
        state_history = [f.agents[agent_id] for f in frames]
        lanelet_sequence = self.get_lanelet_sequence(state_history)
        current_lanelet = lanelet_sequence[-1]

        if route is None:
            route = self.route_to_goal(current_lanelet, goal)
        if route is None:
            raise ValueError('Unreachable goal')

        speed = current_state.v_lon
        acceleration = current_state.a_lon
        in_correct_lane = self.in_correct_lane(route)
        path_to_goal_length = self.path_to_goal_length(current_state, goal, route)
        angle_in_lane = self.angle_in_lane(current_state, current_lanelet)
        angle_to_goal = self.angle_to_goal(current_state, goal)
        goal_type = self.goal_type(current_state, goal, route)

        vehicle_in_front_id, vehicle_in_front_dist = self.vehicle_in_front(current_state, route, current_frame)
        if vehicle_in_front_id is None:
            vehicle_in_front_speed = 20
            vehicle_in_front_dist = 100
        else:
            vehicle_in_front = current_frame.agents[vehicle_in_front_id]
            vehicle_in_front_speed = vehicle_in_front.v_lon

        oncoming_vehicle_dist = self.oncoming_vehicle_dist(route, current_frame)

        return {'path_to_goal_length': path_to_goal_length,
                'in_correct_lane': in_correct_lane,
                'speed': speed,
                'acceleration': acceleration,
                'angle_in_lane': angle_in_lane,
                'angle_to_goal': angle_to_goal,
                'vehicle_in_front_dist': vehicle_in_front_dist,
                'vehicle_in_front_speed': vehicle_in_front_speed,
                'oncoming_vehicle_dist': oncoming_vehicle_dist,
                'goal_type': goal_type}

    @staticmethod
    def angle_in_lane(state, lanelet):
        lane_heading = LaneletHelpers.heading_at(lanelet, state.point)
        angle_diff = np.diff(np.unwrap([lane_heading, state.heading]))[0]
        return angle_diff

    def reachable_goals(self, start_lanelet, goals):
        goals_and_routes = {}
        for goal_idx, goal in enumerate(goals):
            route = self.route_to_goal(start_lanelet, goal)
            if route is not None:
                goals_and_routes[goal_idx] = route
        return goals_and_routes

    def route_to_goal(self, start_lanelet, goal):
        goal_point = BasicPoint2d(goal[0], goal[1])
        end_lanelets = self.lanelets_at(goal_point)
        best_route = None
        for end_lanelet in end_lanelets:
            route = self.routing_graph.getRoute(start_lanelet, end_lanelet)
            if (route is not None and route.shortestPath()[-1] == end_lanelet
                    and (best_route is None or route.length2d() < best_route.length2d())):
                best_route = route
        return best_route

    def get_current_lanelet(self, state, previous_lanelet=None):
        point = state.point
        radius = 3
        bounding_box = BoundingBox2d(BasicPoint2d(point.x - radius, point.y - radius),
                                     BasicPoint2d(point.x + radius, point.y + radius))
        nearby_lanelets = self.lanelet_map.laneletLayer.search(bounding_box)

        best_lanelet = None
        best_angle_diff = None
        best_dist = None
        best_can_pass = False
        for lanelet in nearby_lanelets:
            if self.traffic_rules.canPass(lanelet):
                dist_from_point = geometry.distance(lanelet, point)
                angle_diff = abs(self.angle_in_lane(state, lanelet))
                can_pass = (False if previous_lanelet is None
                            else self.can_pass(previous_lanelet, lanelet))
                if (angle_diff < np.pi/2
                        and (best_lanelet is None
                             or (can_pass and not best_can_pass)
                             or ((can_pass or not best_can_pass)
                                 and (dist_from_point < best_dist
                                 or (best_dist == dist_from_point
                                     and angle_diff < best_angle_diff))))):
                    best_lanelet = lanelet
                    best_angle_diff = angle_diff
                    best_dist = dist_from_point
                    best_can_pass = can_pass
        return best_lanelet

    @staticmethod
    def get_vehicles_in_front(route, frame):
        path = route.shortestPath()
        agents = []
        for agent_id, agent in frame.agents.items():
            for lanelet in path:
                if geometry.inside(lanelet, agent.point):
                    agents.append(agent_id)
        return agents

    def get_lanelet_sequence(self, states):
        # get the correspoding lanelets for a seqeunce of frames
        lanelets = []
        lanelet = None
        for state in states:
            lanelet = self.get_current_lanelet(state, lanelet)
            lanelets.append(lanelet)
        return lanelets

    def can_pass(self, a, b):
        # can we legally pass directly from lanelet a to b
        return (a == b or LaneletHelpers.follows(b, a)
                or self.traffic_rules.canChangeLane(a, b))

    def lanelets_at(self, point):
        nearest_lanelets = geometry.findNearest(self.lanelet_map.laneletLayer, point, 1)
        matching_lanelets = []
        for distance, lanelet in nearest_lanelets:
            if distance == 0 and self.traffic_rules.canPass(lanelet):
                matching_lanelets.append(lanelet)
        return matching_lanelets

    def lanelet_at(self, point):
        lanelets = self.lanelets_at(point)
        if len(lanelets) == 0:
            return None
        return lanelets[0]

    @staticmethod
    def in_correct_lane(route):
        path = route.shortestPath()
        return len(path) == len(path.getRemainingLane(path[0]))

    @classmethod
    def path_to_goal_length(cls, state, goal, route):
        end_point = BasicPoint2d(*goal)
        return cls.path_to_point_length(state, end_point, route)

    @classmethod
    def vehicle_in_front(cls, state, route, frame):
        vehicles_in_front = cls.get_vehicles_in_front(route, frame)
        min_dist = np.inf
        vehicle_in_front = None

        # find vehicle in front with closest distance
        for agent_id in vehicles_in_front:
            dist = geometry.distance(frame.agents[agent_id].point, state.point)
            if 0 < dist < min_dist:
                vehicle_in_front = agent_id
                min_dist = dist

        return vehicle_in_front, min_dist

    @staticmethod
    def path_to_point_length(state, point, route):
        path = route.shortestPath()
        end_lanelet = path[-1]
        end_lanelet_dist = LaneletHelpers.dist_along(end_lanelet, point)

        start_point = BasicPoint2d(state.x, state.y)
        start_lanelet = path[0]
        start_lanelet_dist = LaneletHelpers.dist_along(start_lanelet, start_point)

        dist = end_lanelet_dist - start_lanelet_dist
        if len(path) > 1:
            prev_lanelet = start_lanelet
            for idx in range(len(path) - 1):
                lanelet = path[idx]
                lane_change = (prev_lanelet.leftBound == lanelet.rightBound
                               or prev_lanelet.rightBound == lanelet.leftBound)
                if not lane_change:
                    dist += geometry.length2d(lanelet)
        return dist

    @staticmethod
    def angle_to_goal(state, goal):
        goal_heading = np.arctan2(goal[1] - state.y, goal[0] - state.x)
        return np.diff(np.unwrap([goal_heading, state.heading]))[0]

    def lanelets_to_cross(self, route):
        # get higher priority lanelets to cross
        # TODO: may need testing on new scenarios (other than heckstrasse)
        path = route.shortestPath()
        lanelets_to_cross = []
        crossing_points = []

        crossed_line = False
        for path_lanelet in path:
            if not crossed_line:
                crossed_line_lanelet, crossing_point = self.lanelet_crosses_line(path_lanelet)
                if crossed_line_lanelet is not None:
                    lanelets_to_cross.append(crossed_line_lanelet)
                    crossing_points.append(crossing_point)
                    crossed_line = True

            if crossed_line:
                # check if merged
                if len(self.routing_graph.previous(path_lanelet)) > 1:
                    crossed_line = False
                else:
                    for lanelet in self.lanelet_map.laneletLayer:
                        crossing_point = self.lanelet_crosses_lanelet(path_lanelet, lanelet)
                        if crossing_point is not None:
                            lanelets_to_cross.append(lanelet)
                            crossing_points.append(crossing_point)

        return lanelets_to_cross, crossing_points

    def oncoming_vehicle_dist(self, route, frame, max_dist=100):
        oncoming_vehicles = self.oncoming_vehicles(route, frame, max_dist)
        min_dist = max_dist
        for agent_id, (agent, dist) in oncoming_vehicles.items():
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def oncoming_vehicles(self, route, frame, max_dist=30):
        # get oncoming vehicles in lanes to cross
        oncoming_vehicles = {}
        lanelets_to_cross, crossing_points = self.lanelets_to_cross(route)
        for lanelet, point in zip(lanelets_to_cross, crossing_points):
            lanelet_start_dist = LaneletHelpers.dist_along(lanelet, point)
            lanelet_oncoming_vehicles = self.lanelet_oncoming_vehicles(
                frame, lanelet, lanelet_start_dist, max_dist)

            for agent_id, (agent, dist) in lanelet_oncoming_vehicles.items():
                if agent_id not in oncoming_vehicles or dist < oncoming_vehicles[agent_id][1]:
                    oncoming_vehicles[agent_id] = (agent, dist)

        return oncoming_vehicles

    def lanelet_oncoming_vehicles(self, frame, lanelet, lanelet_start_dist, max_dist):
        # get vehicles oncoming to a root lanelet
        oncoming_vehicles = {}
        for agent_id, agent in frame.agents.items():
            if geometry.inside(lanelet, agent.point):
                dist_along_lanelet = LaneletHelpers.dist_along(lanelet, agent.point)
                total_dist_along = lanelet_start_dist - dist_along_lanelet
                if total_dist_along < max_dist:
                    oncoming_vehicles[agent_id] = (agent, total_dist_along)

        if lanelet_start_dist < max_dist:
            for prev_lanelet in self.routing_graph.previous(lanelet):
                if self.traffic_rules.canPass(prev_lanelet):
                    prev_lanelet_start_dist = lanelet_start_dist + geometry.length2d(prev_lanelet)
                    prev_oncoming_vehicles = self.lanelet_oncoming_vehicles(
                        frame, prev_lanelet, prev_lanelet_start_dist, max_dist)

                    for agent_id, (agent, dist) in prev_oncoming_vehicles.items():
                        if agent_id not in oncoming_vehicles or dist < oncoming_vehicles[agent_id][1]:
                            oncoming_vehicles[agent_id] = (agent, dist)

        return oncoming_vehicles

    def lanelet_crosses_lanelet(self, path_lanelet, lanelet):
        # check if a lanelet crosses another lanelet, return overlap centroid
        if path_lanelet != lanelet and self.traffic_rules.canPass(lanelet):
            overlap_area, centroid = LaneletHelpers.overlap_area(path_lanelet, lanelet)
            split = (self.routing_graph.previous(path_lanelet)
                     == self.routing_graph.previous(lanelet))
            if overlap_area > 1 and not split:
                return centroid
        return None

    def lanelet_crosses_line(self, path_lanelet):
        # check if the midline of a lanelet crosses a road marking
        for lanelet in self.lanelet_map.laneletLayer:
            if path_lanelet != lanelet and self.traffic_rules.canPass(lanelet):
                left_virtual = lanelet.leftBound.attributes['type'] == 'virtual'
                right_virtual = lanelet.rightBound.attributes['type'] == 'virtual'
                path_centerline = geometry.to2D(path_lanelet.centerline)
                right_bound = geometry.to2D(lanelet.rightBound)
                left_bound = geometry.to2D(lanelet.leftBound)
                left_intersects = (not left_virtual and
                                   geometry.intersects2d(path_centerline, left_bound))
                right_intersects = (not right_virtual and
                                    geometry.intersects2d(path_centerline, right_bound))
                if path_lanelet != lanelet:
                    if left_intersects:
                        intersection_point = LaneletHelpers.intersection_point(
                            path_centerline, left_bound)
                        return lanelet, intersection_point
                    elif right_intersects:
                        intersection_point = LaneletHelpers.intersection_point(
                            path_centerline, right_bound)
                        return lanelet, intersection_point
        else:
            return None, None

    def goal_type(self, state, goal, route):
        # get the goal type, based on the route
        goal_point = BasicPoint2d(*goal)
        path = route.shortestPath()
        start_heading = LaneletHelpers.heading_at(path[0], state.point)
        end_heading = LaneletHelpers.heading_at(path[-1], goal_point)
        angle_to_goal = np.diff(np.unwrap([end_heading, start_heading]))[0]

        if -np.pi/8 < angle_to_goal < np.pi/8:
            return 'straight-on'
        elif np.pi/8 <= angle_to_goal < np.pi * 3/4:
            return 'turn-right'
        elif -np.pi/8 >= angle_to_goal > np.pi * -3/4:
            return 'turn-left'
        else:
            return 'u-turn'


class GoalDetector:
    """ Detects the goals of agents based on their trajectories"""

    def __init__(self, possible_goals, dist_threshold=1.5):
        self.dist_threshold = dist_threshold
        self.possible_goals = possible_goals

    def detect_goals(self, frames):
        goals = []
        goal_frames = []
        for frame in frames:
            agent_point = np.array([frame.x, frame.y])
            for goal_idx, goal_point in enumerate(self.possible_goals):
                dist = np.linalg.norm(agent_point - goal_point)
                if dist <= self.dist_threshold and goal_idx not in goals:
                    goals.append(goal_idx)
                    goal_frames.append(frame.frame_id)
        return goals, goal_frames

    def get_agents_goals_ind(self, tracks, static_info, meta_info, map_meta, agent_class='car'):
        goal_locations = map_meta.goals
        agent_goals = {}
        for track_idx in range(len(static_info)):
            if static_info[track_idx]['class'] == agent_class:
                track = tracks[track_idx]
                agent_goals[track_idx] = []

                for i in range(static_info[track_idx]['numFrames']):
                    point = np.array([track['xCenter'][i], track['yCenter'][i]])
                    for goal_idx, loc in enumerate(goal_locations):
                        dist = np.linalg.norm(point - loc)
                        if dist < self.dist_threshold and loc not in agent_goals[track_idx]:
                            agent_goals[track_idx].append(loc)
        return agent_goals