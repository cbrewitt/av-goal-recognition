import numpy as np
import lanelet2
from lanelet2.core import BasicPoint2d, BoundingBox2d
from lanelet2 import geometry

from av_goal_recognition.lanelet_helpers import LaneletHelpers


class FeatureExtractor:

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

        return {'path_to_goal_length': path_to_goal_length,
                'in_correct_lane': in_correct_lane,
                'speed': speed,
                'acceleration': acceleration,
                'angle_in_lane': angle_in_lane,
                'angle_to_goal': angle_to_goal}

    @staticmethod
    def angle_in_lane(state, lanelet):
        lane_heading = LaneletHelpers.heading_at(lanelet, state.point)
        angle_diff = np.diff(np.unwrap([lane_heading, state.heading]))
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
                             or (dist_from_point < best_dist
                                 or (best_dist == dist_from_point
                                     and angle_diff < best_angle_diff)))):
                    best_lanelet = lanelet
                    best_angle_diff = angle_diff
                    best_dist = dist_from_point
                    best_can_pass = can_pass
        return best_lanelet

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

    @staticmethod
    def path_to_goal_length(state, goal, route):
        path = route.shortestPath()

        end_point = BasicPoint2d(*goal)
        end_lanelet = path[-1]
        end_lanelet_dist = LaneletHelpers.dist_along(end_lanelet, end_point)

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
        return np.diff(np.unwrap([goal_heading, state.heading]))