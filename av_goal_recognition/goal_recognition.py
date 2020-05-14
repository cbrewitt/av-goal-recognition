import numpy as np
import json
import imageio
import matplotlib.pyplot as plt
import lanelet2
from lanelet2.core import BasicPoint2d, BoundingBox2d
from lanelet2 import geometry

from av_goal_recognition import map_vis_lanelet2
from av_goal_recognition.tracks_import import read_from_csv
from av_goal_recognition.lanelet_helpers import LaneletHelpers


class ScenarioConfig:
    """Metadata about a scenario used for goal recognition"""

    def __init__(self, config_dict):
        self.config_dict = config_dict

    @classmethod
    def load(cls, file_path):
        """Loads the scenario metadata into from a json file

        Args:
            file_path (str): path to the file to load

        Returns:
            ScenarioConfig: metadata about the scenario

        """
        with open(file_path) as f:
            scenario_meta_dict = json.load(f)
        return cls(scenario_meta_dict)

    @property
    def goals(self):
        """list of [int, int]: Possible goals for agents in this scenario"""
        return self.config_dict.get('goals')

    @property
    def lanelet_file(self):
        """str: Path to the *.osm file specifying the lanelet2 map"""
        return self.config_dict.get('lanelet_file')

    @property
    def lat_origin(self):
        """float: Latitude of the origin"""
        return self.config_dict.get('lat_origin')

    @property
    def lon_origin(self):
        """float: Longitude of the origin"""
        return self.config_dict.get('lon_origin')

    @property
    def data_format(self):
        """str: Format in which the data is stored"""
        return self.config_dict.get('data_format')

    @property
    def data_root(self):
        """str: Path to directory in which the data is stored"""
        return self.config_dict.get('data_root')

    @property
    def episodes(self):
        """list of dict: Configuration for all episodes for this scenario"""
        return self.config_dict.get('episodes')

    @property
    def background_image(self):
        """str: Path to background image"""
        return self.config_dict.get('background_image')

    @property
    def background_px_to_meter(self):
        """float: Pixels per meter in background image"""
        return self.config_dict.get('background_px_to_meter')


class EpisodeConfig:
    """Metadata about an episode"""

    def __init__(self, config):
        self.config = config

    @property
    def recording_id(self):
        """str: unique id identifying the episode"""
        return self.config.get('recording_id')


class Agent:

    def __init__(self, state_history, metadata):
        self.state_history = state_history
        self.agent_id = metadata.agent_id
        self.width = metadata.width
        self.length = metadata.length
        self.agent_type = metadata.agent_type
        self.initial_frame = metadata.initial_frame
        self.final_frame = metadata.final_frame
        self.num_frames = metadata.final_frame - metadata.initial_frame + 1

    def plot_trajectory(self, *args, **kwargs):
        x = [s.x for s in self.state_history]
        y = [s.y for s in self.state_history]
        plt.plot(x, y, *args, **kwargs)


class AgentMetadata:

    def __init__(self, agent_id, width, length, agent_type, initial_frame, final_frame):
        self.agent_id = agent_id
        self.width = width
        self.length = length
        self.agent_type = agent_type
        self.initial_frame = initial_frame
        self.final_frame = final_frame


class AgentState:

    def __init__(self, frame_id, x, y, v_x, v_y, heading, a_x, a_y,
                 v_lon, v_lat, a_lon, a_lat):
        self.frame_id = frame_id
        self.x = x
        self.y = y
        self.v_x = v_x
        self.v_y = v_y
        self.heading = heading
        self.a_x = a_x
        self.a_y = a_y
        self.v_lon = v_lon
        self.v_lat = v_lat
        self.a_lon = a_lon
        self.a_lat = a_lat

    @property
    def point(self):
        return BasicPoint2d(self.x, self.y)

    def plot(self):
        plt.plot(self.x, self.y, 'yo')


class Frame:
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.agents = {}

    def add_agent_state(self, agent_id, state):
        self.agents[agent_id] = state


class Episode:

    def __init__(self, agents, frames):
        self.agents = agents
        self.frames = frames


class EpisodeLoader:

    def __init__(self, scenario_config):
        self.scenario_config = scenario_config

    def load(self, recording_id):
        raise NotImplementedError


class IndEpisodeLoader(EpisodeLoader):

    def load(self, config):
        track_file = self.scenario_config.data_root \
            + '/{}_tracks.csv'.format(config.recording_id)
        static_tracks_file = self.scenario_config.data_root \
            + '/{}_tracksMeta.csv'.format(config.recording_id)
        recordings_meta_file = self.scenario_config.data_root \
            + '/{}_recordingMeta.csv'.format(config.recording_id)
        tracks, static_info, meta_info = read_from_csv(
            track_file, static_tracks_file, recordings_meta_file)

        num_frames = round(meta_info['frameRate'] * meta_info['duration'])

        agents = {}
        frames = [Frame(i) for i in range(num_frames)]

        for track_meta in static_info:
            agent_meta = self._agent_meta_from_track_meta(track_meta)
            state_history = []
            track = tracks[agent_meta.agent_id]
            num_frames = agent_meta.final_frame - agent_meta.initial_frame + 1
            for idx in range(num_frames):
                state = self._state_from_tracks(track, idx)
                state_history.append(state)
                frames[state.frame_id].add_agent_state(agent_meta.agent_id, state)
            agent = Agent(state_history, agent_meta)
            agents[agent_meta.agent_id] = agent

        return Episode(agents, frames)

    @staticmethod
    def _state_from_tracks(track, idx):
        heading = np.deg2rad(track['heading'][idx])
        heading = np.unwrap([0, heading])[1]
        return AgentState(track['frame'][idx],
                   track['xCenter'][idx],
                   track['yCenter'][idx],
                   track['xVelocity'][idx],
                   track['yVelocity'][idx],
                   heading,
                   track['xAcceleration'][idx],
                   track['yAcceleration'][idx],
                   track['lonVelocity'][idx],
                   track['latVelocity'][idx],
                   track['lonAcceleration'][idx],
                   track['latAcceleration'][idx])

    @staticmethod
    def _agent_meta_from_track_meta(track_meta):
        return AgentMetadata(track_meta['trackId'],
                             track_meta['width'],
                             track_meta['length'],
                             track_meta['class'],
                             track_meta['initialFrame'],
                             track_meta['finalFrame'])


class EpisodeLoaderFactory:

    episode_loaders = {'ind': IndEpisodeLoader}

    @classmethod
    def get_loader(cls, scenario_config):
        loader = cls.episode_loaders[scenario_config.data_format]
        if loader is None:
            raise ValueError('Invalid data format')
        return loader(scenario_config)


class Scenario:
    def __init__(self, config):
        self.config = config
        self.lanelet_map = self.load_lanelet_map()
        self.episodes = self.load_episodes()

    def load_lanelet_map(self):
        origin = lanelet2.io.Origin(self.config.lat_origin, self.config.lon_origin)
        projector = lanelet2.projection.UtmProjector(origin)
        lanelet_map, _ = lanelet2.io.loadRobust(self.config.lanelet_file, projector)
        return lanelet_map

    def load_episodes(self):
        loader = EpisodeLoaderFactory.get_loader(self.config)
        episodes = [loader.load(EpisodeConfig(c)) for c in self.config.episodes]
        return episodes

    @classmethod
    def load(cls, file_path):
        config = ScenarioConfig.load(file_path)
        return cls(config)

    def plot(self):
        axes = plt.gca()
        map_vis_lanelet2.draw_lanelet_map(self.lanelet_map, axes)

        # plot background image
        if self.config.background_image is not None:
            background_path = self.config.data_root + self.config.background_image
            background = imageio.imread(background_path)
            rescale_factor = self.config.background_px_to_meter
            extent = (0, int(background.shape[1] * rescale_factor),
                      -int(background.shape[0] * rescale_factor), 0)
            plt.imshow(background, extent=extent)

        # plot goals
        goal_locations = self.config.goals
        plt.plot(*zip(*goal_locations), 'ro', markersize=20)
        for i in range(len(goal_locations)):
            label = 'G{}'.format(i)
            axes.annotate(label, goal_locations[i], color='white')


class GoalDetector:
    """ Detects the goals of agents based on their trajectories"""

    def __init__(self, dist_threshold=1.5):
        self.dist_threshold = dist_threshold

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

    def angle_in_lane(self, state, lanelet):
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
            dist_from_point = geometry.distance(lanelet, point)
            angle_diff = abs(self.angle_in_lane(state, lanelet))
            can_pass = (False if previous_lanelet is None
                        else self.can_pass(previous_lanelet, lanelet))
            if (angle_diff < np.pi/2
                    and self.traffic_rules.canPass(lanelet)
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
