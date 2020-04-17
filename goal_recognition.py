import numpy as np
import json
import imageio
import cv2
import matplotlib.pyplot as plt
import lanelet2
from lanelet2.core import LaneletMap, BasicPoint2d
from lanelet2.geometry import findNearest

import map_vis_lanelet2
from tracks_import import read_from_csv


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


class AgentData:

    def __init__(self, state_history, metadata):
        self.state_history = state_history
        self.metadata = metadata


class AgentMetadata:

    def __init__(self, agent_id, width, length, agent_type, initial_frame, final_frame):
        self.agent_id = agent_id
        self.width = width
        self.length = length
        self.agent_type = agent_type
        self.initial_frame = initial_frame
        self.final_frame = final_frame
        self.num_frames = final_frame - initial_frame + 1


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
            for idx, frame in enumerate(track):
                state = self._state_from_tracks(track, idx)
                state_history.append(state)
                frames[state.frame_id].add_agent_state(agent_meta.agent_id, state)
            agent = AgentData(state_history, agent_meta)
            agents[agent_meta.agent_id] = agent

        return Episode(agents, frames)

    @staticmethod
    def _state_from_tracks(track, idx):
        return AgentState(track['frame'][idx],
                   track['xCenter'][idx],
                   track['yCenter'][idx],
                   track['xVelocity'][idx],
                   track['yVelocity'][idx],
                   track['heading'][idx],
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
        plt.plot(*zip(*goal_locations), 'ro', zorder=12, markersize=20)
        for i in range(len(goal_locations)):
            label = 'G{}'.format(i)
            axes.annotate(label, goal_locations[i], zorder=12, color='white')


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
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.Vehicle)
        self.routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)

    def extract_features(self, agent_id, frames, goal):
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

        route = self.route_to_goal(current_state, goal)
        if route is None:
            raise ValueError('Unreachable goal')

        speed = current_state.v_lon
        acceleration = current_state.a_lon
        in_correct_lane = self.in_correct_lane(route)
        distance_to_goal = self.distance_to_goal(route)

        return [distance_to_goal, in_correct_lane, speed, acceleration]

    def route_to_goal(self, state, goal):
        start_lanelet = self.get_current_lanelet(state)
        goal_point = BasicPoint2d(goal[0], goal[1])
        end_lanelet = self.lanelet_map.LaneletLayer.nearest(goal_point, 1)[0]
        route = self.routing_graph.getRoute(start_lanelet, end_lanelet)
        if route[-1] != end_lanelet:
            return None
        else:
            return route

    def get_current_lanelet(self, state):
        point = BasicPoint2d(state.x, state.y)
        lanelet = self.lanelet_map.laneletLayer.nearest(point, 1)[0]
        return lanelet

    @staticmethod
    def in_correct_lane(route):
        for i in range(0, len(route) - 1):
            if route[i+1].id in [route[i].left_id, route[i].right_id]:
                return False
        return True

    @staticmethod
    def distance_to_goal(route):
        # TODO - add distance travelled in first and last lanelet -
        # investigate route object
        return route.length2d()


if __name__ == '__main__':
    scenario = Scenario.load('scenario_config/heckstrasse.json')
    scenario.plot()
    plt.show()




