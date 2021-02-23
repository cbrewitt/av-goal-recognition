#  This file is a modified version of the config class from the file:
# https://github.com/nrhine1/precog/blob/master/precog/dataset/preprocess_nuscenes.py
from typing import List

import numpy as np
import cv2 as cv
import os
import copy

from shapely.geometry import LineString
from core.base import get_data_dir
from core.scenario import AgentState


def collapse(arr):
    ret = arr[:, :, 0]
    for layer in arr.transpose((2, 0, 1)):
        mask = layer < 255
        ret[mask] = layer[mask]
    return ret


def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_CUBIC, borderValue=255)
    return result


class InDConfig:
    def __init__(self, scenario, recordings_meta, draw_map=False):
        self.recordings_meta = recordings_meta
        self.draw_map = draw_map

        self._create_config(scenario)
        self.scenario_features = self._process_scenario(scenario)

    def _create_config(self, scenario):
        self.frame_rate = self.recordings_meta["frameRate"]  # Hz
        self.inv_frame_rate = 1 / self.frame_rate

        # samples are at 2Hz in the original Precog implementation
        self.sample_period = 2  # Hz
        self.sample_frequency = 1. / self.sample_period

        # Predict 2 seconds in the future with 2 seconds of past.
        self.past_horizon_seconds = 2  # Tp
        self.future_horizon_seconds = 4  # Tf
        self.use_padding = True

        # The number of samples we need.
        self.future_horizon_length = int(round(self.future_horizon_seconds * self.frame_rate))
        self.past_horizon_length = int(round(self.past_horizon_seconds * self.frame_rate))
        self.total_length = self.past_horizon_length + self.future_horizon_length

        #  Minimum OTHER agents visible. In our case all agents are OTHER agents
        # There is no dedicated ego vehicle
        self.min_relevant_agents = 1  # A

        if scenario.config.name == "round":
            self.downsample_factor = 1 / 4
        else:
            self.downsample_factor = 1 / 3
        self.image_dims = (900, 700)

        self.vehicle_colors = {
            "car": 0,
            "truck": 0,
            "truck_bus": 0
        }

        self.image_colors = {
            "background": 255,
            "walkway": 200,
            "exit": 180,
            "road": 63,
            "bicycle_lane": 127,
            "parking": 100,
            "freespace": 100,
            "vegetation": 100,
            "keepout": 100
        }

        # Space between dashed markings and length of dashes in metres
        self.dash_space = 2
        self.marking_length = 2
        self.arrow_thickness = 2
        self.marking_thickness = 1
        self.marking_color = 240

        # Configuration for temporal interpolation.
        self.target_sample_period_past = 10
        self.target_sample_period_future = 10
        self.target_sample_frequency_past = 1. / self.target_sample_period_past
        self.target_sample_frequency_future = 1. / self.target_sample_period_future
        self.target_past_horizon = self.past_horizon_seconds * self.target_sample_period_past
        self.target_future_horizon = self.future_horizon_seconds * self.target_sample_period_future
        self.target_past_times = -1 * np.arange(0, self.past_horizon_seconds, self.target_sample_frequency_past)[::-1]
        # Hacking negative zero to slightly positive for temporal interpolation purposes?
        self.target_past_times[np.isclose(self.target_past_times, 0.0, atol=1e-5)] = 1e-8

        # The final relative future times to interpolate
        self.target_future_times = np.arange(self.target_sample_frequency_future,
                                             self.future_horizon_seconds + self.target_sample_frequency_future,
                                             self.target_sample_frequency_future)

    def _process_scenario(self, scenario):
        features_list = []

        pavement_layer = self._get_surfaces_layer(scenario)
        features_list.extend(pavement_layer)

        markings_layer = self._get_markings_layer(scenario)
        features_list.extend(markings_layer)

        return features_list

    def _get_surfaces_layer(self, scenario):
        layer_keys = ["walkway", "exit", "road", "bicycle_lane", "parking"]
        w, h = scenario.display_wh
        layers = [np.full((h, w), self.image_colors["background"], np.uint8) for _ in range(len(layer_keys))]
        scale = 1.0 / scenario.config.background_px_to_meter

        def draw_surface(elem):
            if "subtype" in elem.attributes:
                polygon = elem.polygon2d() if hasattr(elem, "polygon2d") else elem.outerBoundPolygon()
                points = [[int(scale * pt.x), int(-scale * pt.y)] for pt in polygon]
                if points and elem.attributes["subtype"] in layer_keys:
                    subtype = elem.attributes["subtype"]
                    layer_idx = layer_keys.index(subtype)
                    points = np.stack(points)
                    points[points < 0] = 0
                    points = points.reshape((-1, 1, 2))
                    cv.fillPoly(layers[layer_idx], [points], self.image_colors[subtype], cv.LINE_AA)

        for lanelet in scenario.lanelet_map.laneletLayer:
            draw_surface(lanelet)
        for area in scenario.lanelet_map.areaLayer:
            draw_surface(area)

        return layers

    def _get_markings_layer(self, scenario):
        marking_keys = ["line_thin_dashed", "line_thick_dashed", "line_thin_solid", "line_think_solid",
                        "arrow_straight", "arrow_left", "arrow_right", "arrow_straight_right", "arrow_straight_left"]
        w, h = scenario.display_wh
        layers = [np.full((h, w), self.image_colors["background"], np.uint8) for _ in range(2)]
        scale = 1.0 / scenario.config.background_px_to_meter

        for linestring in scenario.lanelet_map.lineStringLayer:
            if "type" in linestring.attributes and "subtype" in linestring.attributes:
                points = [[int(scale * pt.x), int(-scale * pt.y)] for pt in linestring]
                if not points:
                    continue

                points = np.stack(points)
                points[points < 0] = 0

                line_key = f"{linestring.attributes['type']}_{linestring.attributes['subtype']}"

                if line_key in marking_keys:
                    layer_idx = 0 if linestring.attributes["type"].startswith("line") else 1

                    if linestring.attributes["subtype"] == "dashed":
                        ls = LineString(points)
                        pts = []
                        for dist in np.arange(0, ls.length, scale * (self.dash_space + self.marking_length)):
                            start_pt = ls.interpolate(dist)
                            end_pt = ls.interpolate(dist + scale * self.marking_length)
                            line_pts = np.array([[start_pt.x, start_pt.y], [end_pt.x, end_pt.y]], np.int32)
                            line_pts = line_pts.reshape(-1, 1, 2)
                            pts.append(line_pts)
                        cv.polylines(layers[layer_idx], pts, False, self.marking_color, self.marking_thickness)

                    elif linestring.attributes["type"] == "arrow":
                        dir_vector = (points[1] - points[0]) / np.linalg.norm(points[1] - points[0])
                        norm_vector = np.array([[0, -1], [1, 0]]) @ dir_vector
                        if linestring.attributes["subtype"] in ["left", "right"]:
                            dir_mul = scale if linestring.attributes["subtype"] == "right" else -scale
                            p1 = points[1] + dir_mul * 0.7 * norm_vector
                            p2 = points[1] + dir_mul * 0.5 * norm_vector + scale * dir_vector
                            p3 = points[1] + dir_mul * 0.5 * norm_vector - scale * dir_vector
                            head = [p1, p2, p3, p1]

                        elif linestring.attributes["subtype"] in ["straight_left", "straight_right"]:
                            dir_mul = scale if linestring.attributes["subtype"] == "straight_right" else -scale
                            p0 = points[1] - scale * 2 * dir_vector
                            p1 = p0 + dir_mul * 0.7 * norm_vector
                            p2 = p0 + dir_mul * 0.5 * norm_vector + scale * 0.75 * dir_vector
                            p3 = p0 + dir_mul * 0.5 * norm_vector - scale * 0.75 * dir_vector
                            p4 = points[1] + scale * 2 * dir_vector
                            p5 = points[1] + scale * 0.3 * norm_vector
                            p6 = points[1] - scale * 0.3 * norm_vector
                            head = [p0, p1, p2, p3, p1, p0, p4, p5, p6, p4]

                        else:  # Straight arrow
                            p1 = points[1] - scale * 2 * dir_vector + scale * 0.3 * norm_vector
                            p2 = points[1] - scale * 2 * dir_vector - scale * 0.3 * norm_vector
                            head = [p1, p2, points[1]]

                        points = np.append(points, head, axis=0).astype(np.int32)
                        points = points.reshape(-1, 1, 2)
                        cv.polylines(layers[layer_idx], [points], False, self.marking_color, self.arrow_thickness)

                    else:
                        points = points.reshape(-1, 1, 2)
                        cv.polylines(layers[layer_idx], [points], False, self.marking_color, self.marking_thickness)
        return layers


class InDMultiagentDatum:
    t = 0

    def __init__(self, player_past, agent_pasts, player_future, agent_futures,
                 player_yaw, agent_yaws, overhead_features, metadata={}):
        self.player_past = player_past
        self.agent_pasts = agent_pasts
        self.player_future = player_future
        self.agent_futures = agent_futures
        self.player_yaw = player_yaw
        self.agent_yaws = agent_yaws
        self.overhead_features = overhead_features
        self.metadata = metadata

    @classmethod
    def from_ind_trajectory(cls, agents_to_include: List[int], goals, episode, scenario,
                            reference_frame: int, cfg: InDConfig):
        # There is no explicit ego in the InD dataset
        player_past = np.zeros((cfg.target_past_horizon, 3))
        player_future = np.zeros((cfg.target_future_horizon, 3))
        player_yaw = 0.0

        all_agent_pasts = []
        all_agent_futures = []
        all_agent_yaws = []

        agent_dims = []

        for agent_id in agents_to_include:
            agent = episode.agents[agent_id]
            if cfg.use_padding:
                state_history, (lp, rp) = InDMultiagentDatum.pad_trajectory(
                    agent.state_history,
                    reference_frame - cfg.past_horizon_length,
                    reference_frame + cfg.future_horizon_length,
                    cfg
                )
                local_frame = reference_frame - (agent.initial_frame - lp)
            else:
                state_history = agent.state_history
                local_frame = reference_frame - agent.initial_frame

            # Interpolate for past trajectory
            agent_past_trajectory = state_history[local_frame - cfg.past_horizon_length:local_frame]
            past_timestamps = -np.arange(0, cfg.past_horizon_seconds, cfg.inv_frame_rate)[::-1]
            agent_past_interpolated = InDMultiagentDatum.interpolate_trajectory(agent_past_trajectory,
                                                                                cfg.target_past_times,
                                                                                past_timestamps)
            all_agent_pasts.append(agent_past_interpolated)

            # Interpolation for future trajectory
            agent_future_trajectory = state_history[local_frame:local_frame + cfg.future_horizon_length]
            future_timestamps = np.arange(cfg.future_horizon_seconds, 0.0, -cfg.inv_frame_rate)[::-1]
            agent_future_interpolated = InDMultiagentDatum.interpolate_trajectory(agent_future_trajectory,
                                                                                  cfg.target_future_times,
                                                                                  future_timestamps)
            all_agent_futures.append(agent_future_interpolated)

            all_agent_yaws.append(state_history[local_frame - 1].heading)

            agent_dims.append([agent.width, agent.length, agent.agent_type])

        # (A, Tp, d). Agent past trajectories in local frame at t=now
        agent_pasts = np.stack(all_agent_pasts, axis=0)
        # (A, Tf, d). Agent future trajectories in local frame at t=now
        agent_futures = np.stack(all_agent_futures, axis=0)
        # (A,). Agent yaws in ego frame at t=now
        agent_yaws = np.asarray(all_agent_yaws)

        assert agent_pasts.shape[0] == agent_futures.shape[0] == agent_yaws.shape[0] == len(agents_to_include)
        assert agent_pasts.shape[1] == cfg.target_past_horizon
        assert agent_futures.shape[1] == cfg.target_future_horizon

        overhead_features, visualisation = InDMultiagentDatum.get_image_features(
            scenario, agent_pasts[:, -1, :], agent_yaws, agent_dims, cfg)

        # Scale the trajectory to match the units of the images features
        vis_scale = cfg.downsample_factor / scenario.config.background_px_to_meter
        agent_pasts *= vis_scale
        agent_futures *= vis_scale

        true_goals = np.zeros(1 + len(agents_to_include))
        for i, agent_id in enumerate(agents_to_include):
            true_goals[1 + i] = goals[agent_id]

        metadata = {"vis_layer": visualisation,
                    "vis_scale": vis_scale,
                    "agent_dims": agent_dims,
                    "true_goals": true_goals}

        return cls(player_past, agent_pasts,
                   player_future, agent_futures,
                   player_yaw, agent_yaws,
                   overhead_features,
                   metadata)

    @classmethod
    def get_image_features(cls, scenario, agent_poses, agent_yaws, agent_dims, cfg):
        features_list = cfg.scenario_features.copy()
        agents_layer = InDMultiagentDatum.get_agent_boxes(scenario, agent_poses, agent_yaws, agent_dims, cfg)
        features_list.append(agents_layer)

        features_list = InDMultiagentDatum.crop_resize_features(features_list, scenario, cfg)
        image = np.stack(features_list, axis=-1)

        visualisation = collapse(image).copy()
        if cfg.draw_map:
            path_to_viz = get_data_dir() + f"precog/{scenario.config.name}/map_viz/"
            if not os.path.exists(path_to_viz):
                os.mkdir(path_to_viz)
            cv.imwrite(os.path.join(path_to_viz, f"{cls.t}.png"), visualisation)
            cls.t += 1

        # Turn image to binary
        mask = image != 255
        image[mask] = 1
        image[~mask] = 0

        return image, visualisation

    @staticmethod
    def crop_resize_features(features_list, scenario, cfg):
        if scenario.config.name == "round":
            return [cv.resize(layer[100:, 100:], (0, 0),
                              fx=cfg.downsample_factor, fy=cfg.downsample_factor, interpolation=cv.INTER_AREA)
                    for layer in features_list]
        else:
            return [cv.resize(layer[:cfg.image_dims[1], :cfg.image_dims[0]], (0, 0),
                              fx=cfg.downsample_factor, fy=cfg.downsample_factor, interpolation=cv.INTER_AREA)
                    for layer in features_list]

    @staticmethod
    def pad_trajectory(trajectory, initial_frame, final_frame, cfg):
        agent_initial_frame = trajectory[0].frame_id
        agent_final_frame = trajectory[-1].frame_id

        pad_left = 0
        pad_right = 0
        if initial_frame < agent_final_frame < final_frame:
            pad_right = final_frame - agent_final_frame
        if initial_frame < agent_initial_frame < final_frame:
            pad_left = agent_initial_frame - initial_frame

        # Create constant-velocity lane-follow padding
        left = []
        f = trajectory[0]
        for i in range(pad_left, 0, -1):
            pad = copy.copy(f)
            pad.frame_id = f.frame_id - i
            pad.x = f.x - cfg.inv_frame_rate * i * f.v_x
            pad.y = f.y - cfg.inv_frame_rate * i * f.v_y
            left.append(pad)

        right = []
        f = trajectory[-1]
        for i in range(1, pad_right + 1):
            pad = copy.copy(f)
            pad.frame_id = f.frame_id + i
            pad.x = f.x + cfg.inv_frame_rate * i * f.v_x
            pad.y = f.y + cfg.inv_frame_rate * i * f.v_y
            right.append(pad)

        return left + trajectory + right, (pad_left, pad_right)

    @staticmethod
    def get_agent_boxes(scenario, agent_poses, agent_yaws, agent_dims, cfg):
        w, h = scenario.display_wh
        layer = np.full((h, w), cfg.image_colors["background"], np.uint8)

        for i in range(agent_poses.shape[0]):
            agent_dim = agent_dims[i]
            agent_box = Box(agent_poses[i, :], agent_dim[0], agent_dim[1], agent_yaws[i]).get_display_box(scenario)
            agent_box = agent_box.reshape((-1, 1, 2))
            color = cfg.vehicle_colors[agent_dim[2]]
            cv.fillPoly(layer, [agent_box], color, cv.LINE_AA)
        return layer

    @staticmethod
    def interpolate_trajectory(trajectory, target_timestamps, timestamps):
        poses = np.array([[frame.x, frame.y, 0.0] for frame in trajectory])
        ret = []
        for axis in poses.T:
            interp = np.interp(target_timestamps, timestamps, axis)
            ret.append(interp)
        return np.array(ret).T


class Box:
    def __init__(self, center, width, length, heading):
        self.center = np.array(center)[:2]
        self.width = width
        self.length = length
        self.heading = heading

        self.bounding_box = self.get_bounding_box()

    def get_bounding_box(self):
        box = np.array(
            [np.array([-self.length / 2, self.width / 2]),  # top-left
             np.array([self.length / 2, self.width / 2]),  # top-right
             np.array([-self.length / 2, -self.width / 2]),  # bottom-left
             np.array([self.length / 2, -self.width / 2])]  # bottom-right
        )

        rotation = np.array([[np.cos(self.heading), -np.sin(self.heading)],
                             [np.sin(self.heading), np.cos(self.heading)]])
        return self.center + box @ rotation.T

    def get_display_box(self, scenario):
        scale = 1.0 / scenario.config.background_px_to_meter
        permuter = np.array(  # Need to permute vertices for display
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
        )

        return permuter @ (scale * np.abs(self.bounding_box)).astype(np.int32)
