#  This file is a modified version of the config class from the file:
# https://github.com/nrhine1/precog/blob/master/precog/dataset/preprocess_nuscenes.py
from typing import List

import numpy as np

from core.scenario import Frame


class InDConfig:
    def __init__(self, recordings_meta):
        self.recordings_meta = recordings_meta
        self._create_config()

    def _create_config(self):
        self.frame_rate = self.recordings_meta["frameRate"]  # Hz

        # samples are at 2Hz in the original Precog implementation
        self.sample_period = 2  # Hz
        self.sample_frequency = 1. / self.sample_period

        # Predict 4 seconds in the future with 1 second of past.
        self.past_horizon_seconds = 1  # Tp
        self.future_horizon_seconds = 2  # Tf

        # The number of samples we need.
        self.future_horizon_length = int(round(self.future_horizon_seconds * self.frame_rate))
        self.past_horizon_length = int(round(self.past_horizon_seconds * self.frame_rate))
        self.total_length = self.past_horizon_length + self.future_horizon_length

        # Minimum OTHER agents visible.
        self.min_relevant_agents = 1

        #  Configuration for temporal interpolation.
        # For InD usually this is not necessary as the frame rate of the recording is high
        # but the sample rate is low
        self.target_sample_period_past = 5
        self.target_sample_period_future = 5
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


class InDMultiagentDatum:
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
    def from_ind_trajectory(cls, agent_id: int, trajectory: List[Frame], cfg: InDConfig):
        #  Get how many segments we can extract from the trajectory
        # given by the Tp and Tf times in the config
        num_segments = len(trajectory) // cfg.total_length

        if num_segments < 1:
            print(f"Agent {agent_id}: Not enough frames to build segment. "
                  f"Required: {cfg.total_length}; Got {len(trajectory)}")
            return []

        ret = []

        # Makes sure that the goal frame is included
        start_frame_idx = len(trajectory) - num_segments * cfg.total_length
        for seg_idx in range(num_segments):
            end_frame_idx = start_frame_idx + cfg.total_length
            subtrajectory = trajectory[start_frame_idx:end_frame_idx]
            ret.append(cls._process_subtrajectory(subtrajectory, cfg))
            start_frame_idx += cfg.total_length

        return ret

    @classmethod
    def _process_subtrajectory(cls, subtrajectory, cfg):
        assert(len(subtrajectory) == cfg.total_length)

        player_past = []

        agent_pasts = []

        player_future = []

        agent_futures = []

        agent_yaws = []

        overhead_features = None

        metadata = {}

        return cls(player_past, agent_pasts,
                   player_future, agent_futures,
                   player_yaw=0.0, agent_yaws=agent_yaws,
                   overhead_features=overhead_features,
                   metadata=metadata)