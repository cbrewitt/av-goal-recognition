#  This file is a modified version of the config class from the file:
# https://github.com/nrhine1/precog/blob/master/precog/dataset/preprocess_nuscenes.py
from typing import List

import numpy as np

from core.scenario import Episode


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

        #  Minimum OTHER agents visible. In our case all agents are OTHER agents
        # There is no dedicated ego vehicle
        self.min_relevant_agents = 1  # A

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
    def from_ind_trajectory(cls, agents_to_include: List[int], episode: Episode,
                            reference_frame: int, cfg: InDConfig):
        # There is no explicit ego in the InD dataset
        player_past = []
        player_future = []
        player_yaw = None

        all_agent_pasts = []
        all_agent_futures = []
        all_agent_yaws = []

        for agent_id in agents_to_include:
            agent = episode.agents[agent_id]
            local_frame = reference_frame - agent.initial_frame

            agent_past_trajectory = agent.state_history[local_frame - cfg.past_horizon_length:local_frame]
            all_agent_pasts.append([[frame.x, frame.y, 0.0] for frame in agent_past_trajectory])

            agent_future_trajectory = agent.state_history[local_frame:local_frame + cfg.future_horizon_length]
            all_agent_futures.append([[frame.x, frame.y, 0.0] for frame in agent_future_trajectory])

            all_agent_yaws.append(agent.state_history[local_frame].heading)

        # TODO: Use UTM coordinate frame instead?
        # (A, Tp, d). Agent past trajectories in local frame at t=now
        agent_pasts = np.stack(all_agent_pasts, axis=0)
        # (A, Tf, d). Agent future trajectories in local frame at t=now
        agent_futures = np.stack(all_agent_futures, axis=0)
        # (A,). Agent yaws in ego frame at t=now
        agent_yaws = np.asarray(all_agent_yaws)

        assert agent_pasts.shape[0] == agent_futures.shape[0] == agent_yaws.shape[0] == len(agents_to_include)

        overhead_features = None

        metadata = {}

        return cls(player_past, agent_pasts,
                   player_future, agent_futures,
                   player_yaw, agent_yaws,
                   overhead_features,
                   metadata)