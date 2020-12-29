#  This file is a modified version of the config class from the file:
# https://github.com/nrhine1/precog/blob/master/precog/dataset/preprocess_nuscenes.py
import numpy as np


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
    def from_ind_trajectory(cls, trajectory):
        pass


class InDConfig:
    def __init__(self, episode_meta):
        self.episode_meta = episode_meta
        self._create_config()

    def _create_config(self):
        self.frame_rate = self.episode_meta["frameRate"]  # Hz

        # samples are at 2Hz in the original Precog implementation
        self.sample_rate = 2  # Hz
        self.sample_frequency = 1. / self.sample_rate

        # Predict 4 seconds in the future with 1 second of past.
        self.past_horizon_seconds = 1
        self.future_horizon_seconds = 4

        # The number of samples we need.
        self.future_horizon_length = int(round(self.future_horizon_seconds * self.frame_rate))
        self.past_horizon_length = int(round(self.past_horizon_seconds * self.frame_rate))

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
