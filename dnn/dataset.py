import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from core.base import get_scenario_config_dir
from core.feature_extraction import GoalDetector
from core.scenario import Scenario
from core.data_processing import get_dataset

from dnn.dataset_base import GRITDataset


class GRITTrajectoryDataset(GRITDataset):
    def __init__(self, scenario_name, split_type="train"):
        super(GRITTrajectoryDataset, self).__init__(scenario_name, split_type)

        self.scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(self.scenario_name))
        self.episodes = self.scenario.load_episodes(self.dataset_split)

        self._prepare_dataset()

    def _prepare_dataset(self):
        trajectories = []
        goals = []
        lengths = []
        for episode_idx, episode in enumerate(self.episodes):
            trimmed_trajectories, gs, lens = self._trim_trajectories(episode)
            trajectories.extend(trimmed_trajectories)
            goals.extend(gs)
            lengths.extend(lens)
        sequences = pad_sequence(trajectories, batch_first=True, padding_value=0.0)
        goals = torch.LongTensor(goals)
        lengths = torch.Tensor(lengths)
        self.dataset, self.labels, self.lengths = sequences, goals, lengths

    def _trim_trajectories(self, episode):
        trimmed_trajectories = []
        goals = []
        lengths = []

        # detect goal, and trim trajectory past the goal
        goal_detector = GoalDetector(self.scenario.config.goals)
        for agent_id, agent in episode.agents.items():
            if agent.agent_type in ['car', 'truck_bus']:
                agent_goals, goal_frames = goal_detector.detect_goals(agent.state_history)
                if len(agent_goals) > 0:
                    final_goal_frame_idx = goal_frames[-1] - agent.initial_frame
                    trimmed_trajectory = agent.state_history[0:final_goal_frame_idx]
                    trimmed_trajectory = torch.Tensor([[frame.x, frame.y, frame.heading]
                                                       for frame in trimmed_trajectory])
                    goals.append(agent_goals[-1])
                    trimmed_trajectories.append(trimmed_trajectory)
                    lengths.append(trimmed_trajectory.shape[0])
        return trimmed_trajectories, goals, lengths


class GRITFeaturesDataset(GRITDataset):
    def __init__(self, scenario_name, split_type="train"):
        super(GRITFeaturesDataset, self).__init__(scenario_name, split_type)

        self._prepare_dataset()

    def _prepare_dataset(self):
        dataset = get_dataset(self.scenario_name, subset=self.split_type)
        dataset = dataset.reset_index(drop=True)

        timing_keys = ["frame_id", "initial_frame_id", "fraction_observed", "episode"]
        # self.timing_info = dataset[["agent_id", "goal_type"] + timing_keys]
        dataset = dataset.drop(timing_keys, axis=1)

        label_keys = ["true_goal"]  # , "true_goal_type"]
        # labels = pd.get_dummies(dataset[label_keys], columns=label_keys)
        labels = dataset[label_keys]
        dataset = dataset.drop(["true_goal", "true_goal_type"], axis=1)

        dataset["in_correct_lane"] = dataset["in_correct_lane"].astype(int)
        group_keys = ["agent_id", "goal_type", "possible_goal", "in_correct_lane"]
        groups = dataset.groupby(group_keys).groups
        dataset = pd.get_dummies(dataset, columns=["goal_type", "possible_goal"])
        dataset = dataset.drop(["agent_id"], axis=1)

        sequences = []
        goals = []
        lengths = []
        for key, indices in groups.items():
            if len(indices) > 11 and len(indices) % 11 == 0:
                for i in range(len(indices) // 11):
                    idxs = indices[11 * i:11 * i + 11]
                    sequences.append(torch.Tensor(dataset.loc[idxs].values))
                    goals.append(labels.loc[indices].values[0][0])
                    lengths.append(len(idxs))
            elif len(indices) <= 11:
                sequences.append(torch.Tensor(dataset.loc[indices].values))
                goals.append(labels.loc[indices].values[0][0])
                lengths.append(len(indices))

        sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        goals = torch.LongTensor(goals)
        lengths = torch.Tensor(lengths)
        self.dataset, self.labels, self.lengths = sequences, goals, lengths


DATASET_MAP = {"trajectory": GRITTrajectoryDataset, "features": GRITFeaturesDataset}
