import os
import argparse
import dill

from core.base import get_scenario_config_dir, get_data_dir
from core.feature_extraction import GoalDetector
from core.scenario import Scenario
from precog.ind_util import InDConfig, InDMultiagentDatum


def prepare_dataset(scenario_name, train_fraction=0.6, valid_fraction=0.2, samples_per_trajectory=10):
    scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
    episodes = scenario.load_episodes()
    for episode_idx, episode in enumerate(episodes):
        cfg = InDConfig(episode.static_meta)

        training_samples_list = []
        validation_samples_list = []
        test_samples_list = []

        print('episode {}/{}'.format(episode_idx, len(episodes) - 1))
        num_frames = len(episode.frames)

        training_set_cutoff_frame = int(num_frames * train_fraction)
        validation_set_cuttoff_frame = training_set_cutoff_frame + int(num_frames * valid_fraction)

        goals = {}  # key: agent id, value: goal idx
        trimmed_trajectories = {}

        # detect goal, and trim trajectory past the goal
        goal_detector = GoalDetector(scenario.config.goals)
        for agent_id, agent in episode.agents.items():
            if agent.agent_type in ['car', 'truck_bus']:
                agent_goals, goal_frames = goal_detector.detect_goals(agent.state_history)
                if len(agent_goals) > 0:
                    final_goal_frame_idx = goal_frames[-1] - agent.initial_frame
                    trimmed_trajectory = agent.state_history[0:final_goal_frame_idx]
                    goals[agent_id] = agent_goals[-1]
                    trimmed_trajectories[agent_id] = trimmed_trajectory

        for agent_idx, (agent_id, trajectory) in enumerate(trimmed_trajectories.items()):
            print('agent_id {}/{}'.format(agent_idx, len(trimmed_trajectories) - 1))
            # iterate through each sampled point in time for trajectory

            split_type = None
            if trajectory[-1].frame_id <= training_set_cutoff_frame:
                split_type = "train"
            elif (trajectory[0].frame_id > training_set_cutoff_frame
                  and trajectory[-1].frame_id <= validation_set_cuttoff_frame):
                split_type = "val"
            elif trajectory[0].frame_id > validation_set_cuttoff_frame:
                split_type = "test"

            for sample in trajectory:
                datum = InDMultiagentDatum.from_ind_trajectory(trajectory)

                with open(os.path.join(get_data_dir(), "precog", split_type, "ma_datum_{count:06d}.dill"), "wb") as f:
                    dill.dump(datum, f)


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    args = parser.parse_args()

    if args.scenario is None:
        scenarios = ['heckstrasse', 'bendplatz', 'frankenberg']
    else:
        scenarios = [args.scenario]

    for scenario_name in scenarios:
        print('Processing dataset for scenario: ' + scenario_name)
        prepare_dataset(scenario_name)


if __name__ == '__main__':
    main()
