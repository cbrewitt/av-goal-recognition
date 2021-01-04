import os
import argparse
import dill
import tqdm

from core.base import get_scenario_config_dir, get_data_dir
from core.feature_extraction import GoalDetector
from core.scenario import Scenario
from precog.ind_util import InDConfig, InDMultiagentDatum


def prepare_dataset(scenario_name, root_dir, train_fraction=0.6, valid_fraction=0.2, samples_per_trajectory=10):
    scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
    episodes = scenario.load_episodes()

    # Used for outputting incremental files for PRECOG processing
    count = 0

    for episode_idx, episode in enumerate(episodes):
        cfg = InDConfig(episode.recordings_meta)

        training_samples_list = []
        validation_samples_list = []
        test_samples_list = []

        print('\nEpisode {}/{}'.format(episode_idx + 1, len(episodes)))
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
        print("Finished preparing trimmed trajectories\n")

        # Sliding window sampling with a slide step of sample_freq * frame_rate
        prev_split = ""
        split_type = ""
        sample_step = int(cfg.sample_frequency * cfg.frame_rate)
        for initial_frame in tqdm.tqdm(range(0, len(episode.frames), sample_step), "Converting to PRECOG format"):
            final_frame = initial_frame + cfg.total_length

            # Process all valid agents which have at least Tp past and Tf future in its trajectory
            valid_agent_ids = []
            for agent_id in trimmed_trajectories:
                agent = episode.agents[agent_id]
                if agent.final_frame >= final_frame and \
                        agent.initial_frame <= initial_frame and \
                        agent_id in trimmed_trajectories:
                    valid_agent_ids.append(agent_id)

            if len(valid_agent_ids) >= cfg.min_relevant_agents:
                if final_frame <= training_set_cutoff_frame:
                    split_type = "train"
                elif (initial_frame > training_set_cutoff_frame
                      and final_frame <= validation_set_cuttoff_frame):
                    split_type = "val"
                elif initial_frame > validation_set_cuttoff_frame:
                    split_type = "test"

                ref_frame = initial_frame + cfg.past_horizon_length  # Corresponds to 'now' in the datum

                datum = InDMultiagentDatum.from_ind_trajectory(valid_agent_ids, episode, ref_frame, cfg)

                path_to_split = os.path.join(root_dir, split_type)
                if prev_split != split_type:
                    count = 0  # Reset counter if new split is started

                with open(os.path.join(path_to_split, f"ma_datum_{count:06d}.dill"), "wb") as f:
                    dill.dump(datum, f)
                prev_split = split_type
                count += 1


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    args = parser.parse_args()

    precog_path = os.path.join(get_data_dir(), "precog")
    assert(not os.path.exists(precog_path))
    os.mkdir(precog_path)

    if args.scenario is None:
        scenarios = ['heckstrasse', 'bendplatz', 'frankenberg']
    else:
        scenarios = [args.scenario]

    for scenario_name in scenarios:
        scenario_path = os.path.join(precog_path, scenario_name)
        os.mkdir(scenario_path)
        for split_str in ["train", "val", "test"]:
            os.mkdir(os.path.join(scenario_path, split_str))

        print('Processing dataset for scenario: ' + scenario_name)
        prepare_dataset(scenario_name, scenario_path)


if __name__ == '__main__':
    main()
