import os
import argparse
import dill
import tqdm
import json
import pandas as pd
import numpy as np

from core.base import get_scenario_config_dir, get_data_dir, get_core_dir
from core.feature_extraction import GoalDetector, FeatureExtractor
from core.scenario import Scenario
from core.map_vis_lanelet2 import draw_lanelet_map
from precog.dataset.ind_util import InDConfig, InDMultiagentDatum


def dict_list_contains(d, elem):
    """ Get the key of the dictionary that contains the given element """
    for k, v in d.items():
        if elem in v:
            return k

def overlaps(a, b):
    """ Return the overlap between two intervals given as tuples"""
    return min(a[1], b[1]) - max(a[0], b[0])


def prepare_precog(scenario_name, root_dir):
    scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
    episodes = scenario.load_episodes()
    dataset_split = json.load(open(os.path.join(get_core_dir(), "dataset_split.json")))

    count = 0
    prev_split = ""
    df_list = []

    # Used for outputting incremental files for PRECOG processing
    for episode_idx, episode in enumerate(episodes):
        cfg = InDConfig(scenario, episode.recordings_meta, draw_map=True)

        split_type = dict_list_contains(dataset_split[scenario_name], episode_idx)
        if split_type == "valid": split_type = "val"
        if prev_split != split_type: count = 0  # Reset counter if new split is started
        prev_split = split_type

        print('\nEpisode {}/{} - {}'.format(episode_idx + 1, len(episodes), split_type))

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
        sample_step = int(cfg.sample_frequency * cfg.frame_rate)
        for initial_frame in tqdm.tqdm(range(0, len(episode.frames), sample_step), "Converting to PRECOG format"):
            ref_frame = initial_frame + cfg.past_horizon_length
            final_frame = initial_frame + cfg.total_length

            #  Process all valid agents which have at least Tp past and Tf future in its trajectory
            # Only use agents in trimmed_trajectories to ensure only relevant vehicles are used
            valid_agent_ids = []
            for agent_id, trajectory in trimmed_trajectories.items():
                agent = episode.agents[agent_id]
                if not cfg.use_padding:
                    if agent.final_frame >= final_frame and \
                            agent.initial_frame <= initial_frame:
                        valid_agent_ids.append(agent_id)
                else:
                    if agent.initial_frame <= ref_frame <= agent.final_frame:
                        valid_agent_ids.append(agent_id)

            if len(valid_agent_ids) >= cfg.min_relevant_agents:
                datum = InDMultiagentDatum.from_ind_trajectory(
                    valid_agent_ids, goals, episode, scenario, ref_frame, cfg)

                path_to_split = os.path.join(root_dir, split_type)
                with open(os.path.join(path_to_split, f"ma_datum_{count:06d}.dill"), "wb") as f:
                    dill.dump(datum, f)
                count += 1


def prepare_dt(scenario_name, samples_per_trajectory=10):
    scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
    episodes = scenario.load_episodes()
    dataset_split = json.load(open(os.path.join(get_core_dir(), "dataset_split.json")))
    feature_extractor = FeatureExtractor(scenario.lanelet_map)

    samples_list = []

    # Used for outputting incremental files for PRECOG processing
    for episode_idx, episode in enumerate(episodes):
        cfg = InDConfig(scenario, episode.recordings_meta, draw_map=True)

        split_type = dict_list_contains(dataset_split[scenario_name], episode_idx)
        if split_type != "test": continue

        print('\nEpisode {}/{} - {}'.format(episode_idx + 1, len(episodes), split_type))

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

        # Retrieve timesteps valid for PRECOG
        sample_step = int(cfg.sample_frequency * cfg.frame_rate)
        for initial_frame in tqdm.tqdm(range(0, len(episode.frames), sample_step), "Sample window"):
            final_frame = initial_frame + cfg.total_length

            #  Process all valid agents which have at least Tp past and Tf future in its trajectory
            # Only use agents in trimmed_trajectories to ensure only relevant vehicles are used

            for agent_id, trajectory in trimmed_trajectories.items():
                agent_initial_frame = trajectory[0].frame_id
                agent_final_frame = trajectory[-1].frame_id
                if agent_final_frame < final_frame or agent_initial_frame > initial_frame:
                    continue

                reachable_goals_list = []
                rel_initial_frame = initial_frame - trajectory[0].frame_id
                rel_final_frame = final_frame - trajectory[0].frame_id

                # get reachable goals at each timestep
                for idx in range(rel_initial_frame, rel_final_frame):
                    goal_routes = feature_extractor.get_goal_routes(trajectory[idx],
                                                                    scenario.config.goals)
                    if len([r for r in goal_routes if r is not None]) > 1:
                        reachable_goals_list.append(goal_routes)
                    else:
                        break

                # iterate through "samples_per_trajectory" points
                true_goal_idx = goals[agent_id]
                if (len(reachable_goals_list) > samples_per_trajectory
                        and reachable_goals_list[0][true_goal_idx] is not None):

                    # get true goal
                    true_goal_loc = scenario.config.goals[true_goal_idx]
                    true_goal_route = reachable_goals_list[0][true_goal_idx]
                    true_goal_type = feature_extractor.goal_type(trajectory[rel_initial_frame],
                                                                 true_goal_loc, true_goal_route)

                    reachable_goals = reachable_goals_list[-1]
                    state = trajectory[min(rel_final_frame, len(trajectory)) - 1]
                    frames = episode.frames[trajectory[rel_initial_frame].frame_id:state.frame_id + 1]

                    # iterate through each goal for that point in time
                    for goal_idx, route in enumerate(reachable_goals):
                        if route is not None:
                            goal = scenario.config.goals[goal_idx]

                            features = feature_extractor.extract(agent_id, frames, goal, route)

                            sample = features.copy()
                            sample['agent_id'] = agent_id
                            sample['possible_goal'] = goal_idx
                            sample['true_goal'] = true_goal_idx
                            sample['true_goal_type'] = true_goal_type
                            sample['frame_id'] = state.frame_id
                            sample['initial_frame_id'] = trajectory[0].frame_id
                            sample['fraction_observed'] = (final_frame - initial_frame) / len(trajectory)
                            samples_list.append(sample)

        samples = pd.DataFrame(data=samples_list)
        samples.to_csv(get_data_dir() + '{}_e{}.csv'.format(scenario_name, episode_idx), index=False)


def main():
    parser = argparse.ArgumentParser(description='Process the dataset')
    parser.add_argument('--scenario', type=str, help='Name of scenario to process', default=None)
    parser.add_argument('--type', type=str, help="Name of goal prediction method", default="precog")
    args = parser.parse_args()

    precog_path = os.path.join(get_data_dir(), "precog")
    if not os.path.exists(precog_path):
        os.mkdir(precog_path)

    if args.scenario is None:
        scenarios = ['heckstrasse', 'bendplatz', 'frankenberg']
    else:
        scenarios = [args.scenario]

    for scenario_name in scenarios:
        scenario_path = os.path.join(precog_path, scenario_name)
        if not os.path.exists(scenario_path):
            os.mkdir(scenario_path)
        for split_str in ["train", "val", "test"]:
            set_path = os.path.join(scenario_path, split_str)
            if not os.path.exists(set_path):
                os.mkdir(set_path)

        print(f"Processing {len(scenarios)} scenarios for {args.type}.")
        print('Processing dataset for scenario: ' + scenario_name)
        if args.type == "precog":
            prepare_precog(scenario_name, scenario_path)
        elif args.type == "dt":
            prepare_dt(scenario_name)


if __name__ == '__main__':
    main()
