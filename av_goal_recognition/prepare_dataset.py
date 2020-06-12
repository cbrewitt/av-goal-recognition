import matplotlib.pyplot as plt
import pandas as pd

from av_goal_recognition.scenario import Scenario
from av_goal_recognition.feature_extraction import GoalDetector, FeatureExtractor
from av_goal_recognition.base import get_data_dir, get_scenario_config_dir

FPS = 25
sample_freq = 1
training_set_fraction = 0.8
samples_per_trajectory = 10


scenario = Scenario.load(get_scenario_config_dir() + 'heckstrasse.json')
# use only episode 0 for now
episodes = scenario.load_episodes()

feature_extractor = FeatureExtractor(scenario.lanelet_map)

training_samples_list = []
test_samples_list = []


for episode_idx, episode in enumerate(episodes):
    print('episode {}/{}'.format(episode_idx, len(episodes) - 1))
    num_frames = len(episode.frames)

    training_set_cutoff_frame = int(num_frames * training_set_fraction)

    goals = {}  # key: agent id, value: goal idx
    trimmed_trajectories = {}

    # detect goal, and trim trajectory past the goal
    #
    goal_detector = GoalDetector(scenario.config.goals)
    for agent_id, agent in episode.agents.items():

        agent_goals, goal_frames = goal_detector.detect_goals(agent.state_history)
        if len(agent_goals) > 0:
            first_goal_frame_idx = goal_frames[0] - agent.initial_frame
            trimmed_trajectory = agent.state_history[0:first_goal_frame_idx]
            goals[agent_id] = agent_goals[0]
            trimmed_trajectories[agent_id] = trimmed_trajectory

            # plot the trajectory
            # scenario.plot()
            # x = [state.x for state in trimmed_trajectory]
            # y = [state.y for state in trimmed_trajectory]
            # plt.plot(x, y, color='yellow')
            # print(agent_goals[0])
            # plt.show()

    #get features and reachable goals

    for agent_id, trajectory in trimmed_trajectories.items():
        lanelet_sequence = feature_extractor.get_lanelet_sequence(trajectory)
        agent = episode.agents[agent_id]

        print('agent_id {}/{}'.format(agent_id, len(trimmed_trajectories)-1))
        # iterate through each sampled point in time for trajectory

        reachable_goals_list = []

        # get reachable goals at each timestep
        for idx in range(0, len(trajectory)):
            reachable_goals = feature_extractor.reachable_goals(lanelet_sequence[idx],
                                                                scenario.config.goals)
            if len(reachable_goals) > 1:
                reachable_goals_list.append(reachable_goals)
            else:
                break

        # iterate through "samples_per_trajectory" points
        if len(reachable_goals_list) >= samples_per_trajectory:
            for idx in range(0, len(reachable_goals_list), len(reachable_goals_list) // samples_per_trajectory):
                reachable_goals = reachable_goals_list[idx]
                state = trajectory[idx]
                frames = episode.frames[trajectory[0].frame_id:state.frame_id + 1]

                # iterate through each goal for that point in time
                for goal_idx, route in reachable_goals.items():
                    goal = scenario.config.goals[goal_idx]
                    features = feature_extractor.extract(agent_id, frames, goal, route)

                    sample = features.copy()
                    sample['agent_id'] = agent_id
                    sample['possible_goal'] = goal_idx
                    sample['true_goal'] = goals[agent_id]
                    sample['frame_id'] = state.frame_id
                    sample['fraction_oberserved'] = idx / (len(reachable_goals) - 1)

                    if trajectory[-1].frame_id <= training_set_cutoff_frame:
                        training_samples_list.append(sample)
                    elif trajectory[0].frame_id > training_set_cutoff_frame:
                        test_samples_list.append(sample)

    training_samples = pd.DataFrame(data=training_samples_list)
    test_samples = pd.DataFrame(data=test_samples_list)

    training_samples.to_csv(get_data_dir() + 'heckstrasse_e{}_train.csv'.format(episode_idx), index=False)
    test_samples.to_csv(get_data_dir() + 'heckstrasse_e{}_test.csv'.format(episode_idx), index=False)
