import matplotlib.pyplot as plt
import pandas as pd

from av_goal_recognition.scenario import Scenario
from av_goal_recognition.feature_extraction import GoalDetector, FeatureExtractor

scenario = Scenario.load('../scenario_config/heckstrasse.json')
# use only episode 0 for now
episodes = scenario.load_episodes()
episode = episodes[0]

FPS = 25
sample_freq = 1
num_frames = len(episode.frames)
training_set_fraction = 0.8
training_set_cutoff_frame = int(num_frames * training_set_fraction)

goals = {}  # key: agent id, value: goal idx
trimmed_trajectories = {}

# detect goal, and trim trajectory past the goal
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


# divide into samples (1s apart), get features and reachable goals
feature_extractor = FeatureExtractor(scenario.lanelet_map)

training_samples_list = []
test_samples_list = []


for agent_id, trajectory in trimmed_trajectories.items():
    resampled_trajectory = [trajectory[idx] for idx in range(0, len(trajectory), FPS // sample_freq)]
    lanelet_sequence = feature_extractor.get_lanelet_sequence(trajectory)
    agent = episode.agents[agent_id]

    print('agent_id {}/{}'.format(agent_id, len(trimmed_trajectories)))
    # iterate through each sampled point in time for trajectory
    for idx in range(0, len(trajectory), FPS // sample_freq):
        print('trajectory idx {}'.format(idx))
        reachable_goals = feature_extractor.reachable_goals(lanelet_sequence[idx],
                                                            scenario.config.goals)
        state = trajectory[idx]
        frames = episode.frames[trajectory[0].frame_id:state.frame_id + 1]

        # iterate through each goal for that point in time
        for goal_idx, route in reachable_goals.items():
            print('goal idx {}'.format(goal_idx))
            goal = scenario.config.goals[goal_idx]
            features = feature_extractor.extract(agent_id, frames, goal, route)

            # TODO add to dataframe: agent_id, time, fraction of traj, features, true_goal
            sample = features.copy()
            sample['agent_id'] = agent_id
            sample['possible_goal'] = goal_idx
            sample['true_goal'] = goals[agent_id]
            sample['frame_id'] = state.frame_id
            sample['fraction_oberserved'] = idx / (len(trajectory) - 1)

            if trajectory[-1].frame_id <= training_set_cutoff_frame:
                training_samples_list.append(sample)
            elif trajectory[0].frame_id > training_set_cutoff_frame:
                test_samples_list.append(sample)


training_samples = pd.DataFrame(data=training_samples_list)
test_samples = pd.DataFrame(data=test_samples_list)

training_samples.to_csv('heckstrasse_e0_train.csv', index=False)
test_samples.to_csv('heckstrasse_e0_test.csv', index=False)

