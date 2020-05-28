from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from av_goal_recognition.goal_recognition import Scenario
from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.lanelet_helpers import LaneletHelpers


def main():
    scenario = Scenario.load('../scenario_config/heckstrasse.json')
    # extract features from agent 0

    episode = scenario.episodes[0]
    agent_id = 73
    agent = episode.agents[agent_id]
    frames = episode.frames[agent.initial_frame:agent.final_frame+1]
    feature_extractor = FeatureExtractor(scenario.lanelet_map)
    lanelet_sequence = feature_extractor.get_lanelet_sequence(agent.state_history)
    prev_ll = None
    scenario.plot()
    agent.plot_trajectory('y')

    for idx, ll in enumerate(lanelet_sequence):
        if ll != prev_ll:
            prev_ll = ll
            LaneletHelpers.plot(ll)
    plt.show()

    all_features = defaultdict(dict)

    for x in range(agent.num_frames):
        print('frame: {}'.format(x))
        reachable_goals = feature_extractor.reachable_goals(lanelet_sequence[x],
                                                            scenario.config.goals)
        for goal_idx, route in reachable_goals.items():
            goal = scenario.config.goals[goal_idx]
            features = feature_extractor.extract(agent_id, frames[:x+1], goal, route)
            all_features[goal_idx][x] = features
            print('goal {}'.format(goal_idx))
            print(features)

    FPS = 25

    # plot features
    for goal_idx, feature_dict in all_features.items():
        timesteps = list(feature_dict.keys())
        if len(feature_dict) > 0:
            feature_names = next(iter(feature_dict.values())).keys()
            plt.subplot(len(feature_names), 1, 1)
            plt.title('Goal {}'.format(goal_idx))
            for feature_idx, feature_name in enumerate(feature_names):
                feature_values = [v[feature_name] for v in feature_dict.values()]
                plt.subplot(len(feature_names), 1, feature_idx+1)
                plt.plot(np.array(timesteps)/FPS, feature_values)
                plt.legend([feature_name])
            plt.xlabel('time (s)')
            plt.show()


if __name__ == '__main__':
    main()
