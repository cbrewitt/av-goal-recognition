import pandas as pd

from av_goal_recognition.scenario import Scenario
from av_goal_recognition.base import get_data_dir, get_scenario_config_dir


def get_dataset(scenario_name, subset='train', features=True):
    scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
    episode_training_sets = []

    for episode_idx in range(len(scenario.config.episodes)):
        print('episode {}'.format(episode_idx))
        episode_training_set = pd.read_csv(
            get_data_dir() + '{}_e{}_{}.csv'.format(scenario_name, episode_idx, subset))
        print('training samples: {}'.format(episode_training_set.shape[0]))
        episode_training_set['episode'] = episode_idx
        episode_training_sets.append(episode_training_set)
    training_set = pd.concat(episode_training_sets)
    if features:
        return training_set
    else:
        unique_training_samples = training_set[['episode', 'agent_id', 'initial_frame_id', 'frame_id',
                                            'true_goal', 'true_goal_type', 'fraction_observed']
                                            ].drop_duplicates().reset_index()
        return unique_training_samples


def get_goal_priors(training_set, goal_types, alpha=0):
    agent_goals = training_set[['episode', 'agent_id', 'true_goal', 'true_goal_type']].drop_duplicates()
    print('training_vehicles: {}'.format(agent_goals.shape[0]))
    goal_counts = pd.DataFrame(data=[(x, t, 0) for x in range(len(goal_types)) for t in goal_types[x]],
                               columns=['true_goal', 'true_goal_type', 'goal_count'])
    goal_counts = goal_counts.set_index(['true_goal', 'true_goal_type'])
    goal_counts['goal_count'] += agent_goals.groupby(['true_goal', 'true_goal_type']).size()
    goal_counts = goal_counts.fillna(0)
    goal_priors = ((goal_counts.goal_count + alpha) / (agent_goals.shape[0] + alpha * goal_counts.shape[0])).rename('prior')
    goal_priors = goal_priors.reset_index()
    return goal_priors
