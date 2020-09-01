import numpy as np
import pandas as pd

from av_goal_recognition.base import get_scenario_config_dir, get_data_dir
from av_goal_recognition.data_processing import get_dataset
from av_goal_recognition.goal_recognition import PriorBaseline, HandcraftedGoalTrees, TrainedDecisionTrees
from av_goal_recognition.scenario import Scenario

scenario_name = 'heckstrasse'
scenario = Scenario.load(get_scenario_config_dir() + scenario_name + '.json')
print('loading episodes')
episodes = scenario.load_episodes()

models = {'prior_baseline': PriorBaseline,
          'handcrafted_trees': HandcraftedGoalTrees,
          'trained_trees': TrainedDecisionTrees}

dataset_names = ['train', 'test']

accuracies = pd.DataFrame(index=models.keys(), columns=dataset_names)
cross_entropies = pd.DataFrame(index=models.keys(), columns=dataset_names)

for dataset_name in dataset_names:
    dataset = get_dataset(scenario_name, dataset_name)
    predictions = {}
    num_goals = len(scenario.config.goals)
    targets = dataset.true_goal.to_numpy()

    for model_name, model in models.items():
        model = model.load(scenario_name)
        unique_samples = model.batch_goal_probabilities(dataset)
        unique_samples['model_correct'] = (unique_samples['model_prediction']
                                                  == unique_samples['true_goal'])
        cross_entropy = -np.mean(np.log(unique_samples.loc[
                                                    unique_samples.model_probs != 0, 'model_probs']))
        accuracy = unique_samples.model_correct.mean()

        # TODO: figure out a way of not having zeroed predictions - consider multiple possible current lanelet

        accuracies.loc[model_name, dataset_name] = accuracy
        cross_entropies.loc[model_name, dataset_name] = cross_entropy

        print('{} accuracy: {:.3f}'.format(model_name, accuracy))
        print('{} cross entropy: {:.3f}'.format(model_name, cross_entropy))

print('accuracy:')
print(accuracies)
print('cross entropy:')
print(cross_entropies)
