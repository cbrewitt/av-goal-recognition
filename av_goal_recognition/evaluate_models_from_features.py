import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
entropies = pd.DataFrame(index=models.keys(), columns=dataset_names)
norm_entropies = pd.DataFrame(index=models.keys(), columns=dataset_names)
avg_max_prob = pd.DataFrame(index=models.keys(), columns=dataset_names)
avg_min_prob = pd.DataFrame(index=models.keys(), columns=dataset_names)

predictions = {}

for dataset_name in dataset_names:
    dataset = get_dataset(scenario_name, dataset_name)
    dataset_predictions = {}
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
        entropies.loc[model_name, dataset_name] = unique_samples.model_entropy.mean()
        norm_entropies.loc[model_name, dataset_name] = unique_samples.model_entropy_norm.mean()
        avg_max_prob.loc[model_name, dataset_name] = unique_samples.max_probs.mean()
        avg_min_prob.loc[model_name, dataset_name] = unique_samples.min_probs.mean()
        dataset_predictions[model_name] = unique_samples
        print('{} accuracy: {:.3f}'.format(model_name, accuracy))
        print('{} cross entropy: {:.3f}'.format(model_name, cross_entropy))

    predictions[dataset_name] = dataset_predictions

print('accuracy:')
print(accuracies)
print('\ncross entropy:')
print(cross_entropies)
print('\nentropy:')
print(entropies)
print('\nnormalised entropy:')
print(norm_entropies)
print('\naverage max probability:')
print(avg_max_prob)
print('\naverage min probability:')
print(avg_min_prob)


for dataset_name in dataset_names:

    fig, ax = plt.subplots()
    for model_name, model in models.items():
        unique_samples = predictions[dataset_name][model_name]
        accuracy = unique_samples[['model_correct', 'fraction_observed']].groupby('fraction_observed').mean()
        accuracy.rename(columns={'model_correct': model_name}).plot(ax=ax)
    plt.xlabel('fraction of trajectory observed')
    plt.title('Accuracy ({})'.format(dataset_name))
    plt.ylim([0, 1])
    plt.show()

    fig, ax = plt.subplots()
    for model_name, model in models.items():
        unique_samples = predictions[dataset_name][model_name]
        entropy_norm = unique_samples[['model_entropy', 'fraction_observed']].groupby('fraction_observed').mean()
        entropy_norm.rename(columns={'model_entropy': model_name}).plot(ax=ax)
    plt.xlabel('fraction of trajectory observed')
    plt.title('Normalised Entropy ({})'.format(dataset_name))
    plt.ylim([0,1])
    plt.show()