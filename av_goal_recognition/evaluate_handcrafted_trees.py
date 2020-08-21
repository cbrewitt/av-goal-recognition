import pickle

import pandas as pd
import matplotlib.pyplot as plt

from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.handcrafted_trees import scenario_trees
from av_goal_recognition.base import get_data_dir, get_scenario_config_dir
from av_goal_recognition.scenario import Scenario
from av_goal_recognition.data_processing import get_dataset, get_goal_priors

plt.style.use('ggplot')

scenario_name = 'heckstrasse'
handcrafted_trees = scenario_trees[scenario_name]
with open(get_data_dir() + 'trained_trees_{}.p'.format(scenario_name), 'rb') as f:
    trained_trees = pickle.load(f)

training_set = get_dataset(scenario_name, 'train')

goal_priors = get_goal_priors(training_set)

dataset = get_dataset(scenario_name, 'test')

print('goal priors:')
print(goal_priors)

handcrafted_tree_likelihoods = []
trained_tree_likelihoods = []

for index, row in dataset.iterrows():
    features = row[FeatureExtractor.feature_names]

    handcrafted_tree = handcrafted_trees[row['possible_goal']][row['goal_type']]
    handcrafted_tree_likelihood = handcrafted_tree.traverse(features)
    handcrafted_tree_likelihoods.append(handcrafted_tree_likelihood)

    if row['goal_type'] in trained_trees[row['possible_goal']]:
        trained_tree = trained_trees[row['possible_goal']][row['goal_type']]
        trained_tree_likelihood = trained_tree.traverse(features)
    else:
        trained_tree_likelihood = 0

    trained_tree_likelihoods.append(trained_tree_likelihood)

dataset['handcrafted_tree_likelihood'] = handcrafted_tree_likelihoods
dataset['trained_tree_likelihood'] = trained_tree_likelihoods
dataset.to_csv(get_data_dir() + 'heckstrasse_train_predictions.csv', index=False)

# one for each agent state
unique_samples = dataset[['episode', 'agent_id', 'frame_id', 'true_goal',
                          'true_goal_type', 'fraction_observed']].drop_duplicates()

print('unique training samples: {}'.format(unique_samples.shape[0]))

handcrafted_tree_predictions = []
trained_tree_predictions = []
prior_predictions = []

for index, row in unique_samples.iterrows():
    indices = ((dataset.episode == row.episode)
               & (dataset.agent_id == row.agent_id)
               & (dataset.frame_id == row.frame_id))
    goals = dataset.loc[indices][['possible_goal', 'goal_type', 'handcrafted_tree_likelihood',
                                       'trained_tree_likelihood']]
    goals = goals.merge(goal_priors, 'left', left_on=['possible_goal', 'goal_type'],
                        right_on=['true_goal', 'true_goal_type'])

    goals['handcrafted_tree_prob'] = goals['handcrafted_tree_likelihood'] * goals['prior']
    goals['handcrafted_tree_prob'] = goals['handcrafted_tree_prob'] / goals['handcrafted_tree_prob'].sum()
    idx = goals['handcrafted_tree_prob'].idxmax()
    handcrafted_tree_prediction = goals['possible_goal'].loc[idx]
    handcrafted_tree_predictions.append(handcrafted_tree_prediction)

    goals['trained_tree_prob'] = goals['trained_tree_likelihood'] * goals['prior']
    goals['trained_tree_prob'] = goals['trained_tree_prob'] / goals['trained_tree_prob'].sum()
    idx = goals['trained_tree_prob'].idxmax()
    trained_tree_prediction = goals['possible_goal'].loc[idx]
    trained_tree_predictions.append(trained_tree_prediction)

    prior_prediction = goals['possible_goal'].loc[goals['prior'].idxmax()]
    prior_predictions.append(prior_prediction)

unique_samples['handcrafted_tree_prediction'] = handcrafted_tree_predictions
unique_samples['handcrafted_tree_correct'] = (unique_samples['handcrafted_tree_prediction']
                                              == unique_samples['true_goal'])

unique_samples['trained_tree_prediction'] = trained_tree_predictions
unique_samples['trained_tree_correct'] = (unique_samples['trained_tree_prediction']
                                          == unique_samples['true_goal'])

unique_samples['prior_prediction'] = prior_predictions
unique_samples['prior_correct'] = (unique_samples['prior_prediction']
                                   == unique_samples['true_goal'])

handcrafted_tree_correct = unique_samples.handcrafted_tree_correct.sum()
trained_tree_correct = unique_samples.trained_tree_correct.sum()
prior_correct = unique_samples.prior_correct.sum()
total_samples = unique_samples.shape[0]

print('correct prediction from handcrafted tree: {}/{} = {}'.format(
      handcrafted_tree_correct, total_samples, handcrafted_tree_correct / total_samples))

print('correct prediction from trained tree: {}/{} = {}'.format(
      trained_tree_correct, total_samples, trained_tree_correct / total_samples))

print('correct prediction from prior: {}/{} = {}'.format(
      prior_correct, total_samples, prior_correct / total_samples))


# plot at different fractions observed
print()
results_over_fraction_observed = unique_samples[
          ['fraction_observed', 'handcrafted_tree_correct', 'trained_tree_correct',
           'prior_correct']].groupby('fraction_observed').mean()
print(results_over_fraction_observed)

results_over_fraction_observed.rename(columns={'handcrafted_tree_correct': 'handcrafted decision tree',
                                               'trained_tree_correct': 'trained decision tree',
                                               'prior_correct': 'prior only'}).plot()
plt.ylim([0, 1])
plt.xlabel('fraction of trajectory observed')
plt.title('Model comparison')
plt.ylabel('Accuracy')
plt.show()

