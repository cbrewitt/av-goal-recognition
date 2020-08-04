import pandas as pd
import matplotlib.pyplot as plt

from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.handcrafted_trees import scenario_trees
from av_goal_recognition.base import get_data_dir

trees = scenario_trees['heckstrasse']
episode_indices = [0, 1, 2]

episode_training_sets = []

for episode_idx in episode_indices:
    print('episode {}'.format(episode_idx))
    episode_training_set = pd.read_csv(get_data_dir() + 'heckstrasse_e{}_train.csv'.format(episode_idx))
    print('training samples: {}'.format(episode_training_set.shape[0]))
    episode_training_set['episode'] = episode_idx
    episode_training_sets.append(episode_training_set)

training_set = pd.concat(episode_training_sets)


# compute goal priors
agent_goals = training_set[['episode', 'agent_id', 'true_goal']].drop_duplicates()
print('training_vehicles: {}'.format(agent_goals.shape[0]))

goal_counts = agent_goals['true_goal'].value_counts()
goal_priors = (goal_counts / agent_goals.shape[0]).rename('prior')
print('goal priors:')
print(goal_priors)

tree_likelihoods = []

for index, row in training_set.iterrows():
    features = row[FeatureExtractor.feature_names]
    tree = trees[row['possible_goal']]
    tree_likelihood = tree.traverse(features)
    tree_likelihoods.append(tree_likelihood)

training_set['tree_likelihood'] = tree_likelihoods
training_set.to_csv(get_data_dir() + 'heckstrasse_train_predictions.csv', index=False)

# one for each agent state
unique_training_samples = training_set[['episode', 'agent_id', 'frame_id', 'true_goal', 'fraction_observed']].drop_duplicates()

print('unique training samples: {}'.format(unique_training_samples.shape[0]))

tree_predictions = []
prior_predictions = []

for index, row in unique_training_samples.iterrows():
    indices = ((training_set.episode == row.episode)
               & (training_set.agent_id == row.agent_id)
               & (training_set.frame_id == row.frame_id))
    goals = training_set.loc[indices][['possible_goal', 'tree_likelihood']]
    goals = goals.merge(goal_priors, 'left', left_on='possible_goal',
                        right_index=True)

    goals['tree_prob'] = goals['tree_likelihood'] * goals['prior']
    goals['tree_prob'] = goals['tree_prob'] / goals['tree_prob'].sum()
    idx = goals['tree_prob'].idxmax()
    tree_prediction = goals['possible_goal'].loc[idx]
    prior_prediction = goals['possible_goal'].loc[goals['prior'].idxmax()]
    tree_predictions.append(tree_prediction)
    prior_predictions.append(prior_prediction)

unique_training_samples['tree_prediction'] = tree_predictions
unique_training_samples['prior_prediction'] = prior_predictions

unique_training_samples['tree_correct'] = (unique_training_samples['tree_prediction']
                                           == unique_training_samples['true_goal'])
unique_training_samples['prior_correct'] = (unique_training_samples['prior_prediction']
                                            == unique_training_samples['true_goal'])

tree_correct = unique_training_samples.tree_correct.sum()
prior_correct = unique_training_samples.prior_correct.sum()
total_samples = unique_training_samples.shape[0]

print('correct prediction from tree: {}/{} = {}'.format(
      tree_correct, total_samples, tree_correct / total_samples))

print('correct prediction from prior: {}/{} = {}'.format(
      prior_correct, total_samples, prior_correct / total_samples))

# TODO plot evolution of goal prob for agents over time
# agent_0_samples = unique_training_samples.loc[unique_training_samples.agent_id == 8]
# print(agent_0_samples)

# plot at different fractions observed
print()
results_over_fraction_observed = unique_training_samples[
          ['fraction_observed', 'tree_correct', 'prior_correct']].groupby(
            'fraction_observed').mean()
print(results_over_fraction_observed)
results_over_fraction_observed.rename(columns={'tree_correct': 'decision tree',
                                               'prior_correct': 'prior only'}).plot()
plt.ylim([0, 1])
plt.xlabel('fraction of trajectory observed')
#plt.grid('on')
plt.title('Model comparison')
plt.ylabel('Accuracy')
plt.show()

# find examples where prediction became worse over time
for agent_id in unique_training_samples.agent_id.unique():
    agent_samples = unique_training_samples[unique_training_samples.agent_id == agent_id]
    prev_correct = False
    for index, row in agent_samples.iterrows():
        if row['tree_correct'] and not prev_correct and row['fraction_observed'] == 0.2:
            print(row)
        prev_correct = row['tree_correct']

