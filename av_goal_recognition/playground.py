import argparse
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

from av_goal_recognition.base import get_scenario_config_dir
from av_goal_recognition.data_processing import get_dataset, get_goal_priors
from av_goal_recognition.decision_tree import Node
from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.goal_recognition import DecisionTreeGoalRecogniser
from av_goal_recognition.scenario import Scenario

scenario_name = 'heckstrasse'
alpha = 1
# model = DecisionTreeGoalRecogniser.train_grid_search(scenario_name)
# model.save(scenario_name)


scenario = Scenario.load(get_scenario_config_dir() + scenario_name + '.json')
training_set = get_dataset(scenario_name, subset='train')
validation_set = get_dataset(scenario_name, subset='valid')
goal_priors = get_goal_priors(training_set, scenario.config.goal_types, alpha=alpha)

goal_idx = 1
goal_types = goal_priors.loc[goal_priors.true_goal == goal_idx].true_goal_type.unique()
goal_type = 'straight-on'
dt_training_set = training_set.loc[(training_set.possible_goal == goal_idx)
                                   & (training_set.goal_type == goal_type)]
dt_validation_set = validation_set.loc[(validation_set.possible_goal == goal_idx)
                                       & (validation_set.goal_type == goal_type)]

X_train = dt_training_set[FeatureExtractor.feature_names.keys()].to_numpy()
y_train = (dt_training_set.possible_goal == dt_training_set.true_goal).to_numpy()
train_prior = ((dt_training_set.true_goal == goal_idx).sum() + alpha) / (dt_training_set.shape[0] + 2 * alpha)

# do grid search in here
X_valid = dt_validation_set[FeatureExtractor.feature_names.keys()].to_numpy()
y_valid = (dt_validation_set.possible_goal == dt_validation_set.true_goal).to_numpy()

#max_leaf_nodes_grid = np.unique(np.round(np.logspace(0.3, 3, 30))).astype(int)
ccp_alpha_grid = np.logspace(-4,0)
criterion_grid = ['gini', 'entropy']
# np.unique(np.round(np.logspace(0.3, 3, 30))).astype(int)

accuracies = []
acc_sem = []

cross_entropies = []
cross_entropy_sems = []

best_tree = None
best_accuracy = 0
best_cross_entropy = np.inf
best_ccp_alpha = None

for ccp_alpha in ccp_alpha_grid:

        clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha,
                                          min_samples_leaf=1,
                                          max_depth=None,
                                          class_weight='balanced',
                                          criterion='entropy')
        clf = clf.fit(X_train, y_train)

        goal_tree = Node.from_sklearn(clf, FeatureExtractor.feature_names)
        goal_tree.set_values(dt_training_set, goal_idx, alpha=alpha)

        model_probs = []

        for index, row in dt_validation_set.iterrows():
            features = row[FeatureExtractor.feature_names]
            model_likelihood = goal_tree.traverse(features)
            model_prob = model_likelihood * train_prior / (model_likelihood * train_prior
                                                           + (1 - model_likelihood) * (1 - train_prior))
            model_probs.append(model_prob)

        model_probs = np.array(model_probs)
        valid_pred = model_probs > 0.5
        accuracy = np.mean(valid_pred == y_valid)
        std_err = np.std(valid_pred == y_valid) / np.sqrt(valid_pred.shape[0])

        # if np.any(valid_pred == 0) or np.any(valid_pred == 1):
        #     raise Exception('pred should not be 0 or 1')

        sample_cross_entropy = -(y_valid * np.log(model_probs) + (1 - y_valid) * np.log(1 - model_probs))
        cross_entropy_sem = np.std(sample_cross_entropy) / np.sqrt(sample_cross_entropy.shape[0])
        mean_cross_entropy = np.mean(sample_cross_entropy)

        accuracies.append(accuracy)
        cross_entropies.append(mean_cross_entropy)
        acc_sem.append(std_err)
        cross_entropy_sems.append(cross_entropy_sem)

        if accuracy > best_accuracy:
            best_cross_entropy = mean_cross_entropy
            best_ccp_alpha = ccp_alpha
            best_accuracy = accuracy
            best_tree = goal_tree

accuracies = np.array(accuracies)
cross_entropies = np.array(cross_entropies)
best_tree.pydot_tree().write_png('test.png')
print('{} training samples'.format(X_train.shape[0]))
print('best accuracy: {}'.format(best_accuracy))
print('best cross entropy: {}'.format(best_cross_entropy))
print('best ccp alpha: {}'.format(best_ccp_alpha))

plt.semilogx(ccp_alpha_grid, accuracies)
plt.fill_between(ccp_alpha_grid, accuracies + acc_sem, accuracies - acc_sem, alpha=0.2)
plt.title('accuracy')

plt.figure()
plt.semilogx(ccp_alpha_grid, cross_entropies)
plt.fill_between(ccp_alpha_grid, cross_entropies + cross_entropy_sems,
                 cross_entropies - cross_entropy_sems, alpha=0.2)
plt.title('cross entropy loss')
plt.show()
