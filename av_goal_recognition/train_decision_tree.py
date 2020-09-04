import pickle

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import _tree

from av_goal_recognition.base import get_img_dir, get_data_dir, get_scenario_config_dir
from av_goal_recognition.data_processing import get_dataset, get_goal_priors
from av_goal_recognition.decision_tree import Node
from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.scenario import Scenario

alpha = 1
scenario_name = 'heckstrasse'
scenario = Scenario.load(get_scenario_config_dir() + scenario_name + '.json')
training_set = get_dataset(scenario_name)
goal_priors = get_goal_priors(training_set, scenario.config.goal_types, alpha=alpha)
goal_priors.to_csv(get_data_dir() + '{}_priors.csv'.format('heckstrasse'), index=False)
print(training_set.columns)

decision_trees = {}

for goal_idx in goal_priors.true_goal.unique():
    decision_trees[goal_idx] = {}
    goal_types = goal_priors.loc[goal_priors.true_goal==goal_idx].true_goal_type.unique()
    for goal_type in goal_types:
        dt_training_set = training_set.loc[(training_set.possible_goal == goal_idx)
                                           & (training_set.goal_type == goal_type)]
        X = dt_training_set[FeatureExtractor.feature_names.keys()].to_numpy()
        y = (dt_training_set.possible_goal == dt_training_set.true_goal).to_numpy()
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=7, min_samples_leaf=1,
                                          class_weight='balanced')
        clf = clf.fit(X, y)
        feature_names = [*FeatureExtractor.feature_names]
        goal_tree = Node.from_sklearn(clf, FeatureExtractor.feature_names)
        goal_tree.set_values(dt_training_set, goal_idx, alpha=alpha)
        decision_trees[goal_idx][goal_type] = goal_tree
        pydot_tree = goal_tree.pydot_tree()
        pydot_tree.write_png(get_img_dir() + 'trained_tree_{}_G{}_{}.png'.format(
            scenario_name, goal_idx, goal_type))
        # tree.plot_tree(clf, feature_names=feature_names)
        # plt.title('goal {}, {}'.format(goal_idx, goal_type))
        #
        # plt.show()

with open(get_data_dir() + 'trained_trees_{}.p'.format(scenario_name), 'wb') as f:
    pickle.dump(decision_trees, f)

