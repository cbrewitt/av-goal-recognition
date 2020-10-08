import pickle
from sklearn import tree
import numpy as np
import pandas as pd

from av_goal_recognition.base import get_data_dir, get_scenario_config_dir, get_img_dir
from av_goal_recognition.data_processing import get_goal_priors, get_dataset
from av_goal_recognition.decision_tree import Node
from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.handcrafted_trees import scenario_trees
from av_goal_recognition.scenario import Scenario
from av_goal_recognition.metrics import entropy


class BayesianGoalRecogniser:

    def __init__(self, goal_priors, scenario):
        self.goal_priors = goal_priors
        self.feature_extractor = FeatureExtractor(scenario.lanelet_map)
        self.scenario = scenario

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
        raise NotImplementedError

    def goal_likelihood_from_features(self, features, goal_type, goal):
        raise NotImplementedError

    def goal_probabilities(self, frames, agent_id):
        state_history = [f.agents[agent_id] for f in frames]
        current_state = state_history[-1]
        lanelet_sequence = self.feature_extractor.get_lanelet_sequence(state_history)
        current_lanelet = lanelet_sequence[-1]

        goal_probs = []
        for goal_idx in range(len(self.scenario.config.goals)):

            goal_loc = self.scenario.config.goals[goal_idx]
            route = self.feature_extractor.route_to_goal(current_lanelet, goal_loc)
            if route is None:
                goal_prob = 0
            else:
                # get un-normalised "probability"
                prior = self.get_goal_prior(goal_idx, current_state, route)
                if prior == 0:
                    goal_prob = 0
                else:
                    likelihood = self.goal_likelihood(goal_idx, frames, route, agent_id)
                    goal_prob = likelihood * prior
            goal_probs.append(goal_prob)
        goal_probs = np.array(goal_probs)
        goal_probs = goal_probs / goal_probs.sum()
        return goal_probs

    def batch_goal_probabilities(self, dataset):
        dataset = dataset.copy()
        model_likelihoods = []

        for index, row in dataset.iterrows():
            features = row[FeatureExtractor.feature_names]
            goal_type = row['goal_type']
            goal = row['possible_goal']
            model_likelihood = self.goal_likelihood_from_features(features, goal_type, goal)

            model_likelihoods.append(model_likelihood)
        dataset['model_likelihood'] = model_likelihoods
        unique_samples = dataset[['episode', 'agent_id', 'frame_id', 'true_goal',
                                  'true_goal_type', 'fraction_observed']].drop_duplicates()
        model_predictions = []
        model_probs = []
        min_probs = []
        max_probs = []
        model_entropys = []
        model_norm_entropys = []
        for index, row in unique_samples.iterrows():
            indices = ((dataset.episode == row.episode)
                       & (dataset.agent_id == row.agent_id)
                       & (dataset.frame_id == row.frame_id))
            goals = dataset.loc[indices][['possible_goal', 'goal_type', 'model_likelihood']]
            goals = goals.merge(self.goal_priors, 'left', left_on=['possible_goal', 'goal_type'],
                                right_on=['true_goal', 'true_goal_type'])
            goals['model_prob'] = goals.model_likelihood * goals.prior
            goals['model_prob'] = goals.model_prob / goals.model_prob.sum()
            idx = goals['model_prob'].idxmax()

            goal_prob_entropy = entropy(goals.model_prob)
            uniform_entropy = entropy(np.ones(goals.model_prob.shape[0])
                                      / goals.model_prob.shape[0])
            norm_entropy = goal_prob_entropy / uniform_entropy
            model_prediction = goals['possible_goal'].loc[idx]
            model_predictions.append(model_prediction)
            model_prob = goals['model_prob'].loc[idx]
            max_prob = goals.model_prob.max()
            min_prob = goals.model_prob.min()
            max_probs.append(max_prob)
            min_probs.append(min_prob)
            model_probs.append(model_prob)
            model_entropys.append(goal_prob_entropy)
            model_norm_entropys.append(norm_entropy)

        unique_samples['model_prediction'] = model_predictions
        unique_samples['model_probs'] = model_probs
        unique_samples['max_probs'] = max_probs
        unique_samples['min_probs'] = min_probs
        unique_samples['model_entropy'] = model_entropys
        unique_samples['model_entropy_norm'] = model_norm_entropys
        return unique_samples

    @classmethod
    def load(cls, scenario_name):
        priors = cls.load_priors(scenario_name)
        scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
        return cls(priors, scenario)

    @staticmethod
    def load_priors(scenario_name):
        return pd.read_csv(get_data_dir() + scenario_name + '_priors.csv')

    def get_goal_prior(self, goal_idx, state, route):
        goal_loc = self.scenario.config.goals[goal_idx]
        goal_type = self.feature_extractor.goal_type(state, goal_loc, route)
        prior_series = self.goal_priors.loc[(self.goal_priors.true_goal == goal_idx) & (self.goal_priors.true_goal_type == goal_type)].prior
        if prior_series.shape[0] == 0:
            return 0
        else:
            return float(prior_series)


class PriorBaseline(BayesianGoalRecogniser):

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
        return 0.5

    def goal_likelihood_from_features(self, features, goal_type, goal):
        return 0.5


class DecisionTreeGoalRecogniser(BayesianGoalRecogniser):

    def __init__(self, goal_priors, scenario, decision_trees):
        super().__init__(goal_priors, scenario)
        self.decision_trees = decision_trees

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
        goal_loc = self.scenario.config.goals[goal_idx]
        features = self.feature_extractor.extract(agent_id, frames, goal_loc, route)
        likelihood = self.decision_trees[goal_idx][features['goal_type']].traverse(features)
        return likelihood

    def goal_likelihood_from_features(self, features, goal_type, goal):
        if goal_type in self.decision_trees[goal]:
            tree = self.decision_trees[goal][goal_type]
            tree_likelihood = tree.traverse(features)
        else:
            tree_likelihood = 0.5
        return tree_likelihood

    @classmethod
    def load(cls, scenario_name):
        priors = cls.load_priors(scenario_name)
        scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
        decision_trees = cls.load_decision_trees(scenario_name)
        return cls(priors, scenario, decision_trees)

    @staticmethod
    def load_decision_trees(scenario_name):
        raise NotImplementedError

    @classmethod
    def train(cls, scenario_name, alpha=1, ccp_alpha=0, criterion='gini', min_samples_leaf=1,
              max_leaf_nodes=None, max_depth=None, training_set=None):
        decision_trees = {}
        scenario = Scenario.load(get_scenario_config_dir() + scenario_name + '.json')
        if training_set is None:
            training_set = get_dataset(scenario_name, subset='train')
        goal_priors = get_goal_priors(training_set, scenario.config.goal_types, alpha=alpha)

        for goal_idx in goal_priors.true_goal.unique():
            decision_trees[goal_idx] = {}
            goal_types = goal_priors.loc[goal_priors.true_goal == goal_idx].true_goal_type.unique()
            for goal_type in goal_types:
                dt_training_set = training_set.loc[(training_set.possible_goal == goal_idx)
                                                   & (training_set.goal_type == goal_type)]
                if dt_training_set.shape[0] > 0:
                    X = dt_training_set[FeatureExtractor.feature_names.keys()].to_numpy()
                    y = (dt_training_set.possible_goal == dt_training_set.true_goal).to_numpy()
                    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf, max_depth=max_depth, class_weight='balanced',
                        ccp_alpha=ccp_alpha, criterion=criterion)
                    clf = clf.fit(X, y)
                    goal_tree = Node.from_sklearn(clf, FeatureExtractor.feature_names)
                    goal_tree.set_values(dt_training_set, goal_idx, alpha=alpha)
                else:
                    goal_tree = Node(0.5)

                decision_trees[goal_idx][goal_type] = goal_tree
        return cls(goal_priors, scenario, decision_trees)

    def save(self, scenario_name):
        for goal_idx in self.goal_priors.true_goal.unique():
            goal_types = self.goal_priors.loc[self.goal_priors.true_goal == goal_idx].true_goal_type.unique()
            for goal_type in goal_types:
                goal_tree = self.decision_trees[goal_idx][goal_type]
                pydot_tree = goal_tree.pydot_tree()
                pydot_tree.write_png(get_img_dir() + 'trained_tree_{}_G{}_{}.png'.format(
                    scenario_name, goal_idx, goal_type))
        with open(get_data_dir() + 'trained_trees_{}.p'.format(scenario_name), 'wb') as f:
            pickle.dump(self.decision_trees, f)


class HandcraftedGoalTrees(DecisionTreeGoalRecogniser):

    @staticmethod
    def load_decision_trees(scenario_name):
        return scenario_trees[scenario_name]


class TrainedDecisionTrees(DecisionTreeGoalRecogniser):

    @staticmethod
    def load_decision_trees(scenario_name):
        with open(get_data_dir() + 'trained_trees_{}.p'.format(scenario_name), 'rb') as f:
            return pickle.load(f)

