import pickle

import numpy as np
import pandas as pd

from av_goal_recognition.base import get_data_dir, get_scenario_config_dir
from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.handcrafted_trees import scenario_trees
from av_goal_recognition.scenario import Scenario


class BayesianGoalRecogniser:

    def __init__(self, goal_priors, scenario):
        self.goal_priors = goal_priors
        self.feature_extractor = FeatureExtractor(scenario.lanelet_map)
        self.scenario = scenario

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
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
        pass

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


class DecisionTreeGoalRecogniser(BayesianGoalRecogniser):

    def __init__(self, goal_priors, scenario, decision_trees):
        super().__init__(goal_priors, scenario)
        self.decision_trees = decision_trees

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
        goal_loc = self.scenario.config.goals[goal_idx]
        features = self.feature_extractor.extract(agent_id, frames, goal_loc, route)
        likelihood = self.decision_trees[goal_idx][features['goal_type']].traverse(features)
        return likelihood

    @classmethod
    def load(cls, scenario_name):
        priors = cls.load_priors(scenario_name)
        scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
        decision_trees = cls.load_decision_trees(scenario_name)
        return cls(priors, scenario, decision_trees)

    @staticmethod
    def load_decision_trees(scenario_name):
        raise NotImplementedError


class HandcraftedGoalTrees(DecisionTreeGoalRecogniser):

    @staticmethod
    def load_decision_trees(scenario_name):
        return scenario_trees[scenario_name]


class TrainedDecisionTrees(DecisionTreeGoalRecogniser):

    @staticmethod
    def load_decision_trees(scenario_name):
        with open(get_data_dir() + 'trained_trees_{}.p'.format(scenario_name), 'rb') as f:
            return pickle.load(f)

