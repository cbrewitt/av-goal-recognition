from z3 import *

from decisiontree.decision_tree import Node, ThresholdDecision, BinaryDecision
from core.feature_extraction import FeatureExtractor
from decisiontree.dt_goal_recogniser import TrainedDecisionTrees


def add_tree(root, name, features, solver):
    likelihood = Real(name)

    def recurse(node: Node, parent_expr=True):

        if node.decision is None:
            # handle leaf node
            solver.add(Implies(parent_expr, likelihood == node.value))
            pass
        else:
            feature = features[node.decision.feature_name]
            if isinstance(node.decision, ThresholdDecision):
                true_child_expr = And(parent_expr, feature > node.decision.threshold)
            elif isinstance(node.decision, BinaryDecision):
                true_child_expr = And(parent_expr, feature)
            else:
                raise TypeError('invalid decision type')

            false_child_expr = And(parent_expr, Not(true_child_expr))

            recurse(node.decision.true_child, true_child_expr)
            recurse(node.decision.false_child, false_child_expr)

    recurse(root)
    return likelihood


def add_features(goal_name, suffix=''):
    goal_name = str(goal_name)
    features = {}
    feature_types = {'scalar': Real, 'binary': Bool}

    for feature_name, feature_type in FeatureExtractor.feature_names.items():
        features[feature_name] = feature_types[feature_type](feature_name + '_' + goal_name + suffix)
    return features


def add_goal_tree_model(reachable_goals, solver, model, suffix=''):
    probs = {}
    features = {}
    for goal_idx, goal_type in reachable_goals:
        prior = float(model.goal_priors.loc[(model.goal_priors.true_goal == goal_idx)
                                            & (model.goal_priors.true_goal_type == goal_type), 'prior'])
        goal_features = add_features(goal_idx, suffix)
        likelihood = add_tree(model.decision_trees[goal_idx][goal_type],
                              'likelihood_{}_{}{}'.format(goal_idx, goal_type, suffix), goal_features, solver)
        prob = likelihood * prior
        probs[goal_idx] = prob
        features[goal_idx] = goal_features

    # get normalised probabilities
    prob_sum = 0
    for prob in probs.values():
        prob_sum = prob_sum + prob

    probs_norm = {}
    for goal_idx, goal_type in reachable_goals:
        prob_norm = Real('prob_{}_{}{}'.format(goal_idx, goal_type, suffix))
        probs_norm[goal_idx] = prob_norm
        solver.add(prob_norm == probs[goal_idx] / prob_sum)

    return features, probs_norm


def main():
    scenario_name = 'heckstrasse'
    model = TrainedDecisionTrees.load(scenario_name)
    reachable_goals = [(1, 'straight-on'), (2, 'turn-left')]

    s = Solver()

    features, probs = add_goal_tree_model(reachable_goals, s, model)

    # unsatisfiable if G2 always has highest prob
    verify_expr = Implies(And(features[1]['in_correct_lane'], Not(features[2]['in_correct_lane'])), probs[2] < probs[1])
    s.add(Not(verify_expr))

    print(s.check())
    try:
        print(s.model())
    except z3.z3types.Z3Exception:
        print('Successfully verified')


if __name__ == '__main__':
    main()
