import argparse

import pandas as pd
import numpy as np

from av_goal_recognition.base import get_data_dir
from av_goal_recognition.data_processing import get_dataset
from av_goal_recognition.goal_recognition import TrainedDecisionTrees

parser = argparse.ArgumentParser(description='Hyperparameter grid search')
parser.add_argument('--criterion', type=str, help='Function to measure quality of split', default=None)
parser.add_argument('--save_name', type=str, help='results file name', default=None)
args = parser.parse_args()

scenarios = ['heckstrasse', 'bendplatz', 'frankenberg']

if args.criterion is None:
    criterions = ['gini', 'entropy']
else:
    criterions = [args.criterion]

ccp_alphas = np.logspace(-4, -1, 10)
alphas = np.logspace(-4, -1, 10)
min_samples_leafs = [1]

results_scenario = []
results_criterion = []
results_ccp_alpha = []
results_alpha = []
results_min_samples_leaf = []
results_cross_entropy = []
results_accuracy = []

idx = 0
num_iterations = (len(criterions) * len(ccp_alphas) * len(alphas)
                  * len(min_samples_leafs) * len(scenarios))

for scenario_name in scenarios:
    train_set = get_dataset(scenario_name, 'train')
    valid_set = get_dataset(scenario_name, 'valid')
    for criterion in criterions:
        for ccp_alpha in ccp_alphas:
            for alpha in alphas:
                for min_samples_leaf in min_samples_leafs:
                    idx += 1
                    print('iteration {}/{}'.format(idx, num_iterations))
                    model = TrainedDecisionTrees.train(scenario_name, alpha, ccp_alpha, criterion,
                                                       min_samples_leaf, training_set=train_set)
                    unique_samples = model.batch_goal_probabilities(valid_set)
                    unique_samples['model_correct'] = (unique_samples['model_prediction']
                                                       == unique_samples['true_goal'])
                    cross_entropy = -np.mean(np.log(unique_samples.loc[
                                                        unique_samples.model_probs != 0, 'model_probs']))
                    accuracy = unique_samples.model_correct.mean()

                    results_scenario.append(scenario_name)
                    results_criterion.append(criterion)
                    results_ccp_alpha.append(ccp_alpha)
                    results_alpha.append(alpha)
                    results_min_samples_leaf.append(min_samples_leaf)
                    results_cross_entropy.append(cross_entropy)
                    results_accuracy.append(accuracy)

results = pd.DataFrame({'scenario_name': results_scenario,
                        'criterion': results_criterion,
                        'ccp_alpha': results_ccp_alpha,
                        'alpha': results_alpha,
                        'min_samples_leaf': results_min_samples_leaf,
                        'cross_entropy': results_cross_entropy,
                        'accuracy': results_accuracy
                        })

if args.save_name is None:
    save_name = 'grid_search_results.csv'
else:
    save_name = args.save_name

results.to_csv(get_data_dir() + save_name, index=False)
