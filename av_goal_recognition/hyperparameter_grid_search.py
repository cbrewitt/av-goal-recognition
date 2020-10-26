import argparse

import pandas as pd
import numpy as np

from av_goal_recognition.base import get_data_dir
from av_goal_recognition.data_processing import get_dataset
from av_goal_recognition.goal_recognition import TrainedDecisionTrees

parser = argparse.ArgumentParser(description='Hyperparameter grid search')
parser.add_argument('--slice', type=str, help='results file name', default=None)
args = parser.parse_args()

if args.slice is None:
    num_slices = 1
    slice_idx = 0
else:
    slice_idx, num_slices = [int(x) for x in args.slice.split('/')]

scenarios = ['heckstrasse', 'bendplatz', 'frankenberg']
criterions = ['entropy']
ccp_alpha_grid = np.logspace(-3, -2, 20)
alphas = np.logspace(-4, 0, 30)
#max_leaf_nodes_grid = np.unique(np.round(np.logspace(0.3, 2, 20))).astype(int)
#alpha = 1
min_samples_leaf = 1

results_scenario = []
results_criterion = []
results_alpha = []
results_ccp_alpha = []
results_accuracy = []

idx = 0
combinations = []

for scenario_name in scenarios:
    for criterion in criterions:
        for ccp_alpha in ccp_alpha_grid:
            for alpha in alphas:
                combinations.append((scenario_name, criterion, ccp_alpha, alpha))

prev_scenario_name = None
start_idx = len(combinations) // num_slices * slice_idx
end_idx = len(combinations) // num_slices * (slice_idx + 1)
for combination_idx in range(start_idx, end_idx):
    combination = combinations[combination_idx]
    scenario_name, criterion, ccp_alpha, alpha = combination
    if scenario_name != prev_scenario_name:
        train_set = get_dataset(scenario_name, 'train')
        valid_set = get_dataset(scenario_name, 'valid')

    print('iteration {}/{}'.format(combination_idx - start_idx + 1, end_idx - start_idx))
    model = TrainedDecisionTrees.train(scenario_name,
                                       max_depth=7,
                                       alpha=alpha,
                                       criterion=criterion,
                                       min_samples_leaf=min_samples_leaf,
                                       ccp_alpha=ccp_alpha,
                                       training_set=train_set)
    unique_samples = model.batch_goal_probabilities(valid_set)
    unique_samples['model_correct'] = (unique_samples['model_prediction']
                                       == unique_samples['true_goal'])
    accuracy = unique_samples.model_correct.mean()

    results_scenario.append(scenario_name)
    results_criterion.append(criterion)
    results_ccp_alpha.append(ccp_alpha)
    results_accuracy.append(accuracy)
    results_alpha.append(alpha)


results = pd.DataFrame({'scenario_name': results_scenario,
                        'ccp_alpha': results_ccp_alpha,
                        'criterion': results_criterion,
                        'alpha': results_alpha,
                        'accuracy': results_accuracy
                        })


save_name = 'grid_search_results_{}.csv'.format(slice_idx)
results.to_csv(get_data_dir() + save_name, index=False)
