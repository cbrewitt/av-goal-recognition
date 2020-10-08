import pandas as pd

from av_goal_recognition.base import get_data_dir

results = pd.concat([pd.read_csv(get_data_dir() + 'grid_search_results_{}_fine.csv'.format(c))
                     for c in ['entropy']])

scenarios = results.scenario_name.unique()
for scenario in scenarios:
    scenario_results = results.loc[results.scenario_name == scenario]
    print('max accuracy:')
    print(scenario_results.iloc[scenario_results.accuracy.argmax()])
    print('min cross entropy:')
    print(scenario_results.iloc[scenario_results.cross_entropy.argmin()])

mean_results = results.groupby(['criterion', 'ccp_alpha', 'alpha', 'min_samples_leaf']).mean().reset_index()

print('max accuracy:')
print(mean_results.iloc[mean_results.accuracy.argmax()])
print('min cross entropy:')
print(mean_results.iloc[mean_results.cross_entropy.argmin()])
