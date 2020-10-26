import pandas as pd

from av_goal_recognition.base import get_data_dir
from av_goal_recognition.goal_recognition import DecisionTreeGoalRecogniser

results = pd.concat([pd.read_csv(get_data_dir() + 'grid_search_results_{}.csv'.format(idx)) for idx in range(8)])

scenarios = results.scenario_name.unique()
for scenario in scenarios:
    scenario_results = results.loc[results.scenario_name == scenario]
    print('max accuracy:')
    best_idx = scenario_results.accuracy.argmax()
    print(scenario_results.iloc[best_idx])

    model = DecisionTreeGoalRecogniser.train(scenario,
                                             ccp_alpha=scenario_results.ccp_alpha.iloc[best_idx],
                                             criterion=scenario_results.criterion.iloc[best_idx],
                                             max_depth=7)
    model.save(scenario)

mean_results = results.groupby(['criterion', 'ccp_alpha']).mean().reset_index()

print('max accuracy:')
print(mean_results.iloc[mean_results.accuracy.argmax()])
