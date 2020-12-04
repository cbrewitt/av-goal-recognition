import pandas as pd
import json
import os

from core.base import get_data_dir, get_dt_config_dir


def main():
    if not os.path.exists(get_dt_config_dir()):
        os.makedirs(get_dt_config_dir())

    results = pd.concat([pd.read_csv(get_data_dir() + 'grid_search_results_{}.csv'.format(idx)) for idx in range(8)])

    scenarios = results.scenario_name.unique()
    for scenario in scenarios:
        scenario_results = results.loc[results.scenario_name == scenario]
        print('max accuracy:')
        best_idx = scenario_results.accuracy.argmax()
        print(scenario_results.iloc[best_idx])

        dt_params = {'ccp_alpha': scenario_results.ccp_alpha.iloc[best_idx],
                     'criterion': scenario_results.criterion.iloc[best_idx],
                     'max_depth': 7}

        with open(get_dt_config_dir() + scenario + '.json', 'w') as f:
            json.dump(dt_params, f)

    mean_results = results.groupby(['criterion', 'ccp_alpha']).mean().reset_index()

    print('max accuracy:')
    print(mean_results.iloc[mean_results.accuracy.argmax()])


if __name__ == '__main__':
    main()
