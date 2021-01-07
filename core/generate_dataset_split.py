import random
import json
from core.base import get_scenario_config_dir, get_base_dir
from core.scenario import Scenario


def load_dataset_splits():
    with open(get_base_dir() + '/core/dataset_split.json', 'r') as f:
        return json.load(f)


def main():
    random.seed(20210106)
    scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg']

    data_subset_episodes = {}

    for scenario_name in scenario_names:
        scenario = Scenario.load(get_scenario_config_dir() + '{}.json'.format(scenario_name))
        episodes = list(range(len(scenario.config.episodes)))
        random.shuffle(episodes)
        data_subset_episodes[scenario_name] = {'train': episodes[2:],
                                               'valid': [episodes[0]],
                                               'test': [episodes[1]]}

    with open(get_base_dir() + '/core/dataset_split.json', 'w') as f:
        json.dump(data_subset_episodes, f, indent=4)


if __name__ == '__main__':
    main()
    print(load_dataset_splits())
