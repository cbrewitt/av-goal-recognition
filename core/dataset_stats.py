import pandas as pd
from core.data_processing import get_dataset

scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg']
dataset_names = ['train', 'valid', 'test']

total_vehicles = pd.DataFrame(columns=scenario_names, index=dataset_names)
total_samples = pd.DataFrame(columns=scenario_names, index=dataset_names)

for scenario_name in scenario_names:
    for dataset_name in dataset_names:
        dataset = get_dataset(scenario_name, dataset_name)
        vehicles = dataset[['episode', 'agent_id']].drop_duplicates().shape[0]
        samples = dataset[['episode', 'agent_id', 'fraction_observed']].drop_duplicates().shape[0]
        total_vehicles.loc[dataset_name, scenario_name] = vehicles
        total_samples.loc[dataset_name, scenario_name] = samples

print(total_vehicles)
print(total_samples)
