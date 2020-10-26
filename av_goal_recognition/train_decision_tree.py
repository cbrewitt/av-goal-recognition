import argparse
from av_goal_recognition.goal_recognition import DecisionTreeGoalRecogniser

parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
parser.add_argument('--scenario', type=str, help='Name of scenario to validate', default=None)
args = parser.parse_args()

if args.scenario is None:
    scenario_names = ['heckstrasse', 'bendplatz', 'frankenberg']
else:
    scenario_names = [args.scenario]

for scenario_name in scenario_names:
    model = DecisionTreeGoalRecogniser.train(scenario_name, min_samples_leaf=20, max_leaf_nodes=7)
    model.save(scenario_name)
