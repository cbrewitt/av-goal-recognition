import argparse
from av_goal_recognition.goal_recognition import DecisionTreeGoalRecogniser

parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')
parser.add_argument('--scenario', type=str, help='Name of scenario to validate', required=True)
args = parser.parse_args()

alpha = 1
scenario_name = args.scenario
model = DecisionTreeGoalRecogniser.train(scenario_name, alpha)
model.save(scenario_name)

