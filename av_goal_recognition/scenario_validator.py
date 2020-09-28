import argparse

from lanelet2.core import BasicPoint2d
import matplotlib.pyplot as plt

from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.scenario import Scenario
from av_goal_recognition.lanelet_helpers import LaneletHelpers


def validate_scenario(scenario: Scenario):

    feature_extractor = FeatureExtractor(scenario.lanelet_map)
    for start, end in scenario.config.reachable_pairs:
        print(start, end)
        start_point = BasicPoint2d(*start)
        start_lanelet = feature_extractor.lanelet_at(start_point)
        route = feature_extractor.route_to_goal(start_lanelet, end)
        if route is None:
            print('Failed to route from {} to {}'.format(start, end))
            debug_plot(start_lanelet, feature_extractor, scenario)
            import pdb; pdb.set_trace()


def debug_plot(start_lanelet, feature_extractor, scenario):
    reachable_lanelets = feature_extractor.routing_graph.reachableSet(
        start_lanelet, 100.0, 0)
    scenario.plot()
    LaneletHelpers.plot_path(reachable_lanelets)
    fig = plt.gcf()

    def onclick(event):
        if event.button == 1:
            click_point = BasicPoint2d(event.xdata, event.ydata)
            lanelets = feature_extractor.lanelets_at(click_point)
            print('lanelets at ({:.2f}, {:.2f}): {}'.format(
                click_point.x, click_point.y, [l.id for l in lanelets]))
            # for lanelet in lanelets:
            #     LaneletHelpers.plot(lanelet, color='yellow')
            # plt.show()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a scenario lanelet map')
    parser.add_argument('--scenario', type=str, help='Name of scenario to validate', required=True)
    args = parser.parse_args()
    scenario = Scenario.load('../scenario_config/{}.json'.format(args.scenario))
    validate_scenario(scenario)
