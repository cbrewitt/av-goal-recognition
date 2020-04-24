import lanelet2
from lanelet2.geometry import findNearest
from lanelet2.core import BasicPoint2d
import matplotlib.pyplot as plt

from goal_recognition import ScenarioConfig, Scenario, FeatureExtractor
from lanelet_helpers import LaneletHelpers
from shapely.geometry import Point



map_meta = ScenarioConfig.load('scenario_config/heckstrasse.json')
origin = lanelet2.io.Origin(map_meta.lat_origin, map_meta.lon_origin)
projector = lanelet2.projection.UtmProjector(origin)
lanelet_map, _ = lanelet2.io.loadRobust(map_meta.lanelet_file, projector)

scenario = Scenario.load('scenario_config/heckstrasse.json')
scenario.plot()
feature_extractor = FeatureExtractor(lanelet_map)

#start = BasicPoint2d(12.7, -5.6)
#start = BasicPoint2d(72.0, -53.4)
start = BasicPoint2d(11.6, -7.9)
plt.plot([start.x], [start.y], 'yo')

#start_lanelet = findNearest(lanelet_map.laneletLayer, start, 1)[2][1]
start_lanelet = feature_extractor.lanelet_at(start)
LaneletHelpers.plot_lanelet(start_lanelet)
print(start_lanelet)
goal = BasicPoint2d(*map_meta.goals[0])
end_lanelet = lanelet_map.laneletLayer.nearest(goal, 1)[0]

# try routing
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                  lanelet2.traffic_rules.Participants.Vehicle)
graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)

#print(graph.reachableSet(start_lanelet, 100.0, 0))
route = graph.getRoute(start_lanelet, end_lanelet)
path = route.shortestPath()

for ll in path:
    LaneletHelpers.plot_lanelet(ll)

# for ll in graph.reachableSet(start_lanelet, 100.0, 0):
#     plot_lanelet(ll)

# for l in lanelet_map.laneletLayer:
#     if len(l.polygon2d()) > 0:
#         plot_lanelet(l)

plt.show()
