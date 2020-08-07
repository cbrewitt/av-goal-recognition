import lanelet2
from lanelet2 import geometry
from lanelet2.core import BasicPoint2d, BoundingBox2d
import matplotlib.pyplot as plt

from av_goal_recognition.scenario import ScenarioConfig, Scenario
from av_goal_recognition.feature_extraction import FeatureExtractor
from av_goal_recognition.lanelet_helpers import LaneletHelpers

map_meta = ScenarioConfig.load('../scenario_config/heckstrasse.json')
origin = lanelet2.io.Origin(map_meta.lat_origin, map_meta.lon_origin)
projector = lanelet2.projection.UtmProjector(origin)
lanelet_map, _ = lanelet2.io.loadRobust(map_meta.lanelet_file, projector)

scenario = Scenario.load('../scenario_config/heckstrasse.json')
scenario.plot()
feature_extractor = FeatureExtractor(lanelet_map)
episodes = scenario.load_episodes()
#start = BasicPoint2d(12.7, -5.6)
#start = BasicPoint2d(72.0, -53.4)
#start = BasicPoint2d(27.2, -19.6)
start = BasicPoint2d(67.28, -12.94)
plt.plot([start.x], [start.y], 'yo')

#start_lanelet = findNearest(lanelet_map.laneletLayer, start, 1)[2][1]
nearest_lanelets = feature_extractor.lanelets_at(start)
start_lanelet = feature_extractor.lanelet_at(start)

radius = 1
bounding_box = BoundingBox2d(BasicPoint2d(start.x - radius, start.y - radius),
              BasicPoint2d(start.x + radius, start.y + radius))

nearby_lanelets = lanelet_map.laneletLayer.search(bounding_box)
lanelets_distance = [geometry.distance(l, start) for l in nearby_lanelets]
dist_along = [LaneletHelpers.dist_along(l, start) for l in nearby_lanelets]

state = episodes[0].agents[0].state_history[94]
current_lanelet = feature_extractor.get_current_lanelet(state)

print(start_lanelet)
goal = BasicPoint2d(*map_meta.goals[1])
end_lanelet = lanelet_map.laneletLayer.nearest(goal, 1)[0]
#end = BasicPoint2d(64.0, -17.0)
#end_lanelet = lanelet_map.laneletLayer.nearest(end, 1)[0]

LaneletHelpers.plot(start_lanelet)
LaneletHelpers.plot(end_lanelet)

# try routing
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                              lanelet2.traffic_rules.Participants.Vehicle)
graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)

#print(graph.reachableSet(start_lanelet, 100.0, 0))
route = graph.getRoute(start_lanelet, end_lanelet)
path = route.shortestPath()

#LaneletHelpers.plot_path(path)

for ll in feature_extractor.lanelets_to_cross(route):
    LaneletHelpers.plot(ll)

# for l in lanelet_map.laneletLayer:
#     if len(l.polygon2d()) > 0:
#         plot_lanelet(l)

plt.show()
