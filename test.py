import lanelet2
from lanelet2.geometry import findNearest
from lanelet2.core import BasicPoint2d

from goal_recognition import ScenarioConfig
from shapely.geometry import Point

map_meta = ScenarioConfig.load('scenario_config/heckstrasse.json')
origin = lanelet2.io.Origin(map_meta.lat_origin, map_meta.lon_origin)
projector = lanelet2.projection.UtmProjector(origin)
lanelet_map, _ = lanelet2.io.loadRobust(map_meta.lanelet_file, projector)

start = BasicPoint2d(13.3, -8.95)

start_lanelet = lanelet_map.laneletLayer.nearest(start, 1)[0]

reachable_goal = BasicPoint2d(*map_meta.goals[1])
unreachable_goal = BasicPoint2d(*map_meta.goals[0])
end_lanelet = lanelet_map.laneletLayer.nearest(reachable_goal, 1)[0]

# try routing
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                  lanelet2.traffic_rules.Participants.Vehicle)
graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)

#print(graph.reachableSet(start_lanelet, 100.0, 0))

route = graph.getRoute(start_lanelet, end_lanelet)
path = route.shortestPath()
for ll in path:
    print(ll)

