import lanelet2
from lanelet2.projection import UtmProjector
from test.lanelet_test_helpers import get_test_map

lanelet_map = get_test_map()

projector = UtmProjector(lanelet2.io.Origin(49, 8.4))
lanelet2.io.write('test_map_1.osm', lanelet_map, projector)