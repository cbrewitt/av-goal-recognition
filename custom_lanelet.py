"""
Functions relating to lanelet2 maps (using shapely)
"""
import shapely
from shapely.geometry import Polygon, Point
import lanelet2
from lanelet2.core import LaneletMap
from lanelet2.geometry import findNearest


class CustomLaneletMap(LaneletMap):
    """lanelet map with added features"""

    def __init__(self, filepath, lat_origin=0, lon_origin=0):
        self.projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        lanelet_map, _ = lanelet2.io.loadRobust(filepath, self.projector)
        self.lanelet_map = lanelet_map

    def lanelets_at(self, point):
        point = Point(point)
        lanelets = []
        for ll in self.lanelet_map.laneletLayer:
            points = [[pt.x, pt.y] for pt in ll.polygon2d()]
            poly = Polygon(points)
            if poly.contains(point):
                lanelets.append(ll)
        return lanelets

    def nearest_lanelet(self, point):
        return findNearest(self.lanelet_map.laneletLayer, point, 1)

