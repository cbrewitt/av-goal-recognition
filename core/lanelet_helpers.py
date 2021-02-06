import shapely
from lanelet2 import geometry
from lanelet2.core import BasicPoint2d
from shapely.geometry import Polygon, LineString, Point
from shapely.errors import TopologicalError
import numpy as np
import matplotlib.pyplot as plt


class LaneletHelpers:

    @classmethod
    def plot_path(cls, path):
        for l in path:
            cls.plot(l)

    @staticmethod
    def plot(l, color='red'):
        points_x = [p.x for p in l.polygon2d()]
        points_x.append(points_x[0])
        points_y = [p.y for p in l.polygon2d()]
        points_y.append(points_y[0])
        plt.plot(points_x, points_y, color=color)

    @staticmethod
    def dist_along(lanelet, point):
        centerline = geometry.to2D(lanelet.centerline)
        return geometry.toArcCoordinates(centerline, point).length

    @staticmethod
    def heading_at(lanelet, point):
        direction = LaneletHelpers.direction_at(lanelet, point)
        heading = np.arctan2(direction[1], direction[0])
        return heading

    @staticmethod
    def direction_at(lanelet, point):
        centerline = geometry.to2D(lanelet.centerline)
        proj_point = geometry.project(centerline, point)
        dist_along = LaneletHelpers.dist_along(lanelet, proj_point)
        epsilon = 1e-4
        if geometry.length2d(lanelet) - dist_along > epsilon:
            # get a point after current point
            next_point = geometry.interpolatedPointAtDistance(
                centerline, dist_along + epsilon)
            direction = [next_point.x - proj_point.x, next_point.y - proj_point.y]
        else:
            # get a point before current point
            prev_point = geometry.interpolatedPointAtDistance(
                centerline, dist_along - epsilon)
            direction = [proj_point.x - prev_point.x, proj_point.y - prev_point.y]
        direction = direction / np.linalg.norm(direction)
        return direction

    @staticmethod
    def left_of(a, b):
        # return true if a is left of b
        return a.rightBound.id == b.leftBound.id

    @classmethod
    def beside(cls, a, b):
        return (a.leftBound.id == b.leftBound.id
                or a.rightBound.id == b.rightBound.id
                or a.leftBound.id == b.rightBound.id
                or a.rightBound.id == b.leftBound.id)

    @classmethod
    def adjacent(cls, a, b):
        return cls.left_of(a, b) or cls.left_of(b, a)

    @staticmethod
    def follows(a, b):
        # return true if a follows b
        return a.rightBound[0].id == b.rightBound[-1].id and a.leftBound[0].id == b.leftBound[-1].id

    @classmethod
    def can_pass(cls, a, b):
        return cls.follows(a, b) or cls.follows(b, a) or cls.adjacent(a, b)

    @classmethod
    def connected(cls, a, b):
        return cls.follows(a, b) or cls.follows(b, a) or cls.beside(a, b)

    @staticmethod
    def dist_from_center(point, lanelet):
        return geometry.toArcCoordinates(geometry.to2D(lanelet.centerline), point).distance

    @staticmethod
    def shapely_polygon(lanelet):
        linestrings = lanelet.polygon2d().lineStrings()
        points = [(p.x, p.y) for ls in linestrings for p in ls]
        return Polygon(points)

    @staticmethod
    def shapely_point_to_lanelet(p):
        return BasicPoint2d(p.x, p.y)

    @staticmethod
    def lanelet_point_to_shapely(p):
        return Point(p.x, p.y)

    @classmethod
    def overlap_area(cls, l1, l2):
        p1 = cls.shapely_polygon(l1)
        p2 = cls.shapely_polygon(l2)
        try:
            overlap = p1.intersection(p2)
            if overlap.area > 0:
                centroid = LaneletHelpers.shapely_point_to_lanelet(overlap.centroid)
            else:
                centroid = None
            return overlap.area, centroid
        except TopologicalError:
            return 0.0, None

    @staticmethod
    def shapely_linestring(ls):
        return LineString([(p.x, p.y) for p in ls])

    @classmethod
    def intersection_point(cls, ls1, ls2):
        ls1_shapely = cls.shapely_linestring(ls1)
        ls2_shapely = cls.shapely_linestring(ls2)
        p = ls1_shapely.intersection(ls2_shapely)
        return cls.shapely_point_to_lanelet(p)

    @staticmethod
    def virtual(l):
        return l.leftBound.attributes['type'] == 'virtual' and l.rightBound.attributes['type'] == 'virtual'

    @staticmethod
    def get_path_ls(path):
        final_point = path[-1].centerline[-1]
        lane_points = [(p.x, p.y) for l in path for p in list(l.centerline)[:-1]] \
            + [(final_point.x, final_point.y)]
        lane_ls = LineString(lane_points)
        return lane_ls
