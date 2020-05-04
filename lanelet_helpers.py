from lanelet2 import geometry
import numpy as np
import matplotlib.pyplot as plt


class LaneletHelpers:

    @classmethod
    def plot_path(cls, path):
        for l in path:
            cls.plot(l)

    @staticmethod
    def plot(l):
        points_x = [p.x for p in l.polygon2d()]
        points_x.append(points_x[0])
        points_y = [p.y for p in l.polygon2d()]
        points_y.append(points_y[0])
        plt.plot(points_x, points_y, color='red')
        plt.plot(points_x, points_y, color='red')

    @staticmethod
    def dist_along(lanelet, point):
        centerline = geometry.to2D(lanelet.centerline)
        return geometry.toArcCoordinates(centerline, point).length

    @staticmethod
    def heading_at(lanelet, point):
        centerline = geometry.to2D(lanelet.centerline)
        proj_point = geometry.project(centerline, point)
        dist_along = LaneletHelpers.dist_along(lanelet, proj_point)
        epsilon = 1e-4
        if geometry.length2d(lanelet) - dist_along > epsilon:
            # get a point after current point
            next_point = geometry.interpolatedPointAtDistance(
                centerline, dist_along + epsilon)
            x1 = next_point.y - proj_point.y
            x2 = next_point.x - proj_point.x
            lane_heading = np.arctan2(x1, x2)
        else:
            # get a point before current point
            prev_point = geometry.interpolatedPointAtDistance(
                centerline, dist_along - epsilon)
            x1 = proj_point.y - prev_point.y
            x2 = proj_point.x - prev_point.x
            lane_heading = np.arctan2(x1, x2)
        return lane_heading

    @staticmethod
    def left_of(a, b):
        # return true if a is left of b
        return a.rightBound == b.leftBound

    @staticmethod
    def follows(a, b):
        # return true if a follows b
        return a.rightBound[0] == b.rightBound[-1] and a.leftBound[0] == b.leftBound[-1]

    @classmethod
    def connected(cls, a, b):
        return cls.follows(a, b) or cls.follows(b, a) or cls.left_of(a, b) or cls.left_of(b, a)


