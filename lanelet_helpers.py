import matplotlib.pyplot as plt


class LaneletHelpers:

    @classmethod
    def plot_path(cls, path):
        for l in path:
            cls.plot_lanelet(l)

    @staticmethod
    def plot_lanelet(l):
        points_x = [p.x for p in l.polygon2d()]
        points_x.append(points_x[0])
        points_y = [p.y for p in l.polygon2d()]
        points_y.append(points_y[0])
        plt.plot(points_x, points_y, color='red')
        plt.plot(points_x, points_y, color='red')
