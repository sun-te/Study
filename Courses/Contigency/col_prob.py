import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle
import matplotlib.transforms as transforms
from util import confidence_ellipse, validation_rotation
from ipdb import set_trace as tt
POS_RATIO = 0.7

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
class Object:
    def __init__(self, x, y, length, width, angle):
        self.cx = x
        self.cy = y
        self.angle = angle * np.pi / 180.0
        self.ratio = POS_RATIO
        self.length = length
        self.width = width
        self.corner = self.generate()

    def generate(self, theta=None):
        if theta is None:
            theta = self.angle
        x, y = self.cx, self.cy
        min_x, max_x = - self.ratio * self.length, (1 - self.ratio) * self.length
        min_y, max_y = - 0.5 * self.width, 0.5 * self.width

        rot_matrix = rotation_matrix(theta)

        pts = np.array([[min_x, max_y],
                        [max_x, max_y],
                        [max_x, min_y],
                        [min_x, min_y]])
        pts = np.matmul(rot_matrix, pts.T).T
        pts += np.array([x, y])
        return pts

    def plot(self, ax=None):
        pts = np.concatenate([self.corner, self.corner[0].reshape(1, 2)], axis=0).T
        x, y = pts[0], pts[1]
        ax.plot(x, y)

        # plt.show()


class Robot(Object):
    def __init__(self, x, y, length, width, angle):
        super(Robot, self).__init__(x, y, length, width, angle)

    def plot(self, ax=None):
        pts = np.concatenate([self.corner, self.corner[0].reshape(1, 2)], axis=0).T
        x, y = pts[0], pts[1]
        ax.plot(x, y, label="Robot")

class Obstacle(Object):
    def __init__(self, x, y, length, width, angle, pos_cov=None, ang_var=None):
        super(Obstacle, self).__init__(x, y, length, width, angle)
        self.pos_cov = pos_cov
        self.ang_var = ang_var
        self.rot_matrix = None
        self.W = None
        self.setup()

    def setup(self):
        if self.pos_cov is None:
            self.pos_cov = np.array([[self.length, 0],
                                     [0, self.width]])

            theta = self.angle
            self.rot_matrix = np.array([[np.cos(theta), - np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])

            # The covariance of the position of the obstacle
            self.pos_cov = np.matmul(self.rot_matrix, np.matmul(self.pos_cov, self.rot_matrix.T))
        W = np.linalg.cholesky(self.pos_cov)
        # This W is the rotation matrix described in the equation (15), page 918
        self.W = np.linalg.inv(W)

        if self.ang_var is None:
            self.ang_var = np.pi / 30


    def validationRotation(self, ax):
        m = np.random.multivariate_normal([self.cx, self.cy], cov=self.pos_cov, size=5000).T
        x = m[0]
        y = m[1]
        confidence_ellipse(x, y, ax, edgecolor='red')
        ax.scatter(m[0], m[1])
        plt.show()
        m = self.W@m
        x = m[0]
        y = m[1]
        ax.scatter(m[0], m[1])
        confidence_ellipse(x, y, ax, edgecolor='red')
        plt.show()

    def plot(self, ax=None):
        if ax is None:
            figure = plt.figure()
            ax = figure.add_subplot(111, aspect='equal')
        pts = np.concatenate([self.corner, self.corner[0].reshape(1, 2)], axis=0).T
        x, y = pts[0], pts[1]
        plt.plot(x, y, label='Obstacle')
        pts1, pts2 = self.generate(self.angle - self.ang_var), self.generate(self.angle + self.ang_var)
        pts1 = np.concatenate([pts1, pts1[0].reshape(1, 2)], axis=0).T
        pts2 = np.concatenate([pts2, pts2[0].reshape(1, 2)], axis=0).T
        ax.plot(pts1[0], pts1[1])
        ax.plot(pts2[0], pts2[1])
        cov = self.pos_cov
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          edgecolor='red',
                          linestyle='--',
                          facecolor='none'
                          )

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0])
        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1])
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(self.cx, self.cy)

        ellipse.set_transform(transf + ax.transData)
        ax.add_artist(ellipse)


    def collision_box(self):
        # up, right, down, left
        ang_min, ang_max = self.angle - self.ang_var, self.angle + self.ang_var
        pts1, pts2 = self.generate(ang_min), self.generate(ang_max)
        pts = np.concatenate([pts1, pts2])

        x, y = pts.T[0], pts2.T[1]
        cx, cy = self.cx, self.cy

        box = {
            "up": max(y) - cy,
            "down": cy - min(y),
            "left": cx - min(x),
            "right": max(x) - cx
        }
        return box



class Collision:
    def __init__(self):
        self.robot = None
        self.obstacle = None
        self.cb_box = None
        self.demo_robot = None

    def cbPolygon(self, robot_corner=None, plot=False, ax=None):
        if robot_corner is None:
            if self.obstacle is None or self.robot is None:
                raise ValueError("No registered robot or obstacle, please provide both")
            assert self.robot is not None, "Must provide a robot shape"

            robot_corner = self.robot.corner
        cx = np.mean(robot_corner[:, 0])
        cy = np.mean(robot_corner[:, 1])
        robot_plot = None
        if plot:
            if ax is None:
                figure = plt.figure(0)
                ax = figure.add_subplot(111)
            robot_plot = np.concatenate([robot_corner, robot_corner[0].reshape(1, 2)], axis=0)
            robot_plot = Polygon(xy=robot_plot, closed=True, facecolor="blue", edgecolor='b', fill=True,
                                  alpha=0.3, linestyle='-')
            ax.add_artist(robot_plot)

        # We suppose the bounding box of the obstacle is always a rectangular
        obst_box = self.obstacle.collision_box()
        w, h = obst_box['right'] + obst_box['left'], obst_box['up'] + obst_box['down']
        cb_pts = []
        for pts in robot_corner:
            # clockwise adding the points
            x, y = pts

            if y - cy == 0:
                if x - cx > 0:
                    cb_pts.append([x + obst_box["left"], y + obst_box["down"]])
                    cb_pts.append([x + obst_box["left"], y - obst_box["up"]])
                    if plot:
                        ax.add_artist(Rectangle(xy=(x, y), width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))
                        ax.add_artist(Rectangle(xy=(x, y - h),
                                                width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))
                else:
                    cb_pts.append([x - obst_box["right"], y - obst_box["up"]])
                    cb_pts.append([x - obst_box["right"], y + obst_box["down"]])
                    if plot:
                        ax.add_artist(Rectangle(xy=(x - w, y - h),
                                                width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))
                        ax.add_artist(Rectangle(xy=(x - w, y),
                                                width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))

            elif x - cx == 0:
                if y - cy > 0:
                    cb_pts.append([x - obst_box["right"], y + obst_box["down"]])
                    cb_pts.append([x + obst_box["left"], y + obst_box["down"]])
                else:
                    cb_pts.append([x - obst_box["right"], y - obst_box["up"]])
                    cb_pts.append([x + obst_box["left"], y - obst_box["up"]])

            else:
                if y - cy > 0 and x - cx > 0:
                    cb_pts.append([x + obst_box["left"], y + obst_box["down"]])
                    if plot:
                        ax.add_artist(Rectangle(xy=(x, y), width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))
                        self.demo_obst = Obstacle(x + obst_box["left"], y + obst_box["down"],
                                                  width=self.obstacle.width, length=self.obstacle.length,
                                                  angle=self.obstacle.angle * 180 / np.pi)
                        self.demo_obst.plot(ax)
                if y - cy > 0 and x - cx < 0:
                    cb_pts.append([x - obst_box["right"], y + obst_box["down"]])
                    if plot:
                        ax.add_artist(Rectangle(xy=(x - w, y),
                                                width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))
                if y - cy < 0 and x - cx > 0:
                    cb_pts.append([x + obst_box["left"], y - obst_box["up"]])
                    if plot:
                        ax.add_artist(Rectangle(xy=(x, y - h),
                                                width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))
                if y - cy < 0 and x - cx < 0:
                    cb_pts.append([x - obst_box["right"], y - obst_box["up"]])
                    if plot:
                        ax.add_artist(Rectangle(xy=(x - w, y - h),
                                                width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))

        self.cb_box = np.array(cb_pts)

        if plot:
            cb_box = self.cb_box
            cb_box = np.concatenate([cb_box, cb_box[0].reshape(1, 2)], axis=0)
            pg = Polygon(xy=cb_box, closed=True, facecolor="None", edgecolor='b', fill=False, linestyle='--')
            ax.add_artist(pg)
            ax.legend([robot_plot, pg], ["robot", "cb_polygon"] )

        return self.cb_box

    def transformed_cb(self, plot=False):



        return



if __name__ == "__main__":
    # Firstly, let's verify some functionality
    # for i in range(3):
    #     validation_rotation()

    figure = plt.figure()
    ax = figure.add_subplot(111, aspect='equal')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    collision = Collision()
    collision.robot = Robot(0, 0, 10,3, 0)
    collision.obstacle = Obstacle(2, 3, 4.5, 2, 45)
    # collision.obstacle.plot(ax)
    # collision.robot.plot(ax)
    corner = np.array([[1, 0],
                       [0.5, -np.sqrt(3)/2],
                       [-0.5, -np.sqrt(3)/2],
                       [-1, 0],
                       [-0.5, np.sqrt(3)/2],
                       [0.5, np.sqrt(3)/2]]) * 5
    collision.cbPolygon(robot_corner=corner, ax=ax, plot=True)
    # collision.plot_cbPolygon(ax)
    # ax.legend()
    plt.show()


