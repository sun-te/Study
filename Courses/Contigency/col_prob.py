import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from util import confidence_ellipse, validation_rotation
from ipdb import set_trace as tt
POS_RATIO = 0.3
FIGURE = plt.figure()
AX = FIGURE.add_subplot(111, aspect='equal')

class Object:
    def __init__(self, x, y, length, width, angle):
        self.cx = x
        self.cy = y
        self.angle = angle * np.pi / 180.0
        self.ratio = POS_RATIO
        self.length = length
        self.width = width
        self.corner = None
        self.generate()

    def generate(self, theta=None):
        if theta is None:
            theta = self.angle
        x, y = self.cx, self.cy
        min_x, max_x = - self.ratio * self.length, (1 - self.ratio) * self.length
        min_y, max_y = - 0.5 * self.width, 0.5 * self.width

        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])

        pts = np.array([[min_x, max_y],
                        [max_x, max_y],
                        [max_x, min_y],
                        [min_x, min_y]])
        pts = np.matmul(rot_matrix, pts.T).T
        pts += np.array([x, y])
        self.corner = pts

    def plot(self):
        pts = np.concatenate([self.corner, self.corner[0].reshape(1,2)],axis=0).T
        x, y = pts[0], pts[1]
        plt.plot(x, y)

        # plt.show()


class Robot(Object):
    def __init__(self, x, y, length, width, angle):
        super(Robot, self).__init__(x, y, length, width, angle)


class Obstacle(Object):
    def __init__(self, x, y, length, width, angle, pos_cov=None, ang_var=None):
        super(Obstacle, self).__init__(x, y, length, width, angle)
        self.pos_cov = pos_cov
        self.ang_var = ang_var
        self.setup()

    def setup(self):
        if self.pos_cov is None:
            self.var_length, self.var_width = self.length, self.width
            self.pos_cov = np.array([[self.var_length, 0],
                                     [0, self.var_width]])

            theta = self.angle
            self.rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])

            # The covariance of the position of the obstacle
            self.pos_cov = np.matmul(self.rot_matrix, np.matmul(self.pos_cov, self.rot_matrix.T))
        W = np.linalg.cholesky(self.pos_cov)
        # This W is the rotation matrix described in the equation (15), page 918
        self.W = np.linalg.inv(W)

    def validationRotation(self):
        m = np.random.multivariate_normal([self.cx, self.cy], cov=self.pos_cov, size=5000).T
        x = m[0]
        y = m[1]
        confidence_ellipse(x, y, AX, edgecolor='red')
        AX.scatter(m[0], m[1])
        plt.show()
        m = self.W@m
        x = m[0]
        y = m[1]
        AX.scatter(m[0], m[1])
        confidence_ellipse(x, y, AX, edgecolor='red')
        plt.show()

    def plot(self):
        pts = np.concatenate([self.corner, self.corner[0].reshape(1, 2)], axis=0).T
        x, y = pts[0], pts[1]
        plt.plot(x, y)
        ang = self.angle / np.pi * 180
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
        mean_y = np.mean(y)
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(self.cx, self.cy)

        ellipse.set_transform(transf + AX.transData)
        AX.add_artist(ellipse)
        plt.show()




class Collision:
    def __init__(self):
        self.robot = None
        self.obstacle = None

    def cbPolygon(self, robot_corner=None):
        if robot_corner is None:
            robot_corner = self.robot.corner
        cx = np.mean(robot_corner[:, 0])
        cy = np.mean(robot_corner[:, 1])

        # TODO:



if __name__ == "__main__":
    # Firstly, let's verify some functionality
    for i in range(3):
        validation_rotation()
    collision = Collision()
    collision.robot = Robot(1, 1, 10,3, 45)
    collision.obstacle = Obstacle(2, 3, 4.5, 2, 45)
    collision.obstacle.plot()
