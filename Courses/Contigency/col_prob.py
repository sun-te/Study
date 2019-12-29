import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle
import matplotlib.transforms as transforms
from util import confidence_ellipse, validation_rotation, rotation_matrix, close_polygone
from util import compute_proba, monteCarlo
from ipdb import set_trace as tt
POS_RATIO = 0.3



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
        ax.scatter(self.cx, self.cy)

class Obstacle(Object):
    def __init__(self, x, y, length, width, angle, pos_cov=None, ang_var=None):
        super(Obstacle, self).__init__(x, y, length, width, angle)
        self.ratio = POS_RATIO
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
        m = self.W @ m
        x = m[0]
        y = m[1]
        ax.scatter(m[0], m[1])
        confidence_ellipse(x, y, ax, edgecolor='red')
        plt.show()

    def plot(self, ax=None, transformer=np.eye(2)):
        if ax is None:
            figure = plt.figure()
            ax = figure.add_subplot(111, aspect='equal')
        pts = np.concatenate([self.corner, self.corner[0].reshape(1, 2)], axis=0).T
        pts = transformer.T @ (pts - np.array([[self.cx], [self.cy]])) + np.array([[self.cx], [self.cy]])
        x, y = pts[0], pts[1]
        ax.plot(x, y)
        pts1, pts2 = self.generate(self.angle - self.ang_var), self.generate(self.angle + self.ang_var)

        pts1 = np.concatenate([pts1, pts1[0].reshape(1, 2)], axis=0).T
        pts2 = np.concatenate([pts2, pts2[0].reshape(1, 2)], axis=0).T
        pts1, pts2 = transformer.T @ (pts1 - np.array([[self.cx], [self.cy]])) + np.array([[self.cx], [self.cy]]), \
                     transformer.T @ (pts2 - np.array([[self.cx], [self.cy]])) + np.array([[self.cx], [self.cy]])
        ax.plot(pts1[0], pts1[1])
        ax.plot(pts2[0], pts2[1])
        ax.scatter(self.cx, self.cy)
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
        self.robot_corner = None

    def cbPolygon(self, robot_corner=None, plot=False, ax=None):
        if robot_corner is None:
            if self.obstacle is None or self.robot is None:
                raise ValueError("No registered robot or obstacle, please provide both")
            assert self.robot is not None, "Must provide a robot shape"

            robot_corner = self.robot.corner
        self.robot_corner = robot_corner
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
                    self.demo_obst = Obstacle(x + obst_box["left"], y + obst_box["down"],
                                              width=self.obstacle.width, length=self.obstacle.length,
                                              angle=self.obstacle.angle * 180 / np.pi)
                    if plot:
                        ax.add_artist(Rectangle(xy=(x, y), width=w, height=h,
                                                edgecolor='green', linestyle='--', facecolor='none'))

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

    def transformed_cb(self, plot=False, ax1=None, ax2=None):
        """

        :param plot:
        :param ax1:
        :param ax2:
        :return: the rotated bounding box of the combined body and the transformed center of the obstacle
        """
        W = self.obstacle.W

        # min_x, min_y, width, height
        bbx = min(self.cb_box[:,0]), min(self.cb_box[:,1]), max(self.cb_box[:,0]) - min(self.cb_box[:,0]), \
                  max(self.cb_box[:,1]) - min(self.cb_box[:,1])

        bounding_box_corner = np.array([[bbx[0], bbx[1]],
                                        [bbx[0], bbx[1] + bbx[3]],
                                        [bbx[0] + bbx[2], bbx[1] + bbx[3]],
                                        [bbx[0] + bbx[2], bbx[1]]
                                        ])

        self.cb_box_transformed = self.cb_box @ W.T
        cb_box_t = np.concatenate([self.cb_box_transformed, self.cb_box_transformed[0].reshape(1, 2)], axis=0)
        pg = Polygon(xy=cb_box_t, closed=True, facecolor="None", edgecolor='b', fill=False, linestyle='--')

        robot_plot = self.robot_corner @ W.T
        trans_robot = robot_plot
        robot_plot = np.concatenate([robot_plot, robot_plot[0].reshape(1, 2)], axis=0)
        robot_plot = Polygon(xy=robot_plot, closed=True, facecolor="blue", edgecolor='b', fill=True,
                             alpha=0.3, linestyle='-')

        demo_center = np.array([self.demo_obst.cx, self.demo_obst.cy]) @ W.T
        demo_obst = Obstacle(demo_center[0], demo_center[1],
                             width=self.obstacle.width, length=self.obstacle.length,
                             angle=self.obstacle.angle * 180 / np.pi,
                             pos_cov=W@self.obstacle.pos_cov@W.T)


        transformed_bbx = bounding_box_corner @ W.T
        plot_transformed_bbx = Polygon(xy=transformed_bbx, closed=True, facecolor="None", edgecolor='b',
                                       linestyle='--')

        # compute angle
        line = np.array([1, 0]).reshape(1, 2)
        line = line @ W.T
        theta = np.arctan(line[0, 1] / line[0, 0])
        rot_matrix = rotation_matrix(- theta)
        # re-rotated combined body and its bounding box
        rot_transform_cb = cb_box_t @ rot_matrix.T
        rot_pg = Polygon(xy=rot_transform_cb, closed=True, facecolor="None", edgecolor='b', fill=False, linestyle='--')
        rot_bbx = min(rot_transform_cb[:,0]), min(rot_transform_cb[:,1]), \
                  max(rot_transform_cb[:,0]) - min(rot_transform_cb[:,0]), \
                  max(rot_transform_cb[:,1]) - min(rot_transform_cb[:,1])
        bounding_box_corner = np.array([[rot_bbx[0], rot_bbx[1]],
                                        [rot_bbx[0], rot_bbx[1] + rot_bbx[3]],
                                        [rot_bbx[0] + rot_bbx[2], rot_bbx[1] + rot_bbx[3]],
                                        [rot_bbx[0] + rot_bbx[2], rot_bbx[1]]
                                        ])
        plot_rotated_bbx = Polygon(xy=bounding_box_corner, closed=True, facecolor="None", edgecolor='orange',
                                       linestyle='--')
        demo_center = np.array([self.obstacle.cx, self.obstacle.cy]) @ W.T @ rot_matrix.T
        demo_obst_rot = Obstacle(demo_center[0], demo_center[1],
                             width=self.obstacle.width, length=self.obstacle.length,
                             angle=self.obstacle.angle * 180 / np.pi,
                             pos_cov=W @ self.obstacle.pos_cov @ W.T)


        # TODO: bouding box for the shape
        if plot:
            ax1.add_artist(pg)
            demo_obst.plot(ax1, transformer=W)
            ax1.add_artist(robot_plot)
            # ax.add_artist(plot_transformed_bbx)
            ax1.legend([robot_plot, pg], ["transformed robot", 'transformed cb polygon'])
            ax2.add_artist(plot_rotated_bbx)
            ax2.add_artist(rot_pg)
            demo_obst_rot.plot(ax2, transformer=W)
            ax2.legend([plot_rotated_bbx, rot_pg], ["bounding box", 'rotated cb polygon'])
        print("Collision probability upper bound: {}".format(compute_proba(bounding_box_corner, demo_center)))
        return bounding_box_corner, demo_center



if __name__ == "__main__":
    # # Firstly, let's verify some functionality
    # # Idealy, the transformed data should be of covariance of
    # # Identity matrix, which means that the equal probability line should be
    # # a circle
    # for i in range(3):
    #     validation_rotation()
    #
    # ##########################################################
    # # This part is for the validation of the implementation  #
    # # for probability upper bound computation                #
    # ##########################################################
    # figure = plt.figure(figsize=[15,5])
    # ax = figure.add_subplot(131, aspect='equal')
    # ax.set_xlim([-10, 10])
    # ax.set_ylim(  [-10, 10])
    # ax.set_title("Figure with demo obstacle")
    # collision = Collision()
    # collision.robot = Robot(0, 0, 4.5, 2, 0)
    # collision.obstacle = Obstacle(9, 8, 4.5, 2, 45)
    # # collision.obstacle.plot(ax)
    # # collision.robot.plot(ax)
    # corner = np.array([[1, 0],
    #                    [0.5, -np.sqrt(3)/2],
    #                    [-0.5, -np.sqrt(3)/2],
    #                    [-1, 0],
    #                    [-0.5, np.sqrt(3)/2],
    #                    [0.5, np.sqrt(3)/2]]) * 5
    # # a hexagon like robot area
    # collision.cbPolygon(robot_corner=corner, ax=ax, plot=True)
    # # de-comment the line below to have a rectangular robot
    # # collision.cbPolygon(ax=ax, plot=True)
    # ax2 = figure.add_subplot(132, aspect='equal')
    # ax2.set_xlim([-7, 7])
    # ax2.set_ylim([-7, 7])
    # ax2.set_title("Figure with demo obstacle")
    # ax3 = figure.add_subplot(133, aspect='equal')
    # ax3.set_title("Figure with real obstacle")
    # ax3.set_xlim([-7, 7])
    # ax3.set_ylim([-7, 7])
    # collision.transformed_cb(plot=True, ax1=ax2, ax2=ax3)
    # plt.show()

    #########################################
    # This part is for the computation of   #
    # the probability during time steps     #
    #########################################
    width = 2
    length = 4
    collision = Collision()
    collision.robot = Robot(0, 0, width=width, length=length, angle=0)

    figure = plt.figure(figsize=[8, 6])
    ax = figure.add_subplot(111, aspect='equal')
    ax.set_xlim([-6, 10])
    ax.set_ylim([-6, 6])
    collision.robot.plot(ax)
    initial_state = [8, 3, 180]
    collision.obstacle = Obstacle(initial_state[0], initial_state[1],
                                  width=width, length=length,
                                  angle=initial_state[2])
    # collision.obstacle.plot(ax)

    position = np.array([initial_state[:2],
                         [6, 3], [4, 3], [2, 3],
                         [0.2, 2.8], [-0.5, 2.5], [-1.7, 1.5],
                         [-2.2, 0.5], [-2.2, -1], [-2.2, -2.5],
                         ])
    # tt()
    angle = np.array([0, 0, 0, 0, 30, 40, 60, 70, 80 ,90 ,90]) + 180
    # uncertainty = np.exp(np.linspace(0.1, 1.5, num=10)) * 0.25
    uncertainty = np.ones(10) * 0.5
    robot_corner = collision.robot.corner
    for i in range(6, len(position)):
        collision.obstacle = Obstacle(position[i, 0], position[i, 1],
                                  width=width, length=length,
                                  angle=angle[i])
        collision.obstacle.pos_cov *= uncertainty[i] ** 2
        pos_cov = collision.obstacle.pos_cov
        collision.obstacle.ang_var *= uncertainty[i]
        real_proba = monteCarlo(position[i, 0], position[i,1], pos_cov, robot_corner, ax=ax)
        print("Real probability : {}".format(real_proba))
        collision.obstacle.plot(ax)
        collision.cbPolygon()
        collision.transformed_cb()

    ax.set_title("Robot-Obstacle Collision Detection")
    plt.show()
