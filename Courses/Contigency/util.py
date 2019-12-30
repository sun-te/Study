import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.special import erf


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    #####
    Inspired by demo of the matplotlib
    #####
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def validation_rotation():
    """
    To verify the correctness of implementation described in equation (13) page 918
    :return:
    """
    x, y = np.random.uniform(-10,10, 2)
    pos_cov = np.array([[np.random.randint(1,10), 0.],
                             [0., np.random.randint(1,10)]])

    theta = np.random.uniform(0, 360)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    pos_cov = np.matmul(rot_matrix, np.matmul(pos_cov, rot_matrix.T))
    W = np.linalg.cholesky(pos_cov)
    # This W is the rotation matrix described in the equation (15), page 918
    figure = plt.figure(figsize=[10,5])
    ax1 = figure.add_subplot(121, aspect='equal')
    ax1.set_title("Distribution before")
    W = np.linalg.inv(W)
    m = np.random.multivariate_normal([x, y], cov=pos_cov, size=1000).T
    ax1.scatter(m[0], m[1], s=5)
    confidence_ellipse(m[0], m[1], ax1, edgecolor='red')
    m = W@m
    # ax = figure.add_subplot(111, aspect='equal')
    ax2 = figure.add_subplot(122, aspect='equal')
    ax2.set_title("Rotated Distribution")
    ax2.scatter(m[0], m[1], s=5)
    confidence_ellipse(m[0], m[1], ax2, edgecolor='red')
    plt.show()

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

def close_polygone(arr):
    if arr.shape[1] != 2:
        arr.reshape([-1, 2])
    pts = np.concatenate([arr, arr[0].reshape(1, 2)], axis=0)
    return pts

def phi(z):
    return 1/2 * (1 + erf(z/np.sqrt(2)))

def compute_proba(cb, obst_center, var=1.0):
    """
    return the probability upper bound defined in the article (Page919), equation (17)
    :param cb:
    :param obst_center:
    :return:
    """
    if cb.shape[1] != 2:
        cb = cb.T
    x, y = obst_center[0], obst_center[1]
    max_x, min_x = np.max(cb[:, 0]), np.min(cb[:, 0])
    max_y, min_y = np.max(cb[:, 1]), np.min(cb[:, 1])
    denom = np.sqrt(var)
    return (phi((max_x - x) / denom) - phi((min_x - x) / denom)) * (phi((max_y - y) / denom) - phi((min_y - y) / denom) )

def monteCarlo(x, y, cov, corner, num=50000, ax=None):
    pts = np.random.multivariate_normal([x, y], cov=cov, size=num)
    ax.scatter(pts.T[0], pts.T[1], s=1)
    max_x, min_x = np.max(corner[:, 0]), np.min(corner[:, 0])
    max_y, min_y = np.max(corner[:, 1]), np.min(corner[:, 1])
    # import ipdb
    # ipdb.set_trace()
    def in_robot(z):
        if (z[0] > min_x) and (z[0] < max_x) and (z[1] < max_y) and (z[1] > min_y):
            return 1
        else:
            return 0
    res = 0
    for pt in pts:
        res += in_robot(pt)
    return res / num


class Point(object):

    def __init__(self, x, y):
        self.x, self.y = x, y


# 向量
class Vector(object):

    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y


ZERO = 1e-9


def negative(vector):
    """取反"""
    return Vector(vector.end, vector.start)


def vector_product(vectorA, vectorB):
    '''计算 x_1 * y_2 - x_2 * y_1'''
    return vectorA.x * vectorB.y - vectorB.x * vectorA.y


def is_intersected(A, B, C, D):
    """
    detect AB and CD segment crossing
    :param A:
    :param B:
    :param C:
    :param D:
    :return:
    """

    '''A, B, C, D 为 Point 类型'''

    AC = Vector(A, C)
    AD = Vector(A, D)
    BC = Vector(B, C)
    BD = Vector(B, D)
    CA = negative(AC)
    CB = negative(BC)
    DA = negative(AD)
    DB = negative(BD)

    return (vector_product(AC, AD) * vector_product(BC, BD) <= ZERO) \
           and (vector_product(CA, CB) * vector_product(DA, DB) <= ZERO)