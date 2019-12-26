import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
