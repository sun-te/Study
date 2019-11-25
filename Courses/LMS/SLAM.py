import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace as tt


def generate_points(noise=True):
    # res = np.zeros((7,2))
    #
    # d_theta = np.pi/3
    # for i in range(1, 7):
    #     d = 10 + int(noise) * np.random.normal()
    #     theta = d_theta * (i-1 + int(noise) * np.random.normal(loc=0,scale=0.1))
    #     res[i] = res[i-1] + d * np.array([np.cos(theta), np.sin(theta)])
    #

    res = np.array([[0.0, 0.0],
                    [10.9923479911905, 0.15767317404539163],
                    [16.2925265546885, 7.841513074571804],
                    [10.158026746497443, 15.869357420429035],
                    [-1.513592254623779, 18.01018237604731],
                    [-6.782074061174459, 9.004213972082178],
                    [-1.5373954470807671, -0.38971623634690467]])
    return res

def plot_result(data, c="r", title=""):
    fig = plt.figure(figsize=[5,5])
    plt.scatter(data.T[0], data.T[1], marker="x", c=c)
    plt.plot(data.T[0], data.T[1], c=c)
    plt.title(title)
    plt.show()


def incremental_step(points):
    """
    Get P_{i+1, i}
    :param points:
    :return:
    """
    return points[1:] - points[:-1]

def inv_vector(arr):
    """
    arr: x, y, theta
    :param arr:
    :return: the inverse of a vector
    """
    assert len(arr) == 3 or arr.shape == (1,3)
    x, y, theta = arr
    ans = [-x*np.cos(theta)-y*np.sin(theta),
           x*np.sin(theta) -y*np.cos(theta),
           -theta]
    return np.array(ans)

def round_plus(arr1, arr2):
    x1, y1, t1 = arr1
    x2, y2, t2 = arr2
    ans = [x2*np.cos(t1) - y2*np.sin(t1)+x1,
           x2*np.sin(t1) + y2*np.cos(t1)+y1,
           t1+t2]
    return np.array(ans)




if __name__ == "__main__":
    data = generate_points()
    steps = incremental_step(points=data)
    inv_vector([1,2,3])
    mm = round_plus([12,3,1], [3,4,-1])
    tt()
    plot_result(data)
    print("OK")