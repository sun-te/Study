import numpy as np
import matplotlib.pyplot as plt


def threePointsToCircle(points_x, points_y):
    """
    https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
    compute a circle by three points
    return center of the circle (h,k) and its radius
    """
    x1, x2, x3 = points_x
    y1, y2, y3 = points_y
    x12 = x1 - x2
    x13 = x1 - x3
    y12 = y1 - y2
    y13 = y1 - y3
    y31 = y3 - y1
    y21 = y2 - y1
    x31 = x3 - x1
    x21 = x2 - x1
    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)
    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)
    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)
    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
          ((y31) * (x12) - (y21) * (x13))))
    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
          (2 * ((x31) * (y12) - (x21) * (y13))))
    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1)
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c
    r = np.sqrt(sqr_of_r)
    return (h, k ,r)

def data_generator(x=0,y=0,r=1):
    data = np.array([(3.4, 12.47), (2.9, 11.70), (2.0, 11.34), (1.5, 10.57),
    (1.5, 9.65), (1.2, 8.85), (0.8, 8.00), (1.1, 7.13), (1.2, 6.27), (1.8, 5.58),
    (2.3, 4.92), (2.8, 4.13), (3.6, 3.77)])

    x_set = [3.34264,   3.00321,   1.85892,   1.41501 ,  1.65205,   1.26188,   0.58662,   1.04786 ,  1.17716 ,  1.95943  , 2.47808 ,  2.72051  , 3.61075]
    y_set = [12.6027,   11.5714,   11.4748,   10.6471,    9.5825,    8.8355 ,   8.0000 ,   7.1268  ,  6.2446  ,  5.6672 ,  5.0448   , 4.0917  ,  3.8617]

    data = np.array([x_set, y_set]).T
    return data

def f(data, x, y, r):
    center = np.array([x,y])
    ans = np.linalg.norm(data - center, axis=1) - r
    return ans

def mse_erro(data, x, y, r):
    deviation = f(data, x, y ,r)
    error = np.sqrt(np.mean(deviation**2))
    error = np.sum(deviation**2)
    return error

def gradient(data, x, y, r, dx=1e-6, dy=1e-6, dr=1e-6):
    #print("Current [x,y,r]: {} {} {}".format(x,y,r))
    ans = np.zeros([len(data), 3])
    ans[:, 0] = (f(data, x+dx, y, r) - f(data, x-dx, y, r))/ (2*dx)
    ans[:, 1] = (f(data, x, y+dy, r) - f(data, x, y-dy, r))/ (2*dy)
    ans[:, 2] = -1
    return ans

def find_init(data):
    y_max_index = np.argmax(data[:,1])
    y_min_index = np.argmin(data[:,1])
    x_min_index = np.argmin(data[:,0])
    p1 = data[x_min_index]
    p2 = data[y_max_index]
    p3 = data[y_min_index]
    print(p1, p2, p3)
    points = np.array([p1,p2,p3]).T
    x, y, r = threePointsToCircle(points[0], points[1])

    return np.array([x,y,r], dtype=np.float64)



def find_circle(data, n_step=10, init=None):
    if init is None:
        alpha = find_init(data)
        print("Initialize the data with {}".format(alpha))
        plot_result(data, alpha, "Initial Config")
    # alpha = np.array([6,8,3], dtype=np.float64)
    errors = []
    for i in range(n_step):

        e = mse_erro(data, alpha[0], alpha[1], alpha[2])
        print("Square error: {}".format(e))
        errors.append(e)
        grad = gradient(data, alpha[0], alpha[1], alpha[2])
        incremental = np.linalg.inv(np.matmul(grad.T,grad))@(grad.T)@f(data, alpha[0], alpha[1], alpha[2])
        alpha -= incremental

    return alpha, errors

def plot_result(data, alpha, title=""):
    fig = plt.figure(figsize=[5,5])
    plt.scatter(data.T[0], data.T[1], marker="x", c="r")

    theta = np.arange(0, 2 * np.pi, 0.01)
    x = alpha[0] + alpha[2] * np.cos(theta)
    y = alpha[1] + alpha[2] * np.sin(theta)
    plt.plot(x,y)
    plt.scatter(alpha[0],alpha[1])
    plt.annotate("(x:{},y:{},r:{})".format(round(alpha[0],2),round(alpha[1],2),round(alpha[2],2)), (alpha[0],alpha[1]))
    plt.title(title)
    plt.show()

if __name__ == "__main__":

    data = data_generator()
    error = mse_erro(data, 6,8,3)
    alpha, errors = find_circle(data)
    plot_result(data, alpha, "Final Config")
    plt.plot(errors)
    plt.show()
    print("OK")