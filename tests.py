import numpy as np


def cost_function(X, y, weight):
    m = X.shape[0]
    h = X.dot(weight)
    loss = h - y
    J = np.sum(loss ** 2) / (2 * m)

    return J


def compute_cost(weight, bias, X, y):
    m = len(y)
    total_cost = 0.0

    for i in range(m):
        f_wb = weight * X[i] + bias
        total_cost += (f_wb - y[i]) ** 2

    cost = total_cost / (2 * m)

    return cost


def gradient_descent(X, y, alpha=0.01, num_iters=1000):
    m = X.shape[0]
    theta = np.zeros((X.shape[1], 1))

    for i in range(num_iters):
        h = X.dot(theta)
        loss = h - y
        gradient = X.T.dot(loss) / m
        theta = theta - alpha * gradient

    return theta


def compute_gradient(w, b, X, y):
    m = len(y)
    dj_dw = 0.0
    dj_db = 0.0

    for i in range(m):
        f_wb = w * X[i] + b
        dj_dw += (f_wb - y[i]) * X[i]
        dj_db += (f_wb - y[i])

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    w = w_init
    b = b_init
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(w, b, X, y)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 100 == 0:  # Print cost every 100 iterations
            cost = compute_cost(w, b, X, y)
            J_history.append(cost)
            print(f"Iteration {i}: Cost {cost}, w {w}, b {b}")

    return w, b, J_history


if __name__ == '__main__':
    X = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    theta = np.array(-0.5)

    print(f'Cost Function Output: {cost_function(X, y, theta)}')
    print(f'Cost Function Output: {compute_cost(theta, 0, X, y)}')

    # Example usage:
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    w_init = 0.0
    b_init = 0.0
    alpha = 0.01
    num_iters = 1000

    w, b, J_history = gradient_descent(X, y, w_init, b_init, alpha, num_iters)
    print(f"Final parameters: w = {w}, b = {b}")
