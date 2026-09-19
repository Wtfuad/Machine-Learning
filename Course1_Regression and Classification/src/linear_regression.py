import numpy as np


def predict(x, w, b):
    x = np.asarray(x, dtype=float)
    return w * x + b


def compute_cost(x, y, w, b):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = len(x)
    predictions = predict(x, w, b)
    errors = predictions - y
    return np.sum(errors ** 2) / (2 * m)


def compute_gradient(x, y, w, b):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = len(x)
    predictions = predict(x, w, b)
    errors = predictions - y

    dj_dw = np.sum(errors * x) / m
    dj_db = np.sum(errors) / m

    return dj_dw, dj_db


def gradient_descent(x, y, w=0.0, b=0.0, alpha=0.01, iterations=1000):
    cost_history = []
    parameter_history = []

    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(x, y, w, b)

        cost_history.append(cost)
        parameter_history.append((w, b))

    return w, b, cost_history, parameter_history
