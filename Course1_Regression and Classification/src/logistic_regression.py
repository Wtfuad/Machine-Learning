import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_probability(X, w, b):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return sigmoid(X @ w + b)


def predict_class(X, w, b, threshold=0.5):
    return (predict_probability(X, w, b) >= threshold).astype(int)


def compute_cost(X, y, w, b):
    y = np.asarray(y, dtype=float)
    p = predict_probability(X, w, b)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))


def compute_gradient(X, y, w, b):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)
    errors = predict_probability(X, w, b) - y
    return (X.T @ errors) / m, np.sum(errors) / m


def gradient_descent(X, y, w=None, b=0.0, alpha=0.1, iterations=5000):
    X = np.asarray(X, dtype=float)
    if w is None:
        w = np.zeros(X.shape[1], dtype=float)

    history = []

    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        history.append(compute_cost(X, y, w, b))

    return w, b, history


def compute_cost_regularized(X, y, w, b, lambda_=1.0):
    m = len(y)
    return compute_cost(X, y, w, b) + (lambda_ / (2*m)) * np.sum(w**2)


def compute_gradient_regularized(X, y, w, b, lambda_=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)
    dj_dw, dj_db = compute_gradient(X, y, w, b)
    dj_dw = dj_dw + (lambda_ / m) * w
    return dj_dw, dj_db


def gradient_descent_regularized(X, y, lambda_=1.0, alpha=0.1, iterations=5000):
    X = np.asarray(X, dtype=float)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    history = []

    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient_regularized(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        history.append(compute_cost_regularized(X, y, w, b, lambda_))

    return w, b, history
