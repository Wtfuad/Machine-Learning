import numpy as np

def predict(X, w, b):
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b

def compute_cost(X, y, w, b):
    y = np.asarray(y, dtype=float)
    errors = predict(X, w, b) - y
    return np.sum(errors**2) / (2 * len(y))

def compute_gradient(X, y, w, b):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    errors = predict(X, w, b) - y
    m = len(y)
    return (X.T @ errors) / m, np.sum(errors) / m

def gradient_descent(X, y, w, b, alpha=0.01, iterations=1000):
    history = []
    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        history.append(compute_cost(X, y, w, b))
    return w, b, history

def zscore_normalize(X):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma
