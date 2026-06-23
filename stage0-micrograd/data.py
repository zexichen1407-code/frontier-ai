"""
Stage 0 data + scoring helpers  (provided — pure data utilities, no autograd).
You may read this; you don't need to edit it.
"""
import numpy as np
from sklearn.datasets import make_moons


def load_two_moons(n=200, noise=0.1, seed=42):
    """Return X (list of [x1, x2]) and y (list of 0/1), features standardized."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.tolist(), y.tolist()


def accuracy(pred_labels, y_true):
    """pred_labels and y_true: lists of 0/1. Returns the fraction correct."""
    return sum(int(p == t) for p, t in zip(pred_labels, y_true)) / len(y_true)


def plot_decision_boundary(score_fn, X, y, path="boundary.png"):
    """score_fn([x1, x2]) -> float, where >= 0 means class 1. Saves a PNG (nice-to-have)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = np.array(X)
    y = np.array(y)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
    Z = np.array(
        [1 if score_fn([float(a), float(b)]) >= 0 else 0
         for a, b in zip(xx.ravel(), yy.ravel())]
    ).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k")
    plt.title("Stage 0 — two-moons decision boundary")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")
