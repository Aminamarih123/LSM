import numpy as np


def fit_ols(X, y, return_diagnostics=False):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    if not return_diagnostics:
        return beta

    diagnostics = {
        "residuals": residuals,
        "rank": rank,
        "singular_values": s,
        "condition_number": s[0] / s[-1] if len(s) > 0 else np.inf,
    }

    return beta, diagnostics


def predict_ols(X, beta):
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if beta.ndim != 1:
        raise ValueError("beta must be a 1D array.")
    if X.shape[1] != beta.shape[0]:
        raise ValueError("X and beta dimensions do not align.")

    return X @ beta


def fit_ridge(X, y, alpha):
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_features = X.shape[1]

    XtX = X.T @ X
    XtX_reg = XtX + alpha * np.eye(n_features)
    Xty = X.T @ y

    beta = np.linalg.solve(XtX_reg, Xty)
    return beta
