import numpy as np


def polynomial_basis(x, degree):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    return np.column_stack([x**d for d in range(degree + 1)])


def laguerre_basis(x, degree):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    n = x.shape[0]
    basis = np.empty((n, degree + 1), dtype=float)

    basis[:, 0] = 1.0

    if degree >= 1:
        basis[:, 1] = 1.0 - x

    for k in range(1, degree):
        basis[:, k + 1] = ((2 * k + 1 - x) * basis[:, k] - k * basis[:, k - 1]) / (k + 1)
      
    return basis

