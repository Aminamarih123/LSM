import numpy as np

from LSM.basis import polynomial_basis, laguerre_basis
from LSM.regression import fit_ols, predict_ols, fit_ridge


class LSMEngine:
    def __init__(self, payoff, basis_type="polynomial", degree=2, regression_type="ols", ridge_alpha=1e-8):
        self.payoff = payoff
        self.basis_type = basis_type
        self.degree = degree
        self.regression_type = regression_type
        self.ridge_alpha = ridge_alpha

        self.models_ = {}
        self.exercise_policy_ = {}
        self.cashflows_ = None
        self.exercise_times_ = None
        self.fitted_ = False
        self.n_steps_ = None
        self.dt_ = None
        self.discount_ = None

    def _basis(self, x):
        if self.basis_type == "polynomial":
            return polynomial_basis(x, self.degree)
        if self.basis_type == "laguerre":
            return laguerre_basis(x, self.degree)
        raise ValueError(f"Unsupported basis_type: {self.basis_type}")

    def _fit_regression(self, X, y):
        if self.regression_type == "ols":
            return fit_ols(X, y)
        if self.regression_type == "ridge":
            return fit_ridge(X, y, alpha=self.ridge_alpha)
        raise ValueError(f"Unsupported regression_type: {self.regression_type}")

    def fit(self, paths, r, maturity):
        paths = np.asarray(paths, dtype=float)

        if paths.ndim != 2:
            raise ValueError("paths must be a 2D array of shape (n_paths, n_times).")

        n_paths, n_times = paths.shape
        n_steps = n_times - 1
        if n_steps < 1:
            raise ValueError("paths must contain at least two time points.")

        dt = maturity / n_steps
        discount = np.exp(-r * dt)

        self.n_steps_ = n_steps
        self.dt_ = dt
        self.discount_ = discount
        self.models_ = {}
        self.exercise_policy_ = {}

        cashflows = self.payoff.terminal_payoff(paths).copy()
        exercise_times = np.full(n_paths, n_steps, dtype=int)

        for t in range(n_steps - 1, 0, -1):
            intrinsic = self.payoff.intrinsic_value(paths, t)
            itm = intrinsic > 0.0

            cashflows *= discount

            if not np.any(itm):
                self.models_[t] = None
                self.exercise_policy_[t] = {
                    "itm_indices": np.array([], dtype=int),
                    "exercise_indices": np.array([], dtype=int),
                }
                continue

            x_t = paths[itm, t]
            X = self._basis(x_t)
            y = cashflows[itm]

            beta = self._fit_regression(X, y)
            continuation = predict_ols(X, beta)

            exercise_now = intrinsic[itm] > continuation
            itm_indices = np.where(itm)[0]
            exercise_indices = itm_indices[exercise_now]

            cashflows[exercise_indices] = intrinsic[exercise_indices]
            exercise_times[exercise_indices] = t

            self.models_[t] = beta
            self.exercise_policy_[t] = {
                "itm_indices": itm_indices,
                "exercise_indices": exercise_indices,
            }

        self.cashflows_ = cashflows
        self.exercise_times_ = exercise_times
        self.fitted_ = True

        return self

    def price(self):
        if not self.fitted_:
            raise RuntimeError("Call fit(...) before price().")
        return np.mean(self.cashflows_)

    def price_out_of_sample(self, paths, r, maturity):
        if not self.fitted_:
            raise RuntimeError("Call fit(...) before price_out_of_sample(...).")

        paths = np.asarray(paths, dtype=float)

        if paths.ndim != 2:
            raise ValueError("paths must be a 2D array of shape (n_paths, n_times).")

        n_paths, n_times = paths.shape
        n_steps = n_times - 1

        if n_steps != self.n_steps_:
            raise ValueError("Out-of-sample paths must have the same number of time steps as fitted paths.")

        dt = maturity / n_steps
        discount = np.exp(-r * dt)

        alive = np.ones(n_paths, dtype=bool)
        values = np.zeros(n_paths, dtype=float)

        for t in range(1, n_steps):
            intrinsic = self.payoff.intrinsic_value(paths, t)
            itm_alive = alive & (intrinsic > 0.0)

            beta = self.models_.get(t)

            if beta is None or not np.any(itm_alive):
                continue

            x_t = paths[itm_alive, t]
            X = self._basis(x_t)
            continuation = predict_ols(X, beta)

            exercise_now = intrinsic[itm_alive] > continuation
            idx = np.where(itm_alive)[0]
            exercise_idx = idx[exercise_now]

            values[exercise_idx] = intrinsic[exercise_idx] * np.exp(-r * dt * t)
            alive[exercise_idx] = False

        terminal_idx = np.where(alive)[0]
        terminal_payoff = self.payoff.terminal_payoff(paths[terminal_idx])
        values[terminal_idx] = terminal_payoff * np.exp(-r * maturity)

        return np.mean(values)
