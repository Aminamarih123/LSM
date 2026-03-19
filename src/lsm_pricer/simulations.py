import numpy as np


class GBM_Simulator:
    def __init__(self, S0, r, sigma, maturity, n_steps, n_paths):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.maturity = maturity
        self.n_steps = n_steps
        self.n_paths = n_paths

    def validate_gbm_inputs(self):
        if self.S0 <= 0.0:
            raise ValueError("S0 must be strictly positive.")
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative.")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be strictly positive.")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1.")
        if self.n_paths < 1:
            raise ValueError("n_paths must be at least 1.")

    def simulate_gbm_paths(self, antithetic=False, seed=None):
        self.validate_gbm_inputs()

        S0 = self.S0
        r = self.r
        sigma = self.sigma
        maturity = self.maturity
        n_steps = self.n_steps
        n_paths = self.n_paths

        rng = np.random.default_rng(seed)
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)

        if antithetic:
            n_half = (n_paths + 1) // 2
            z_half = rng.standard_normal(size=(n_half, n_steps))
            z = np.vstack((z_half, -z_half))
            z = z[:n_paths, :]
        else:
            z = rng.standard_normal(size=(n_paths, n_steps))

        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * sqrt_dt * z
        log_increments = drift + diffusion

        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))

        return paths
