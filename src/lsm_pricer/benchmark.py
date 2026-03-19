import math
import numpy as np


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_put_price(s0, strike, r, sigma, maturity):
   
    if s0 <= 0.0:
        raise ValueError("s0 must be strictly positive.")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive.")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive.")

    if sigma == 0.0:
        discounted_strike = strike * math.exp(-r * maturity)
        return max(discounted_strike - s0, 0.0)

    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(s0 / strike) + (r + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    put_price = strike * math.exp(-r * maturity) * _normal_cdf(-d2) - s0 * _normal_cdf(-d1)
    return put_price


def crr_american_put(s0, strike, r, sigma, maturity, n_steps):
    if s0 <= 0.0:
        raise ValueError("s0 must be strictly positive.")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive.")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")

    dt = maturity / n_steps
    if sigma == 0.0:
        # Deterministic path under risk-neutral dynamics
        values = []
        for j in range(n_steps + 1):
            s_t = s0 * math.exp(r * dt * j)
            values.append(max(strike - s_t, 0.0))
        return max(values)

    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            "Risk-neutral probability is outside [0, 1]. "
            "Try increasing n_steps or check parameters."
        )

    # Terminal stock prices S(T, j), j = 0..n_steps
    stock_prices = np.array(
        [s0 * (u ** j) * (d ** (n_steps - j)) for j in range(n_steps + 1)],
        dtype=float,
    )
    option_values = np.maximum(strike - stock_prices, 0.0)

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        stock_prices = stock_prices[:-1] / d
        continuation = disc * (
            p * option_values[1:] + (1.0 - p) * option_values[:-1]
        )
        exercise = np.maximum(strike - stock_prices, 0.0)
        option_values = np.maximum(exercise, continuation)

    return float(option_values[0])
