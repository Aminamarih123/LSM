import numpy as np
from abc import ABC, abstractmethod


class Payoff(ABC):
    @abstractmethod
    def intrinsic_value(self, state, t):
        raise NotImplementedError

    @abstractmethod
    def terminal_payoff(self, state):
        raise NotImplementedError

class AmericanPut(Payoff):
    def __init__(self, strike):
        if strike <= 0.0:
            raise ValueError("strike must be strictly positive.")
        self.strike = strike

    def intrinsic_value(self, state, t):
        s_t = state[:, t] if np.ndim(state) == 2 else state
        return np.maximum(self.strike - s_t, 0)

    def terminal_payoff(self, state):
        s_T = state[:, -1] if np.ndim(state) == 2 else state
        return np.maximum(self.strike - s_T, 0.0)

class BermudanCall(Payoff):
    def __init__(self, strike, exercise_dates=None):
        if strike <= 0.0:
            raise ValueError("strike must be strictly positive.")
        self.strike = strike
        self.exercise_dates = set(exercise_dates) if exercise_dates is not None else None

    def can_exercise(self, t) -> bool:
        if self.exercise_dates is None:
            return True
        return t in self.exercise_dates

    def intrinsic_value(self, state, t):
        s_t = state[:, t] if np.ndim(state) == 2 else state

        if not self.can_exercise(t):
            return np.zeros_like(s_t, dtype=float)

        return np.maximum(s_t - self.strike, 0)

    def terminal_payoff(self, state):
        s_T = state[:, -1] if np.ndim(state) == 2 else state
        return np.maximum(s_T - self.strike, 0)


class BermudanAsianCall(BermudanCall):
    def intrinsic_value(self, state, t):
        if np.ndim(state) != 2:
            raise ValueError("BermudanAsianCall requires full path state of shape (n_paths, n_times).")

        running_avg = np.mean(state[:, : t + 1], axis=1)

        if not self.can_exercise(t):
            return np.zeros_like(running_avg, dtype=float)

        return np.maximum(running_avg - self.strike, 0)

    def terminal_payoff(self, state):
        if np.ndim(state) != 2:
            raise ValueError("BermudanAsianCall requires full path state of shape (n_paths, n_times).")

        avg_T = np.mean(state, axis=1)
        return np.maximum(avg_T - self.strike, 0)

