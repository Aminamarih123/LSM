import time
import numpy as np


def monte_carlo_standard_error(samples):
   
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 1:
        raise ValueError("samples must be a 1D array.")
    if samples.size < 2:
        raise ValueError("samples must contain at least two observations.")

    return np.std(samples, ddof=1) / np.sqrt(samples.size)


def absolute_error(estimate, reference):
   
    return abs(float(estimate) - float(reference))


def relative_error(estimate, reference):
 
    estimate = float(estimate)
    reference = float(reference)

    if reference == 0.0:
        return np.nan

    return abs(estimate - reference) / abs(reference)


def policy_exercise_frequency(exercise_times, n_steps=None):
    
    exercise_times = np.asarray(exercise_times)

    if exercise_times.ndim != 1:
        raise ValueError("exercise_times must be a 1D array.")
    if exercise_times.size == 0:
        raise ValueError("exercise_times cannot be empty.")

    if n_steps is None:
        n_steps = int(np.max(exercise_times))

    return np.mean(exercise_times < n_steps)


def exercise_time_distribution(exercise_times):
    
    exercise_times = np.asarray(exercise_times)

    if exercise_times.ndim != 1:
        raise ValueError("exercise_times must be a 1D array.")
    if exercise_times.size == 0:
        raise ValueError("exercise_times cannot be empty.")

    unique, counts = np.unique(exercise_times, return_counts=True)
    return {int(t): int(c) for t, c in zip(unique, counts)}


def runtime_stats(start_time, end_time=None):
   
    if end_time is None:
        end_time = time.perf_counter()

    elapsed = float(end_time - start_time)

    return {
        "start_time": float(start_time),
        "end_time": float(end_time),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
    }
