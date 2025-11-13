from __future__ import annotations

import numpy as np

from gaq.analysis.hypercube_walks import hitting_time_exact, simulate_hitting_time


def test_exact_hitting_time_small_hypercubes() -> None:
    assert hitting_time_exact(1, (0,), (1,)) == 1.0
    assert hitting_time_exact(2, (0, 0), (1, 1)) == 4.0
    assert abs(hitting_time_exact(3, (0, 0, 0), (1, 1, 1)) - 10.0) < 1e-9


def test_simulation_matches_exact_expectation() -> None:
    rng = np.random.default_rng(123)
    result = simulate_hitting_time(3, (0, 0, 0), (1, 1, 1), trials=2000, rng=rng)
    assert abs(result.simulated_mean - result.expected_steps) < 0.5
