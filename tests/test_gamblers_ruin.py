#!/usr/bin/env python3
import numpy as np

from gamblers_ruin import mcmc, p_hit_b


def test_vanilla_mcmc():
    a, b, s0 = 5, 5, 0
    for p in [0.49, 0.25, 0.1]:
        n, p_hit_b_est, _ = mcmc(a, b, s0, p)
        p_hit_b_true = p_hit_b(a, b, s0, p)
        assert np.isclose(
            p_hit_b_est, p_hit_b_true, rtol=0.1
        ), f"Vanilla MCMC produced an inaccurate estimate for p={p} ({n} samples)!"


def test_exp_tilt_mcmc():
    a, b, s0 = 5, 5, 0
    for p in [0.49, 0.25, 0.1]:
        n, p_hit_b_est, _ = mcmc(a, b, s0, p, epsilon=1e-7)
        p_hit_b_true = p_hit_b(a, b, s0, p)
        assert np.isclose(
            p_hit_b_est, p_hit_b_true, rtol=0.1
        ), f"Exponential Tilt MCMC produced an inaccurate estimate for p={p} ({n} samples)!"
