#!/usr/bin/env python3
import numpy as np

from gamblers_ruin import exp_tilt_mcmc, mcmc


def test_mcmc():
    starting_state = 5
    lower = 0
    upper = 10
    p_success = 0.49
    _, est_p_win, _ = mcmc(starting_state, lower, upper, p_success)
    true_p_win = p_win(starting_state, upper, p_success)
    assert np.isclose(
        est_p_win - true_p_win, 0, atol=0.01
    ), "Vanilla MCMC produced an inaccurate estimate!"


def test_exp_tilt_mcmc():
    starting_state = 5
    lower = 0
    upper = 10
    p_success = 0.49
    _, est_p_win, _ = exp_tilt_mcmc(starting_state, lower, upper, p_success)
    true_p_win = p_win(starting_state, upper, p_success)
    assert np.isclose(
        est_p_win - true_p_win, 0, atol=0.01
    ), "Exponential Tilt MCMC produced an inaccurate estimate!"


def p_win(starting_state, upper, p_success):
    if p_success == 0.5:
        return starting_state / upper
    q_div_p = (1 - p_success) / p_success
    return (1 - q_div_p**starting_state) / (1 - q_div_p**upper)
