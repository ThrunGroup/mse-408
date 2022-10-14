#!/usr/bin/env python3
import argparse
import logging
import multiprocessing as mp
import sys

import numpy as np

# GFN Questions:
# 1. How do you design the reward function? R(x)=1 for X in A and 0 otherwise?
# 2. When do you stop training?


def mcmc(
    starting_state: int = 5,
    lower: int = 0,
    upper: int = 10,
    p_success: float = 0.49,
    n_burn_in: int = int(1e4),
    epsilon: float = 1e-6,
    log_every_n: int = int(100),
) -> tuple[int, float, list[list[int]]]:
    if not starting_state > lower and starting_state < upper:
        raise InvalidStateError("starting state must lie between [lower, upper]")
    is_win, traj = sample_trajectory(starting_state, lower, upper, p_success)
    n = 1
    p_win = is_win
    traj_cond_win = []
    if is_win:
        traj_cond_win += [traj]
    delta = 0
    while n < n_burn_in:
        is_win, traj = sample_trajectory(starting_state, lower, upper, p_success)
        n += 1
        delta = (is_win - p_win) / n
        p_win += delta
        if is_win:
            traj_cond_win += [traj]
        if n % log_every_n == 0:
            logging.debug(f"P(WIN)={p_win:.4f}")
    while delta > epsilon:
        is_win, traj = sample_trajectory(starting_state, lower, upper, p_success)
        n += 1
        delta = (is_win - p_win) / n
        p_win += delta
        if is_win:
            traj_cond_win += [traj]
        if n % log_every_n == 0:
            logging.debug(f"P(WIN)={p_win:.4f}")
    return n, p_win, traj_cond_win


def sample_trajectory(starting_state, lower, upper, p_success):
    traj = [starting_state]
    while traj[-1] > lower and traj[-1] < upper:
        delta = (np.random.binomial(1, p_success) - 0.5) * 2
        traj += [traj[-1] + delta]
    return int(traj[-1] == upper), traj


def exp_tilt_mcmc(
    starting_state: int = 5,
    lower: int = 0,
    upper: int = 10,
    p_success: float = 0.49,
    n_burn_in: int = int(1e4),
    epsilon: float = 1e-6,
    log_every_n: int = int(100),
) -> tuple[int, float, list[list[int]]]:
    if not starting_state > lower and starting_state < upper:
        raise InvalidStateError("starting state must lie between [lower, upper]")
    traj_cond_win = []
    is_win, traj = exp_tilt_sample_trajectory(starting_state, lower, upper, p_success)
    if is_win:
        traj_cond_win += [traj]
    d = upper - starting_state
    n = 1
    q = 1 - p_success
    theta = np.log(q / p_success)  # analytically optimal theta
    # theta = np.log(p_success / q)  # analytically optimal theta
    psi_theta = np.log(q + p_success * np.exp(theta))
    p_win = is_win = is_win / np.exp(theta * d - len(traj) * psi_theta)
    delta = 0
    while n < n_burn_in:
        is_win, traj = exp_tilt_sample_trajectory(
            starting_state, lower, upper, p_success
        )
        if is_win:
            traj_cond_win += [traj]
        n += 1
        # reweight by importance sampling weight
        is_win = is_win / np.exp(theta * d - len(traj) * psi_theta)
        delta = (is_win - p_win) / n
        p_win += delta
        if n % log_every_n == 0:
            logging.debug(f"P(WIN)={p_win:.4f}")
    while delta > epsilon:
        is_win, traj = exp_tilt_sample_trajectory(
            starting_state, lower, upper, p_success
        )
        if is_win:
            traj_cond_win += [traj]
        n += 1
        # reweight by importance sampling weight
        is_win = is_win / np.exp(theta * d - len(traj) * psi_theta)
        delta = (is_win - p_win) / n
        p_win += delta
        if n % log_every_n == 0:
            logging.debug(f"P(WIN)={p_win:.4f}")
    return n, p_win, traj_cond_win


def exp_tilt_sample_trajectory(starting_state, lower, upper, p_success):
    traj = [starting_state]
    q = 1 - p_success
    theta = np.log(q / p_success)  # analytically optimal theta
    # theta = np.log(p_success / q)  # analytically optimal theta
    pt = p_success * np.exp(theta)
    p = pt / (q + pt)
    while traj[-1] > lower and traj[-1] < upper:
        delta = 2 * np.random.binomial(1, p) - 1
        traj += [traj[-1] + delta]
    return int(traj[-1] == upper), traj


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--starting_state",
        help="starting state, must be in [lower, upper], defaults to floor((upper-lower)/2)",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-l",
        "--lower",
        help="lower bound",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-u",
        "--upper",
        help="upper bound",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-p",
        "--p_success",
        help="probability of success",
        type=float,
        default=0.49,
    )
    parser.add_argument(
        "-nb",
        "--n_burn_in",
        help="number of burn in samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        help="determines when sampling should stop (delta < epsilon)",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "-logn",
        "--log_every_n",
        help="log estimate every n samples",
        type=int,
        default=10000,
    )
    args = parser.parse_args(argv[1:])
    if not args.starting_state:
        args.starting_state = int((args.upper - args.lower) / 2)
    return args


class InvalidStateError(Exception):
    pass


if __name__ == "__main__":
    args = parse_args(sys.argv)
    _, p_win, _ = mcmc(
        args.starting_state,
        args.lower,
        args.upper,
        args.p_success,
        args.n_burn_in,
        args.epsilon,
        args.log_every_n,
    )
    print(f"mcmc: P(WIN)={p_win}")
    _, p_win, _ = exp_tilt_mcmc(
        args.starting_state,
        args.lower,
        args.upper,
        args.p_success,
        args.n_burn_in,
        args.epsilon,
        args.log_every_n,
    )
    print(f"exp tilt mcmc: P(WIN)={p_win}")
