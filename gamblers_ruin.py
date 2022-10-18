#!/usr/bin/env python3
import argparse
import logging
import multiprocessing as mp
import sys
import time
from typing import Callable

import numpy as np

# TODO: when do you tell sampling to stop? what should your burn in be?

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s: %(message)s", level=logging.DEBUG
)


def p_hit_b(a, b, s0, p):
    """True probability of hitting `b` before `-a` with success probability `p`."""
    if p == 0.5:
        return s0 / (b + a)
    q_div_p = (1 - p) / p
    return (1 - q_div_p**b) / (1 - q_div_p ** (b + a))


class InvalidStateError(Exception):
    pass


def sampler(
    a: int,
    b: int,
    s0: int,
    p: float,
    q: float,
    theta: float,
    sample: Callable[[int, int, int, float, float, float], None],
    queue: mp.Queue,
    stop: mp.Event,
) -> None:
    """A sampling worker for multiprocessing."""
    while not stop.is_set():
        queue.put(sample(a, b, s0, p, q, theta))


def sample(a, b, s0, p, *rest):
    """A vanilla MCMC sampling."""
    t = [s0]
    while t[-1] > -a and t[-1] < b:
        delta = 2 * np.random.binomial(1, p) - 1
        t += [t[-1] + delta]
    return t


def exp_tilt_sample(a, b, s0, p, q, theta):
    """Exponentially tilted MCMC sampling."""
    p_tilt = p * np.exp(theta) / (q * np.exp(-theta) + p * np.exp(theta))
    t = [s0]
    while t[-1] > -a and t[-1] < b:
        x = 2 * np.random.binomial(1, p_tilt) - 1
        t += [t[-1] + x]
    return t


def exp_tilt_weight(t, theta, psi_theta):
    """Importance sampling weight for exponential tilting."""
    s = t[-1] - t[0]
    n = len(t)
    return 1 / np.exp(theta * s - n * psi_theta)


def mcmc(
    a: int = 5,
    b: int = 5,
    s0: int = 0,
    p: float = 0.49,
    n_burn: int = 10000,
    epsilon: float = 1e-6,
    sample: Callable[[int, int, int, float, float, float], list[int]] = sample,
    weight: Callable[[list[int], float, float], float] = lambda t, theta, psi_theta: 1,
    log_every_n: int = 1000,
    n_samplers: int = 1,
) -> tuple[int, float, list[list[int]]]:
    """MCMC sampling for Gambler's Ruin.

    Args:
        a (int): `-a` is the lower bound.
        b (int): `b` is the upper bound.
        s0 (int): The inital state.
        p (float): The probability of success on each bet.
        n_burn (int): Number of burn in samples.
        epsilon (float): Absolute error for estimate that determines when to stop sampling.
        sample (Callable[[int, int, int, float, float, float], list[int]]): A
            function that takes (a, b, s0, p, q, theta) and returns a sample
            trajectory.
        weight (Callable[[list[int], float, float], float]): Calculates the
            importance weight for a sample.
        log_every_n (int): Log estimate every `log_every_n` samples.
        n_samplers (int): Number of samplers to run in parallel.

    Returns:
        tuple[int, float, list[int]]: Returns a tuple of (number of samples,
            estimate for `P(hit b)`, list of trajectories conditional on hitting b).

    :warning: In many cases, multiprocessing is actually slower than a single
    process due to the queuing lock overhead.
    """
    if not (s0 > -a and s0 < b):
        raise InvalidStateError("s0 must lie in (-a, b)")

    q = 1 - p
    theta = np.log(q / p)
    psi_theta = np.log(q * np.exp(-theta) + p * np.exp(theta))
    p_hit_b = 0
    n = 0
    delta = 0
    t_cond_b = []

    if n_samplers > 1:
        queue = mp.Queue()
        stop = mp.Event()
        samplers = []
        for _ in range(n_samplers):
            s = mp.Process(
                target=sampler,
                args=(
                    a,
                    b,
                    s0,
                    p,
                    q,
                    theta,
                    sample,
                    queue,
                    stop,
                ),
            )
            samplers.append(s)
            s.start()
        # this is hacky, but simple for now (used in update_estimate)
        sample = lambda *_: queue.get()

    def update_estimate():
        nonlocal a, b, s0, p, q, theta, psi_theta, p_hit_b, n, delta, t_cond_b, queue, sample
        t = sample(a, b, s0, p, q, theta)
        w = weight(t, theta, psi_theta)
        hit_b = t[-1] == b
        p_hat = w * hit_b
        n += 1
        delta = (p_hat - p_hit_b) / n
        p_hit_b += delta
        if hit_b:
            t_cond_b.append(t)

    while n < n_burn:
        update_estimate()
        if n % log_every_n == 0:
            logging.debug(f"P(hit b): {p_hit_b:0.5f} ({n} samples)")

    while np.abs(delta) > epsilon:
        update_estimate()
        if n % log_every_n == 0:
            logging.debug(f"P(hit b): {p_hit_b:0.5f} ({n} samples)")

    if n_samplers > 1:
        stop.set()
        for s in samplers:
            s.terminate()
            s.join()
        queue.close()
    return n, p_hit_b, t_cond_b


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s0",
        help="starting state, must be in (-a, b), defaults to floor((b-(-a))/2)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-a",
        help="positive integer, -a = lower bound",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-b",
        help="positive integer, b = upper bound",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-p",
        help="probability of success",
        type=float,
        default=0.49,
    )
    parser.add_argument(
        "-nb",
        "--n_burn",
        help="number of burn in samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        help="determines when sampling should stop (|delta| < epsilon)",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "-et",
        "--exponential_tilt",
        help="Use exponential tilting",
        action="store_true",
    )
    parser.add_argument(
        "-logn",
        "--log_every_n",
        help="log estimate every n samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "-ns",
        "--n_samplers",
        help="number of parallel samplers to use",
        type=int,
        default=1,
    )
    args = parser.parse_args(argv[1:])
    if not args.s0:
        args.s0 = int((args.b + -args.a) / 2)
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv)
    weight = lambda t, theta, psi_theta: 1
    if args.exponential_tilt:
        sample = exp_tilt_sample
        weight = exp_tilt_weight
    start = time.time()
    n, p_hit_b, _ = mcmc(
        args.a,
        args.b,
        args.s0,
        args.p,
        args.n_burn,
        args.epsilon,
        sample,
        weight,
        args.log_every_n,
        args.n_samplers,
    )
    print(f"P(hit b): {p_hit_b:0.5f} ({n} samples)")
    print(time.time() - start, "seconds")
