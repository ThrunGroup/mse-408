#!/usr/bin/env python3
import argparse
import sys

import numpy as np

# MCMC Questions:
# 1. When do you stop sampling?
#   - You reach N valid samples
#   - p(X in A) approaches it's true probability
# GFN Questions:
# 1. How do you design the reward function? R(x)=1 for X in A and 0 otherwise?
# 2. When do you stop training?


def mcmc(starting_state, p_success, lower, upper):
    if not starting_state > lower and starting_state < upper:
        raise InvalidStateError("starting state must lie between [lower, upper]")


def sample_naive(_state, p_success):
    return np.random.binomial(1, p_success)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--starting_state",
        help="starting state, must be in [lower, upper], defaults to floor((upper-lower)/2)",
        dtype=int,
    )
    parser.add_argument(
        "-p",
        "--p_success",
        help="probability of success",
        dtype=float,
        default=0.49,
    )
    parser.add_argument(
        "-l",
        "--lower",
        help="lower bound",
        dtype=int,
        default=0,
    )
    parser.add_argument(
        "-u",
        "--upper",
        help="upper bound",
        dtype=int,
        default=100,
    )
    args = parser.parse_args(argv[1:])
    if not args.starting_state:
        args.starting_state = int((args.upper - args.lower) / 2)
    return args


class InvalidStateError(Exception):
    pass


if __name__ == "__main__":
    args = parse_args(sys.argv)
    mcmc(args.starting_state, args.p_success, args.lower, args.upper)
