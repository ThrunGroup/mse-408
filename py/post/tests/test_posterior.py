#!/usr/bin/env python3
import numpy as np
import pytest
from pytorch_lightning import Trainer

from posterior import PosteriorEnv, PosteriorGFN, flow_matching_loss


@pytest.mark.parametrize(
    "inflow,outflow,delta,loss",
    [
        ([1], [1], 0, 0.0),
        ([1], [1], 0.1, 0.0),
        ([1, 2, 3], [1, 2, 4], 0, 0.027587),
        ([1, 2, 3], [1, 2, 4], 0.1, 0.026056),
    ],
)
def test_flow_matching_loss(inflow, outflow, delta, loss):
    est_loss = flow_matching_loss(inflow, outflow, delta).numpy()
    assert np.isclose(est_loss, loss), "Incorrect flow matching loss!"


@pytest.mark.parameterize("p", [0.1, 25, 0.49])
def test_bernoulli_posterior(p):
    env = PosteriorEnv()
    gfn = PosteriorGFN(env)
    trainer = Trainer(max_steps=1000)
    trainer.fit(gfn)
    samples = gfn.sample(n=1000).numpy()
    assert np.isclose(samples.mean(), p), "Estimated p is incorrect!"
    assert np.isclose(samples.var(), p * (1 - p)), "Estimated variance is incorrect!"
