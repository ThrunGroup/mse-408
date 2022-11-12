#!/usr/bin/env python3
import numpy as np
import pytest
from pytorch_lightning import Trainer
from scipy import stats

from posterior import GFN, Param, PosteriorEnv, flow_matching_loss


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


def test_posterior_env_bernoulli():
    p, batch_size = 0.25, 2
    param_p = Param(name="p", min=0.01, max=0.99, n=99)
    env = PosteriorEnv(
        param_p,
        data=np.zeros(batch_size),
        likelihood=lambda x, p: stats.bernoulli.pmf(x, p).prod(),
        batch_size=batch_size,
    )
    # NOTE: 1 is the stop action
    # stop in s0
    action = 1
    step = env.step(env.s0, action)
    parents, _ = env.parent_transitions(step.state, action)
    assert step.next_state == env.s0, "Incorrect step!"
    assert np.isclose(step.reward, 0.9801), "Incorrect reward!"
    assert len(parents) == 1, "state should have only 1 parent!"
    # step to s1
    action = 0
    step = env.step(env.s0, action)
    parents, _ = env.parent_transitions(step.next_state, action)
    assert step.next_state == np.array([1]), "Incorrect step!"
    assert np.isclose(step.reward, 0), "Incorrect reward!"
    assert len(parents) == 1, "state should have only 1 parent!"
    # stop in s1
    action = 1
    step = env.step(step.next_state, action)
    parents, _ = env.parent_transitions(step.next_state, action)
    assert step.state == step.next_state, "Incorrect step!"
    assert np.isclose(step.reward, 0.9604), "Incorrect reward!"
    assert len(parents) == 1, "state should have only 1 parent!"
    # test state_to_tensor
    t = env.states_to_tensors([np.array([0]), np.array([1])])
    assert t.sum() == 2, "Incorrect tensor from state!"
    assert t[0][0] == 1, "Incorrect tensor from state!"
    assert t.sum() == 2, "Incorrect tensor from state!"
    assert t[1][1] == 1, "Incorrect tensor from state!"

    np.random.seed(0)
    batch_size = 16
    env.data = stats.bernoulli.rvs(p, size=batch_size)
    gfn = GFN(env)
    # trainer = Trainer(max_steps=1000)
    # trainer.fit(gfn)
    # samples = gfn.sample(n=1000).numpy()
    # assert np.isclose(samples.mean(), p), "Estimated p is incorrect!"
    # assert np.isclose(samples.var(), p * (1 - p)), "Estimated variance is incorrect!"


def test_posterior_env_normal():
    np.random.seed(0)
    mu, sigma, batch_size = 2, 2, 32
    param_mu = Param(name="mu", min=-16, max=16, n=100)
    param_sigma = Param(name="sigma", min=0.01, max=10, n=100)
    env = PosteriorEnv(
        param_mu,
        param_sigma,
        data=stats.norm.rvs(mu, sigma, batch_size),
        likelihood=lambda x, mu, sigma: stats.norm.pdf(x, mu, sigma).prod(),
        batch_size=batch_size,
    )
    assert env.s0[0] == 0, "Incorrect s0!"
    assert env.s0[1] == 0, "Incorrect s0!"
