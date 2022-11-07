#!/usr/bin/env python3
import numpy as np
import pytest

from posterior import flow_matching_loss


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
