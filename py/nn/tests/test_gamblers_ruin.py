#!/usr/bin/env python3
import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer

from gamblers_ruin import Committor, p_hit_b


# TODO(danj): flaky on 0.49 surprisingly
@pytest.mark.parametrize("a", [5, 10])
@pytest.mark.parametrize("b", [5, 10])
@pytest.mark.parametrize("p", [0.10, 0.25, 0.49])
def test_gamblers_ruin_nn(a: int, b: int, p: float):
    torch.manual_seed(0)
    committor = Committor(a, b, p)
    trainer = Trainer(max_steps=1000)
    trainer.fit(committor)
    est = committor.p_hit_b().detach().numpy()
    true = p_hit_b(a, b, p)
    diff = (est - true).round(2)
    assert np.allclose(est, true, atol=0.05), f"diff {diff}"
