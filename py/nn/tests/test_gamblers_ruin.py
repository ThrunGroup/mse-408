#!/usr/bin/env python3
import numpy as np
from pytorch_lightning import Trainer

from gamblers_ruin import Committor, p_hit_b


def test_gamblers_ruin_nn():
    a, b, p = 3, 3, 0.49
    committor = Committor(a, b, p)
    trainer = Trainer(max_steps=100)
    trainer.fit(committor)
    est = committor.p_hit_b().detach().numpy().round(2)
    true = p_hit_b(a, b, p).round(2)
    assert np.allclose(est, true), f"est: {est} != true: {true}"
