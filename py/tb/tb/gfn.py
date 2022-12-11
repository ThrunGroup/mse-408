from typing import Iterable

import pytorch_lightning as pl
import torch
from core import Loss, SimpleDiscreteEnvironment
from models import MultiLayerPerceptron


def worker(stream: Iterable):
    for (s, a) in stream:
        pass


class GFN(pl.LightningModule):
    def __init__(self, env: SimpleDiscreteEnvironment, loss: Loss):
        # TODO(danj): add flowmatching / detailed balance loss
        if loss != Loss.TrajectoryBalance:
            raise Exception("Only TrajectoryBalance loss currently supported!")
        self._pf = MultiLayerPerceptron(env.n_states, env.n_actions)
        self._pb = MultiLayerPerceptron(env.n_states, env.n_actions)
        self._log_z = torch.tensor(0)
        self._loss = self._trajectory_balance_loss

    def _trajectory_balance_loss(self):
        pass
