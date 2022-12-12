from typing import Iterable

import pytorch_lightning as pl
import torch
from core import Loss, SimpleDiscreteEnvironment
from models import MultiLayerPerceptron
from torch import Tensor


def worker(stream: Iterable):
    for (s, a) in stream:
        pass


class GFN(pl.LightningModule):
    def __init__(self, env: SimpleDiscreteEnvironment, loss: Loss):
        # TODO(danj): add flowmatching / detailed balance loss
        if loss != Loss.TrajectoryBalance:
            raise Exception("Only TrajectoryBalance loss currently supported!")
        self._loss = self._trajectory_balance_loss
        self._pf = MultiLayerPerceptron(env.n_states, env.n_actions)
        self._pb = MultiLayerPerceptron(env.n_states, env.n_actions)
        self._log_z = torch.tensor(0)

    def _trajectory_balance_loss(self, batch):
        pf_logits = self._pf(batch)
        pb_logits = self._pb()
        pass

    def forward(self, batch) -> Tensor:
        return self._pf(batch)

    def training_step(self, batch, batch_idx) -> float:
        return 0.0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
