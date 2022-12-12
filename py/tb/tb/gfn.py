from typing import Iterable

import pytorch_lightning as pl
import torch
from core import IntoTensor, Loss, SimpleDiscreteEnvironment, Trajectories
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

    def _trajectory_balance_loss(self, trajectories: Trajectories) -> Tensor:
        trajs = []
        actions = []
        rewards = []
        for trajectory in trajectories:
            actions = []
            states = []
            for step in trajectory:
                actions.append(step.transition.action)
                states.append(step.transition.state.into_tensor())
        log_f = self._pf(trajectories)
        log_b = self._pb(trajectories)
        log_r = torch.log(rewards)
        return torch.square(self._log_z + log_f - log_b - log_r)

    def forward(self, states: list[IntoTensor]) -> Tensor:
        return self._pf(torch.tensor([s.into_tensor() for s in states]))

    def training_step(self, batch, _batch_idx) -> Tensor:
        return self._loss(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
