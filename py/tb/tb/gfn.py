import pytorch_lightning as pl
import torch
from core import IntoTensor, Loss, SimpleDiscreteEnvironment
from data import TrajectoryTensor
from models import MultiLayerPerceptron
from torch import Tensor

# TODO(danj): How to interleave on-policy sampling with train_dataloader
# TODO(danj): Make GFN a PolicyService
# TODO(danj): How do you define flow through abitrary points?


class GFN(pl.LightningModule):
    def __init__(self, env: SimpleDiscreteEnvironment, loss: Loss):
        # TODO(danj): add flowmatching / detailed balance loss
        if loss != Loss.TrajectoryBalance:
            raise Exception("Only TrajectoryBalance loss currently supported!")
        self._loss = self._trajectory_balance_loss
        self._log_pf = MultiLayerPerceptron(env.n_states, env.n_actions)
        self._log_pb = MultiLayerPerceptron(env.n_states, env.n_actions)
        self._log_z = torch.tensor(0)

    def _trajectory_balance_loss(self, batch: TrajectoryTensor) -> Tensor:
        log_f = self._log_pf(batch.states)[batch.actions].gather(0, batch.ids).sum()
        log_b = self._log_pb(batch.states)[batch.actions].gather(0, batch.ids).sum()
        log_r = torch.log(batch.rewards)
        return (self._log_z + log_f - log_b - log_r).pow(2).mean()

    def forward(self, states: list[IntoTensor]) -> Tensor:
        # TODO(danj): do we want flows or probabilities?
        return self._log_pf(torch.tensor([s.into_tensor() for s in states]))

    def training_step(self, batch, _batch_idx) -> Tensor:
        # TODO: dataset class needs to prepare data in tensor format
        return self._loss(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
