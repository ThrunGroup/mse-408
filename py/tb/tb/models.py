from typing import Callable

import pytorch_lightning as pl
import torch
from torch import Tensor, nn


class MultiLayerPerceptron(pl.LightningModule):
    """A simple multi-layer perceptron model."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        train: Callable[[pl.LightningModule, Tensor], Tensor],
        n_hidden_layers: int = 2,
        n_hidden_width: int = 256,
        hidden_activation: nn.Module = nn.LeakyReLU(),
    ):
        self._train = train
        layers = [nn.Linear(n_input, n_hidden_width), hidden_activation]
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(n_hidden_width, n_hidden_width), hidden_activation]
        layers += [nn.Linear(n_hidden_width, n_output), nn.Sigmoid()]
        self.m = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.m(x)

    def training_step(self, batch, _batch_idx) -> Tensor:
        return self._train(self, batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
