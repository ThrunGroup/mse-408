#!/usr/bin/env python3
import itertools as it

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import IterableDataset


class InfiniteIterableDataset(IterableDataset):
    """An infinite iterable dataset that simply increases the count."""

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return it.count(0)


class Committor(pl.LightningModule):
    def __init__(
        self,
        a: int = 5,
        b: int = 5,
        p: float = 0.49,
        n_hidden_layers: int = 2,
        n_hidden_width: int = 128,
        hidden_activation: nn.Module = nn.LeakyReLU(),
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.p = p
        self.q = 1 - p
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_width = n_hidden_width
        # input is all values -a->b, excluding -a and b (terminal states)
        layers = [nn.Linear(a + b - 1, n_hidden_width), hidden_activation]
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(n_hidden_width, n_hidden_width), hidden_activation]
        layers += [nn.Linear(n_hidden_width, 1), nn.Sigmoid()]
        self.u = nn.Sequential(*layers)

    def training_step(self, _batch, _batch_idx):
        d = 10e-5
        r_b, r_a = 1, 0
        u = self.p_hit_b()
        # detailed balance loss
        # https://danjenson.github.io/notes/papers/gflownet-foundations#transitions-parameterization-edge-decomposable-loss-detailed-balance

        # without log
        loss = torch.square((u[:-1] * self.p - u[1:] * self.q)).sum()
        loss += torch.square(u[-1] * self.p - r_b * self.q)
        loss += torch.square(u[0] * self.q - r_a * self.p)

        # with log => causes distortion
        # loss = torch.square(
        #     torch.log((d + u[:-1] * self.p) / (d + u[1:] * self.q))
        # ).sum()
        # loss += torch.square(torch.log((d + u[-1] * self.p) / (d + r_b)))  # s->s_b
        # loss += torch.square(torch.log((d + u[0] * self.q) / (d + r_a)))  # s->s_-a

        return loss

    def p_hit_b(self):
        x = torch.arange(self.a + self.b - 1)
        x = nn.functional.one_hot(x, self.a + self.b - 1).float()
        return self.u(x).squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @classmethod
    def train_dataloader(cls):
        # NOTE: triggers the training step, the data isn't used
        return InfiniteIterableDataset()


def p_hit_b(a, b, p):
    """True probability of hitting `b` before `-a` with success probability `p`."""
    n = a + b
    n_1 = np.arange(1, n)
    q_div_p = (1 - p) / p
    return (1 - q_div_p**n_1) / (1 - q_div_p ** (n))


if __name__ == "__main__":
    a, b, p = 3, 3, 0.25
    committor = Committor(a, b, p)
    trainer = Trainer(max_steps=200)
    trainer.fit(committor)
    print("est committor:", committor.p_hit_b().detach().numpy().round(3))
    print("true committor:", p_hit_b(a, b, p).round(3))
