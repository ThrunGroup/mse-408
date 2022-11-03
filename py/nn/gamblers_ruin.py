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
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.p = p
        self.q = 1 - p
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_width = n_hidden_width
        layers = [nn.Linear(a + b - 1, n_hidden_width), nn.LeakyReLU()]
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(n_hidden_width, n_hidden_width), nn.LeakyReLU()]
        layers += [nn.Linear(n_hidden_width, 1), nn.Sigmoid()]
        self.u = nn.Sequential(*layers)

    def training_step(self, _batch, _batch_idx):
        # TODO: alternative is to create the full u-vector and replace first and last values
        u = self.p_hit_b()
        loss = torch.square(torch.log(u[:-1] * self.p / (u[1:] * self.q))).sum()
        self.log("loss", loss)
        return loss

    def p_hit_b(self):
        x = torch.arange(self.a + self.b - 1)
        x = nn.functional.one_hot(x, self.a + self.b - 1).float()
        u = self.u(x).squeeze()
        r = torch.empty(self.a + self.b + 1)
        r[0], r[-1] = 0.0001, 1.0
        r[1:-1] = u
        return r

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @classmethod
    def train_dataloader(cls):
        """Returns a :class:`~InfiniteIterableDataset`.

        Because samples are generated in this model and PyTorch Lightning
        requires some form of DataLoader, this is a dummy class that counts
        forever. These values are ignored in :meth:`training_step`.

        Returns:
            InfiniteIterableDataset: An infinite iterable dataset that
                counts up forever.
        """
        return InfiniteIterableDataset()


def p_hit_b(a, b, p):
    """True probability of hitting `b` before `-a` with success probability `p`."""
    n = a + b
    n_1 = np.arange(n + 1)
    q_div_p = (1 - p) / p
    return (1 - q_div_p**n_1) / (1 - q_div_p ** (n))


if __name__ == "__main__":
    a, b, s0, p = 5, 5, 0, 0.49
    committor = Committor(a, b, p)
    trainer = Trainer(max_steps=5)
    trainer.fit(committor)
    print(committor.p_hit_b())
    print(p_hit_b(a, b, p))
