#!/usr/bin/env python3
import pytorch_lightning as pl
import torch
from torch import nn


class Committor(pl.LightningModule):
    def __init__(
        self,
        a: int = 5,
        b: int = 5,
        p: float = 0.49,
        n_hidden_layers: int = 2,
        n_hidden_width: int = 128,
    ):
        self.a = a
        self.b = b
        self.p = torch.tensor([p, 1 - p])
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_width = n_hidden_width
        layers = [nn.Linear(a + b - 1, n_hidden_width), nn.LeakyReLU()]
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(n_hidden_width, n_hidden_width), nn.LeakyReLU()]
        layers += [nn.Linear(n_hidden_width, 2), nn.Softmax()]
        self.u = nn.Sequential(*layers)

    def training_step(self, batch, _batch_idx):
        x = nn.functional.one_hot(batch, self.a + self.b - 1)
        u = self.u(x)
        u_from = (u * self.p).sum(axis=1)
        u_to = u.sum(axis=1)
        loss = torch.square(torch.log(u_from / u_to))
        self.log("loss", loss)
        return loss

    def p_hit_b(self, s0: int = 0):
        x = nn.functional.one_hot(torch.tensor([s0]), self.a + self.b - 1)
        return (self.u(x) * self.p).sum(axis=1)


if __name__ == "__main__":
    main()
