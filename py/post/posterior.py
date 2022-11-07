#!/usr/bin/env python3
from typing import Callable, Union

import numpy as np
import pytorch_lightning as pl
import torch


def flow_matching_loss(
    inflow: torch.Tensor | np.ndarray | list[Union[int, float]],
    outflow: torch.Tensor | np.ndarray | list[Union[int, float]],
    delta: float = 0,
) -> torch.Tensor:
    inflow = torch.as_tensor(inflow)
    outflow = torch.as_tensor(outflow)
    ratio = (delta + inflow) / (delta + outflow)
    return torch.log(ratio).pow(2).mean()


class PosteriorEnv:
    def __init__(self, reward: Callable):
        self.reward = reward
        pass

    def step(**kwargs):
        pass


class PosteriorGFN(pl.LightningModule):
    def __init__(self):
        pass

    def sample(self, n: int) -> torch.Tensor:
        pass
