#!/usr/bin/env python3
from collections import namedtuple
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


Param = namedtuple("Param", ["name", "min", "max", "n"])


class PosteriorEnv:
    def __init__(
        self,
        *params,
        data: np.ndarray,
        likelihood: Callable,
        batch_size=32,
    ):
        self.params = params
        self.value = [np.linspace(p.min, p.max, p.n) for p in params]
        self.data = data
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.s0 = np.zeros(len(params))

    def reward(self, state: np.ndarray) -> float:
        d = {self.params[i].name: self.value[i][s] for i, s in enumerate(state)}
        n = self.data.shape[0]
        idx = np.random.randint(n, size=self.batch_size)
        d["x"] = self.data[idx]
        return self.likelihood(**d)

    def step(self, state: np.ndarray, action: int):
        if action < len(self.params):
            state[action] += 1
        is_stop_action = action == len(self.params)
        is_terminal = state[action] == self.params[action].n - 1
        done = is_stop_action or is_terminal
        reward = 0 if not done else self.reward(state)
        return state, reward

    def parent_transitions(self, state: np.ndarray, action: int):
        is_stop_action = action == len(self.params)
        is_terminal = state[action] == self.params[action].n - 1
        if is_stop_action:
            # When the stop action is used, the state does not change; in
            # this case, the "parent" of the current (terminal) state is
            # itself and the action used to get there is the stop action.
            return [state], [action]
        parents = []
        actions = []
        for action in range(len(self.params)):
            if state[action] > 0:
                parent = np.copy(state)
                parent[action] -= 1
                # can't have a terminal parent
                is_terminal = state[action] == self.params[action].n - 1
                if is_terminal:
                    continue
                parents.append(parent)
                actions.append(action)
        return np.array(parents), np.array(actions)


class PosteriorGFN(pl.LightningModule):
    def __init__(self):
        pass

    def sample(self, n: int) -> np.ndarray:
        pass
