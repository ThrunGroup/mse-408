#!/usr/bin/env python3
import itertools as it
import multiprocessing as mp
from collections import namedtuple
from typing import Callable, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import IterableDataset

State = np.ndarray
Reward = float
Action = int
Step = namedtuple("Step", ["state", "action", "reward", "next_state"])
Param = namedtuple("Param", ["name", "min", "max", "n"])


def flow_matching_loss(
    in_flow: torch.Tensor | np.ndarray | list[Union[int, float]],
    out_flow: torch.Tensor | np.ndarray | list[Union[int, float]],
    delta: float = 0,
) -> torch.Tensor:
    in_flow = torch.as_tensor(in_flow)
    out_flow = torch.as_tensor(out_flow)
    ratio = (delta + in_flow) / (delta + out_flow)
    return torch.log(ratio).pow(2).mean()


class PosteriorEnv:
    def __init__(
        self,
        *params,
        data: np.ndarray,
        likelihood: Callable,
        batch_size=32,
    ):
        self.params = list(params)
        self.value = [np.linspace(p.min, p.max, p.n) for p in params]
        self.data = data
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.s0 = np.zeros(len(params), dtype=np.int64)

    def reward(self, state: State) -> Reward:
        d = {self.params[i].name: self.value[i][s] for i, s in enumerate(state)}
        n = self.data.shape[0]
        idx = np.random.randint(n, size=self.batch_size)
        d["x"] = self.data[idx]
        return self.likelihood(**d)

    def step(self, state: State, action: Action) -> Step:
        assert action >= 0 and action <= len(self.params), "invalid action!"
        next_state = np.copy(state)
        is_terminal = False
        is_stop_action = action == len(self.params)
        if action < len(self.params):
            next_state[action] += 1
            is_terminal = state[action] == self.params[action].n - 1
        done = is_stop_action or is_terminal
        reward = 0 if not done else self.reward(next_state)
        return Step(state, action, reward, next_state)

    def parent_transitions(
        self, state: State, action: Action
    ) -> tuple[list[State], list[Action]]:
        is_stop_action = action == len(self.params)
        if is_stop_action:
            return [state], [action]
        parents = []
        actions = []
        for action in range(len(self.params)):
            if state[action] > 0:
                parent = np.copy(state)
                parent[action] -= 1
                is_terminal = state[action] == self.params[action].n - 1
                if is_terminal:
                    continue
                parents.append(parent)
                actions.append(action)
        return parents, actions

    def loss(self, model: nn.Module, trajectories: list[list[Step]]):
        states_t = torch.tensor(
            [step.next_state for trajectory in trajectories for step in trajectory]
        )
        done_t = torch.tensor(
            [int(step.reward > 0) for trajectory in trajectories for step in trajectory]
        ).unsqueeze(1)
        rewards_t = torch.tensor(
            [step.reward for trajectory in trajectories for step in trajectory]
        ).unsqueeze(1)
        idxs, parents, actions = [], [], []
        with mp.Pool() as pool:
            for i, (pars, acts) in enumerate(
                pool.starmap(
                    parent_transitions,
                    [
                        (self, step.next_state, step.action)
                        for trajectory in trajectories
                        for step in trajectory
                    ],
                )
            ):
                idxs.extend([i] * len(pars))
                parents.extend(pars)
                actions.extend(acts)
        idxs_t = torch.tensor(idxs)
        parents_t = torch.tensor(parents)
        actions_t = torch.tensor(actions)
        in_flows = model(parents_t)[torch.arange(parents_t.shape[0]), actions_t]
        in_flow = torch.zeros(states_t.shape[0]).index_add_(0, idxs_t, in_flows)
        out_flows = model(states_t) * (1 - done_t)
        out_flow = torch.sum(torch.cat([out_flows, rewards_t], 1), 1)
        return flow_matching_loss(in_flow, out_flow)

    def states_to_tensors(self, states: list[State]) -> torch.Tensor:
        states_t = torch.tensor(np.array(states))
        return torch.hstack(
            [F.one_hot(states_t[:, i], p.n) for i, p in enumerate(self.params)]
        ).squeeze()


def parent_transitions(env: PosteriorEnv, state: State, action: Action):
    return env.parent_transitions(state, action)


class InfiniteIterableDataset(IterableDataset):
    """An infinite iterable dataset that simply increases the count."""

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return it.count(0)


class GFN(pl.LightningModule):
    def __init__(self, env, loss=flow_matching_loss):
        self.env = env

    def training_step(self, _batch, _batch_idx):
        # TODO: sample actions
        # TODO: generate batch_size trajectories
        # TODO: update model
        return self.env.loss(self.model, trajectories)

    def sample(self, n: int) -> list[State]:
        # TODO: generate samples from model
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @classmethod
    def train_dataloader(cls):
        # NOTE: triggers the training step, the data isn't used
        return InfiniteIterableDataset()
