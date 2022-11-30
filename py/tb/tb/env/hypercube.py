from dataclasses import dataclass, replace
from typing import Self

import numpy as np
from core import InvalidTransitionError, Step, Transition
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class HypercubeState:
    coordinate: np.ndarray
    n_per_dim: int

    def is_initial(self) -> bool:
        return self.coordinate.sum() == 0

    def is_terminal(self) -> bool:
        return (self.coordinate == self.n_per_dim - 1).any()

    def apply(self, action: "HypercubeAction") -> Self:
        if action.is_terminal():
            return self
        coordinate = self.coordinate.copy()
        coordinate[action.direction] += 1
        if coordinate[action.direction] < self.n_per_dim:
            return replace(self, coordinate=coordinate)
        raise InvalidTransitionError

    def into_tensor(self) -> Tensor:
        return F.one_hot(self.coordinate, self.n_per_dim).flatten()


@dataclass(frozen=True)
class HypercubeAction:
    direction: int
    terminal: int

    def is_terminal(self) -> bool:
        return self.direction == self.terminal


@dataclass(frozen=True)
class Hypercube:
    n_dims: int
    n_per_dim: int
    r_0: float

    def initial_state(self) -> HypercubeState:
        return HypercubeState(np.zeros(self.n_dims), self.n_per_dim)

    def step(
        self, state: HypercubeState, action: HypercubeAction
    ) -> Step[HypercubeState, HypercubeAction]:
        transition = Transition(state, action, state.apply(action))
        return Step(transition, self._reward(transition.next_state))

    def _reward(self, transition):
        if not transition.is_terminal():
            return 0.0
        s_t = transition.next_state
        x_abs = np.abs(s_t / (self.n_per_dim - 1) * 2 - 1)
        return (
            self.r_0
            + 0.5 * (x_abs > 0.5).prod()
            + 2 * ((x_abs > 0.6) * (x_abs < 0.8)).prod()
        )
