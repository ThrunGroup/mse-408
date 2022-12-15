from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

from torch import Tensor

S = TypeVar("S", bound="State")
A = TypeVar("A", bound="Action")


class Loss(Enum):
    FlowMatching = 0
    DetailedBalance = 1
    TrajectoryBalance = 3


@dataclass(frozen=True)
class Transition(Generic[S, A]):
    state: S
    action: A
    next_state: S

    def is_terminal(self):
        return self.action.is_terminal() or self.next_state.is_terminal()


class InvalidTransitionError(Exception):
    ...


@dataclass(frozen=True)
class Step(Generic[S, A]):
    transition: Transition[S, A]
    reward: float

    def is_terminal(self) -> bool:
        return self.transition.is_terminal()


Trajectory = list[Step]
Trajectories = list[Trajectory]


class State(Protocol):
    def is_initial(self) -> bool:
        ...

    def is_terminal(self) -> bool:
        ...


class Action(Protocol):
    def is_terminal(self) -> bool:
        ...


class Environment(Protocol, Generic[S, A]):
    def initial_state(self) -> S:
        ...

    def step(self, state: S, action: A) -> Step[S, A]:
        ...


class SimpleDiscrete(Protocol):
    @property
    def n_states(self) -> int:
        ...

    @property
    def n_actions(self) -> int:
        ...


class SimpleDiscreteEnvironment(Environment, SimpleDiscrete, Protocol):
    pass


class Sampler(Protocol):
    def sample(self, state: State) -> Action:
        ...


class IntoTensor(Protocol):
    def into_tensor(self) -> Tensor:
        ...


class FromTensor(Protocol):
    def from_tensor(self, tensor: Tensor) -> Self:
        ...
