from dataclasses import dataclass
from typing import Generic, Protocol, Self, TypeVar

from torch import Tensor

S = TypeVar("S", bound="State")
A = TypeVar("A", bound="Action")


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


class Sampler(Protocol):
    def sample(self, state: State) -> Action:
        ...


class IntoTensor(Protocol):
    def into_tensor(self) -> Tensor:
        ...


class FromTensor(Protocol):
    def from_tensor(self, tensor: Tensor) -> Self:
        ...
