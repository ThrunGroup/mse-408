from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, Iterable, Protocol, Self, TypeVar

from torch import Tensor, nn

S = TypeVar("S", bound="State")
A = TypeVar("A", bound="Action")


class Loss(Enum):
    FlowMatching = 0
    DetailedBalance = 1
    TrajectoryBalance = 3


@dataclass
class Parameterization:
    input_shape: Iterable[int]
    output_shape: Iterable[int]
    loss: Loss
    train: Callable[[Any], float]


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


class Modelable(Protocol):
    def model(self, loss: Loss) -> "Model":
        ...


class Model(Protocol):
    def train(self, batch: Any) -> Any:
        ...

    def predict(self, batch: Any) -> Any:
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
