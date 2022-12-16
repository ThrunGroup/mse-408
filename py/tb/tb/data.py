import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import Iterable, Protocol, Self

import torch
from torch.utils.data import IterableDataset


class IterableMultiprocessQueue(Queue):
    def __init__(self, sentinal=-1, maxsize=0, *, ctx=None):
        self._sentinal = sentinal
        super().__init__(
            maxsize=maxsize, ctx=ctx if ctx is not None else mp.get_context()
        )

    def __iter__(self):
        return self

    def close(self):
        self.put(self._sentinal)
        while self._buffer:
            time.sleep(0.01)
        super().close()

    def __next__(self):
        try:
            result = self.get()
        except OSError:
            raise StopIteration
        if result == self._sentinal:
            self.put(result)
            raise StopIteration
        return result


@dataclass(frozen=True)
class TrajectoryTensor:
    ids: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    @classmethod
    def from_many(cls, batch: list[Self]) -> Self:
        ids, states, actions, rewards = [], [], [], []
        for tt in batch:
            ids.append(tt.ids)
            states.append(tt.states)
            actions.append(tt.actions)
            rewards.append(tt.rewards)
        return TrajectoryTensor(
            torch.hstack(ids),
            torch.vstack(states),
            torch.vstack(actions),
            torch.tensor(rewards),
        )


class TrajectoryQueueDataset(IterableDataset):
    def __init__(self, queue: IterableMultiprocessQueue):
        super().__init__()
        self.queue = queue

    def __iter__(self):
        for tid, traj in self.queue:
            states, actions, reward = [], [], 0
            for step in traj:
                tr = step.transition
                states.append(tr.state.into_tensor())
                actions.append(tr.action.into_tensor())
                reward = step.reward
            yield TrajectoryTensor(
                torch.tensor(tid).repeat(len(traj)),
                torch.vstack(states),
                torch.vstack(actions),
                torch.tensor([reward]),
            )


def trajectory_collator(batch):
    return TrajectoryTensor.from_many(batch)


class QueueDataset(IterableDataset):
    def __init__(self, queue: IterableMultiprocessQueue):
        super().__init__()
        self.queue = queue

    def __iter__(self):
        return iter(self.queue)


class Serializer(Protocol):
    def save(self, obj):
        ...

    def load(self, obj) -> Iterable:
        ...


class PickleSerializer:
    def __init__(self, path):
        self.path = path
        self.handle = None
        self.pickler = None

    def save(self, obj):
        if not self.handle:
            self.handle = open(self.path, "wb")
            self.pickler = pickle.Pickler(self.handle)
        self.pickler.dump(obj)

    def load(self) -> Iterable:
        with open(self.path, "rb") as f:
            upk = pickle.Unpickler(f)
            while True:
                try:
                    yield upk.load()
                except EOFError:
                    break

    def __del__(self):
        if self.handle:
            self.handle.close()
