import multiprocessing as mp
import pickle
import time
from multiprocessing.queues import Queue
from typing import Iterable, Protocol

from torch.utils.data import IterableDataset

# TODO: need to convert to


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
        while not self.empty():
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


class TrajectoryQueueDataset(IterableDataset):
    def __init__(self, queue: IterableMultiprocessQueue):
        super().__init__()
        self.queue = queue

    def __iter__(self):
        return self.queue


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
