import multiprocessing as mp
import pickle
import queue
from typing import Iterable, Protocol

from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):
    def __init__(self, stream: Iterable, serializer: "Serializer" | None = None):
        self.stream = stream
        self.serializer = serializer

    def __iter__(self):
        for item in self.stream:
            if self.serializer:
                self.serializer.save(item)
            yield item


class StreamingQueueAdapter:
    def __init__(self, queue: queue.Queue, closed: mp.Event):
        self.q = queue
        self.closed = closed

    def __iter__(self):
        while not self.closed.is_set():
            try:
                yield self.q.get(timeout=1)
            except queue.Empty:
                pass


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
