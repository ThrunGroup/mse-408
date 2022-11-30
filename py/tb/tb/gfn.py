from typing import Iterable

from core import Environment, Loss
from data import StreamingDataset, StreamingQueueAdapter
from models import MultiLayerPerceptron


def worker(stream: Iterable):
    for (s, a) in stream:
        pass


class GFN:
    def __init__(self):
        raise NotImplementedError("Use GFN.build(...)")

    def build(self, env: Environment, loss: Loss):
        pass
