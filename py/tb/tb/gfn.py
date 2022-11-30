from typing import Iterable

from core import Environment, Loss, Modelable


def worker(stream: Iterable):
    for (s, a) in stream:
        pass


class GFN:
    def __init__(self, env: Environment | Modelable, loss: Loss):
        self.model = env.model(loss)
