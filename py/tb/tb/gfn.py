from core import Loss
from data import StreamingDataset, StreamingQueueAdapter
from models import MultiLayerPerceptron


def worker(stream: Iterable):
    for (s, a) in stream:
        pass
