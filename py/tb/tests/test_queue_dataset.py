from torch.utils.data import DataLoader

from tb.data import IterableMultiprocessQueue, QueueDataset


def test_queue_dataset(n_items: int = 100):
    q_update = IterableMultiprocessQueue()
    ds = QueueDataset(q_update)
    for i in range(n_items):
        q_update.put(i)
    loader = DataLoader(ds, batch_size=n_items, num_workers=1)
    batch = next(iter(loader))
    q_update.close()
    assert batch.size(dim=0) == n_items, "Incorrect number of items!"
