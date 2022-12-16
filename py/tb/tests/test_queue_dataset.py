from torch.utils.data import DataLoader

from tb.data import (IterableMultiprocessQueue, QueueDataset,
                     TrajectoryQueueDataset, trajectory_collator)
from tb.env import Hypercube


def test_queue_dataset(n_items: int = 100):
    q = IterableMultiprocessQueue()
    q.cancel_join_thread()
    ds = QueueDataset(q)
    for i in range(n_items):
        q.put(i)
    loader = DataLoader(ds, batch_size=n_items, num_workers=1)
    batch = next(iter(loader))
    q.close()
    assert batch.size(dim=0) == n_items, "Incorrect number of items!"


def test_trajectory_dataset(n_traj: int = 100):
    env = Hypercube()
    s0 = env.initial_state()
    step = env.step(s0, env.random_action())
    traj = [step]
    while not step.is_terminal():
        next_state = step.transition.next_state
        action = env.random_action()
        step = env.step(next_state, action)
        traj.append(step)
    q_traj = IterableMultiprocessQueue()
    q_traj.cancel_join_thread()
    ds = TrajectoryQueueDataset(q_traj)
    for tid in range(n_traj):
        q_traj.put((tid, traj))
    loader = DataLoader(
        ds, batch_size=n_traj, num_workers=1, collate_fn=trajectory_collator
    )
    batch = next(iter(loader))
    q_traj.close()
    assert batch.ids.unique().size(dim=0) == n_traj
