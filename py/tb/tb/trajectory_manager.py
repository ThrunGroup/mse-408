import queue
from threading import Queue


class TrajectoryManager:
    def __init__(
        self,
        q_steps: Queue,
        q_updates: Queue,
        q_policy: Queue,
    ):
        self.q_steps = q_steps
        self.q_updates = q_updates
        self.q_policy = q_policy
