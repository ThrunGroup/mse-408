import multiprocessing as mp
import queue
from threading import Event

from ..core import Environment


def worker(
    env: Environment,
    stop: Event,
    q_policy: queue.Queue,
    q_steps: queue.Queue,
):
    q_policy.cancel_join_thread()
    q_steps.cancel_join_thread()
    while not stop.is_set():
        try:
            trajectory_id, state, action = q_policy.get(timeout=0.1)
        except queue.Empty:
            continue
        step = env.step(state, action)
        q_steps.put((trajectory_id, step))


class EnvManager:
    def __init__(
        self,
        env: Environment,
        stop: mp.Event,
        q_policy: queue.Queue,
        q_steps: queue.Queue,
        n_processes=mp.cpu_count(),
    ):
        self._samplers = [
            mp.Process(target=worker, args=(env, stop, q_policy, q_steps))
            for _ in range(n_processes)
        ]

    def start(self):
        for sampler in self._samplers:
            sampler.start()

    def stop(self):
        for sampler in self._samplers:
            sampler.join()

    def __del__(self):
        self.stop()
