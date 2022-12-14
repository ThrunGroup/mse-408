import multiprocessing as mp

from tb.env import EnvManager, Hypercube


def test_env_manager(n_steps: int = 1000):
    q_policy = mp.Queue()
    q_steps = mp.Queue()
    q_policy.cancel_join_thread()
    q_steps.cancel_join_thread()
    stop = mp.Event()
    env = Hypercube()
    mgr = EnvManager(env, stop, q_policy, q_steps)
    mgr.start()
    s0 = env.initial_state()
    # NOTE: put more in pipeline than we need to test safe shutdown
    for t_id in range(n_steps * 2):
        a = env.random_action()
        q_policy.put((t_id, s0, a))
    steps = []
    for _ in range(n_steps):
        steps.append(q_steps.get())
    stop.set()
