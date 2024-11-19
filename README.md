# RL-multiagent-communication

---

Notes from Yining:

The example usage has been moved to `example_usage.py`.

`Logging.py` include a func `setup_logger` and could help managing logs better.

`FrameStackV3.py` includes a wrapper that stacks the environment frames. Implementation does not matter that much, there's just one function interface as shown in the example, so it should be easy to use directly.

`EnvWrapper.py` includes the extended environment with the modified action space. Currently it only supports parallel training (agents take action simultaneously, not one by one). If you want to combine it with `FrameStackV3`, then the EnvWrapper should be wrapped in the FrameStack wrapper; otherwise it won't work. Check the example usage for more details :)

`VolleyballPongEnv.py` includes the customized env for Volleyball Pong, together with an example of usage.