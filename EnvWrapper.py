import importlib


class ExtendedEnv:
    def __init__(self, env_name):
        module = importlib.import_module(f"pettingzoo.{env_name}")
        self.env = module.env()


test_env = ExtendedEnv("atari.volleyball_pong_v3")
