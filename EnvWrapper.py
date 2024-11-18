from Logging import setup_logger
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

logger = setup_logger("EnvWrapper")


class EnvWrapper(OrderEnforcingWrapper):
    def __init__(self, base_env: OrderEnforcingWrapper):
        logger.info(f"Initializing EnvWrapper with base_env: {base_env}")
        super().__init__(base_env)


def main():
    from pettingzoo.atari import volleyball_pong_v3

    base_env = volleyball_pong_v3.env()
    env = EnvWrapper(base_env)


if __name__ == "__main__":
    main()
