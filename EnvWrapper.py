from typing import List
import functools

import numpy as np
from gymnasium.spaces import Tuple, Box, Discrete, MultiDiscrete
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from Logging import setup_logger

logger = setup_logger("EnvWrapper")


class Intention:
    def __init__(
        self,
        src_agent: str,
        dst_agent: List[str],
        candidates: List[str],
        default: int = 0,
    ):
        assert 0 <= default < len(candidates), "Invalid default index"
        self._src_agent = src_agent
        self._dst_agent = dst_agent
        self._candidates = candidates
        self._val = default

    def __repr__(self):
        return f"Intention: {self._src_agent} -> {self._dst_agent} : {self._candidates[self._val]}"

    def is_src_agent(self, agent: str) -> bool:
        return agent == self._src_agent

    def is_dst_agent(self, agent: str) -> bool:
        return agent in self._dst_agent

    @property
    def action_space(self) -> Discrete:
        return Discrete(len(self._candidates))

    @property
    def observation_space(self) -> Tuple:
        return Box(low=0, high=1, shape=(len(self._candidates),), dtype=np.uint8)


class EnvWrapper(OrderEnforcingWrapper):
    def __init__(self, base_env: OrderEnforcingWrapper):
        logger.info(f"Initializing EnvWrapper with base_env: {base_env}")
        super().__init__(base_env)
        self._intentions = []

    def add_intention(self, intention: Intention):
        self._intentions.append(intention)
        logger.info(f"Added intention: {intention}")

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Tuple:
        origin_space = super().observation_space(agent)
        space_list = [origin_space]
        for intention in self._intentions:
            if intention.is_dst_agent(agent):
                space_list.append(intention.observation_space)
        return Tuple(space_list)

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> MultiDiscrete:
        origin_space = super().action_space(agent).n
        space_list = [origin_space]
        for intention in self._intentions:
            if intention.is_src_agent(agent):
                space_list.append(intention.action_space.n)
        return MultiDiscrete(space_list)

    def render(self):
        # TODO
        pass

    def observe(self, agent):
        # TODO
        return super().observe(agent)
        pass

    def close(self):
        # TODO
        pass

    def reset(self, seed=None, options=None):
        # TODO
        super().reset(seed, options)
        pass

    def step(self, action):
        # TODO
        pass


# Example usage
if __name__ == "__main__":
    from pettingzoo.atari import volleyball_pong_v3

    base_env = volleyball_pong_v3.env()
    env = EnvWrapper(base_env)
    env.add_intention(
        Intention(
            "first_0",
            ["first_0", "second_0"],
            ["no_preference", "stay", "jump"],
        )
    )
    env.add_intention(
        Intention(
            "second_0",
            ["first_0", "second_0"],
            ["no_preference", "stay", "jump"],
        )
    )
    env.reset()
    print(env.agents)
    print(env.observation_space(env.agents[0]))
    agent_state = env.observe(env.agents[0])
    print(type(agent_state))
    action_space = env.action_space(env.agents[0])
    print(type(action_space))
    print(action_space)
