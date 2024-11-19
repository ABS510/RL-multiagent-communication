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
        dst_agents: List[str],
        candidates: List[str],
        default: int = 0,
    ):
        """
        Represents the intention of an agent to communicate with other agents.

        Args:
            src_agent (str): The agent who sets the intention.
            dst_agents (List[str]): The agents who can receive the intention
                                    (should include src_agent).
            candidates (List[str]): The list of possible intentions as strings.
            default (int, optional): The initial intention index. Defaults to 0
                                     (the first candidate).
        """
        if not (0 <= default < len(candidates)):
            raise ValueError("Invalid default index")

        self._src_agent = src_agent
        self._dst_agents = dst_agents
        self._candidates = candidates
        self._default = default
        self._val = default

    def __repr__(self) -> str:
        return (
            f"Intention: {self._src_agent} -> {self._dst_agents} : "
            f"{self._candidates[self._val]}"
        )

    def is_src_agent(self, agent: str) -> bool:
        """
        Checks if the given agent is the source agent.

        Args:
            agent (str): The agent to check.

        Returns:
            bool: True if the agent is the source agent, False otherwise.
        """
        return agent == self._src_agent

    def is_dst_agent(self, agent: str) -> bool:
        """
        Checks if the given agent is one of the destination agents.

        Args:
            agent (str): The agent to check.

        Returns:
            bool: True if the agent is a destination agent, False otherwise.
        """
        return agent in self._dst_agents

    @property
    def action_space(self) -> Discrete:
        """
        The action space defined by the number of candidates.

        Returns:
            Discrete: A Discrete space with size equal to the number of candidates.
        """
        return Discrete(len(self._candidates))

    @property
    def observation_space(self) -> Box:
        """
        The observation space representing the candidates as a one-hot vector.

        Returns:
            Box: A Box space with a shape corresponding to the number of candidates.
        """
        return Box(low=0, high=1, shape=(len(self._candidates),), dtype=np.uint8)

    def reset(self) -> None:
        """
        Resets the intention to the default value.
        """
        self._val = self._default


class EnvWrapper(OrderEnforcingWrapper):
    def __init__(self, base_env: OrderEnforcingWrapper):
        logger.info(f"Initializing EnvWrapper with base_env: {base_env}")
        super().__init__(base_env)
        self._intentions = []

    def add_intention(self, intention: Intention):
        self._intentions.append(intention)
        logger.info(f"Added intention: {intention}")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Tuple:
        origin_space = super().observation_space(agent)
        space_list = [origin_space]
        for intention in self._intentions:
            if intention.is_dst_agent(agent):
                space_list.append(intention.observation_space)
        return Tuple(space_list)

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

    def observe(self, agent: str):
        # TODO
        return super().observe(agent)
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        for intention in self._intentions:
            intention.reset()
        logger.info(f"Reset {self}")

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
    logger.info(f"Agents: {env.agents}")
    logger.info(f"Agent 0 observation space: {env.observation_space(env.agents[0])}")
    logger.info(f"Agent 0 action space: {env.action_space(env.agents[0])}")

    logger.info(f"Agent_0 agent space type: {type(env.observe(env.agents[0]))}")
    logger.info(f"Agent_0 agent space shape: {env.observe(env.agents[0]).shape}")
