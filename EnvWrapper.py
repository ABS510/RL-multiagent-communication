from typing import List, Tuple, Dict
import functools

import numpy as np
import gymnasium as gym
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

from FrameStackV3 import frame_stack_v3
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

    def check_validity(self, agents: List[str]):
        """Checks if the source and destination agents are valid.

        Args:
            agents (List[str]): The list of agents in the environment.
        """

        if self._src_agent not in agents:
            raise ValueError("Invalid source agent")
        for agent in self._dst_agents:
            if agent not in agents:
                raise ValueError("Invalid destination agent")

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
    def action_space(self) -> gym.spaces.Discrete:
        """
        The action space defined by the number of candidates.

        Returns:
            gym.spaces.Discrete: A gym.spaces.Discrete space with size equal to the number of candidates.
        """
        return gym.spaces.Discrete(len(self._candidates))

    @property
    def observation_space(self) -> gym.spaces.Box:
        """
        The observation space representing the candidates as a one-hot vector.

        Returns:
            gym.spaces.Box: A gym.spaces.Box space with a shape corresponding to the number of candidates.
        """
        return gym.spaces.Box(
            low=0, high=1, shape=(len(self._candidates),), dtype=np.uint8
        )

    def reset(self) -> None:
        """
        Resets the intention to the default value.
        """
        self._val = self._default

    def observe(self) -> np.ndarray:
        """
        Returns the one-hot vector representing the current intention.

        Returns:
            np.ndarray: A one-hot vector representing the current intention.
        """
        obs = np.zeros(len(self._candidates))
        obs[self._val] = 1
        return obs

    def step(self, action: np.ndarray):
        """
        Updates the intention based on the given action.

        Args:
            action (np.ndarray): The action to take.
        """
        action = int(action)
        if not (0 <= action < len(self._candidates)):
            raise ValueError("Invalid action index")
        self._val = action


class EnvWrapper(aec_to_parallel_wrapper):
    def __init__(self, base_env: aec_to_parallel_wrapper):
        """initializes the EnvWrapper with the base environment.

        Args:
            base_env (aec_to_parallel_wrapper): The base environment to wrap.
        """
        logger.info(f"Initializing EnvWrapper with base_env: {base_env}")
        super().__init__(base_env.aec_env)
        self._intentions: List[Intention] = []
        self.reset()

    def add_intention(self, intention: Intention):
        """Adds an intention to the environment.

        Args:
            intention (Intention): The intention to add.
        """
        intention.check_validity(self.agents)
        self._intentions.append(intention)
        logger.info(f"Added intention: {intention}")

    def observation_space(self, agent: str) -> gym.spaces.Tuple:
        """Returns the observation space for the given agent.

        Args:
            agent (str): The agent to get the observation space for.

        Returns:
            gym.spaces.Tuple: The observation space for the agent.
        """
        origin_space = super().observation_space(agent)
        space_list = [origin_space]
        for intention in self._intentions:
            if intention.is_dst_agent(agent):
                space_list.append(intention.observation_space)
        return gym.spaces.Tuple(space_list)

    def action_space(self, agent: str) -> gym.spaces.Tuple:
        """Returns the action space for the given agent.

        Args:
            agent (str): The agent to get the action space for.

        Returns:
            gym.spaces.Tuple: The action space for the agent.
        """
        space_list = [super().action_space(agent)]
        for intention in self._intentions:
            if intention.is_src_agent(agent):
                space_list.append(intention.action_space)
        return gym.spaces.Tuple(space_list)

    def __observe(self, agent: str) -> List[np.ndarray]:
        observation_list = [self.aec_env.observe(agent)]
        for intention in self._intentions:
            if intention.is_dst_agent(agent):
                observation_list.append(intention.observe())
        return observation_list

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict]:
        """Resets the environment and the intentions.

        Args:
            seed (_type_, optional): The seed to reset. Defaults to None.
            options (_type_, optional): The option to reset. Defaults to None.

        Returns:
            Tuple[Dict[str, List[np.ndarray]], Dict]: The observations and the info dictionary.
        """
        _, info = super().reset(seed, options)
        for intention in self._intentions:
            intention.reset()
        logger.info(f"Reset {self}")
        observations = {agent: self.__observe(agent) for agent in self.agents}
        return observations, info

    def step(
        self, action: Dict[str, Tuple[np.ndarray]]
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict, Dict, Dict, Dict]:
        """Steps the environment with the given actions.

        Args:
            action (Dict[str, Tuple[np.ndarray]]): The actions for each agent.

        Returns:
            Tuple[Dict[str, List[np.ndarray]], Dict, Dict, Dict, Dict]: The observations, rewards, terminations, truncations, and infos.
        """
        original_action = {agent: action[agent][0] for agent in self.agents}
        _, rewards, terminations, truncations, infos = super().step(original_action)
        for agent in self.agents:
            action_idx = 1
            for intention in self._intentions:
                if intention.is_src_agent(agent):
                    intention.step(action[agent][action_idx])
                    action_idx += 1
        observations = {agent: self.__observe(agent) for agent in self.agents}
        rewards = self.add_reward(action, rewards)
        return observations, rewards, terminations, truncations, infos

    def add_reward(
        self, action: Dict[str, Tuple[np.ndarray]], rewards: Dict[str, float]
    ) -> Dict[str, float]:
        """Adds the rewards based on the intentions. Needs to be overridden for customized rewards.

        Args:
            action (Dict[str, Tuple[np.ndarray]]): The actions for each agent.
            rewards (Dict[str, float]): The original rewards.

        Returns:
            Dict[str, float]: The updated rewards.
        """
        return rewards


# TODO: For customized rewards that depend on the intentions, override the add_reward method.

# Example usage
if __name__ == "__main__":
    from pettingzoo.atari import volleyball_pong_v3

    # create the env
    env = EnvWrapper(volleyball_pong_v3.parallel_env())

    # add the intentions
    agents = env.agents
    logger.info(f"Agents: {env.agents}")
    env.add_intention(
        Intention(agents[0], agents[0:2], ["no_preference", "stay", "jump"])
    )
    env.add_intention(
        Intention(agents[1], agents[0:2], ["no_preference", "stay", "jump"])
    )

    # stack the frames
    env = frame_stack_v3(env, 4)

    # must call reset!
    observations, info = env.reset()

    logger.info("-" * 20)
    # check the observation and action spaces
    logger.info(f"Agents: {agents}")
    for agent in agents:
        logger.info(f"Agent {agent} observation space: {env.observation_space(agent)}")
        logger.info(f"Agent {agent} action space: {env.action_space(agent)}")
        logger.info(f"Agent {agent} random action: {env.action_space(agent).sample()}")
        logger.info(
            f"Agent {agent} random action type: {type(env.action_space(agent).sample())}"
        )
        for observation in observations[agent]:
            logger.info(f"Agent {agent} observation shape: {observation.shape}")

    # the training loop here
    frame_num = 1
    for i in range(frame_num):
        # insert policy here; use the dictionary observation to get the observation for each agent
        actions = {
            agent: env.action_space(agent).sample() for agent in agents
        }  # random actions
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # train network here; use the dictionary observation to get the observation for each agent

        logger.info("-" * 20)
        logger.info(f"Frame {i}")
        for observation in observations[agents[0]]:
            logger.info(f"Agent {agents[0]} observation shape: {observation.shape}")
        logger.info(f"Rewards: {rewards}")
        logger.info(f"Terminations: {terminations}")
        logger.info(f"Truncations: {truncations}")
        logger.info(f"Infos: {infos}")
