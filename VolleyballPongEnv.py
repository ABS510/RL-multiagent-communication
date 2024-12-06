from pettingzoo.atari import volleyball_pong_v3
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import aec_to_parallel
from FrameStackV3 import frame_stack_v3
from EnvWrapper import EnvWrapper, Intention
from Logging import setup_logger
import numpy as np
from typing import Dict, Tuple, List
from AECWrapper import AECWrapper
from make_models import make_models
from utils import np_to_torch, torch_to_np, get_torch_device


logger = setup_logger("VolleyballPongEnv", "test.log")
device = get_torch_device()


class VolleyballPongEnvWrapper(EnvWrapper):
    def __init__(self, base_env, penalty: float = 0.1):
        logger.info("Initializing VolleyballPongEnvWrapper")
        self.penalty = penalty
        super().__init__(base_env)

    def add_reward(
        self,
        action: Dict[str, Tuple[np.ndarray]],
        rewards: Dict[str, float],
        intentions: List[Intention],
    ) -> Dict[str, float]:
        """Adds the rewards based on the intentions. Needs to be overridden for customized rewards.

        Args:
            action (Dict[str, Tuple[np.ndarray]]): The actions for each agent.
            rewards (Dict[str, float]): The original rewards.
            intentions (List[Intention]): The list of intentions.

        Returns:
            Dict[str, float]: The updated rewards.
        """
        for intention in intentions:
            src = intention.get_src_agent()
            intention_val = intention.get_intention()
            if intention_val == "stay" and action[src][0].astype(int) == 2:
                rewards[src] -= self.penalty
            elif intention_val == "jump" and action[src][0].astype(int) != 2:
                rewards[src] -= self.penalty
        return rewards


def create_env():
    # create the env
    env = volleyball_pong_v3.env()

    env = AECWrapper(env)

    env = aec_to_parallel(env)

    env = VolleyballPongEnvWrapper(env)

    # add the intentions
    agents = env.agents
    logger.info(f"Agents: {env.agents}")
    env.add_intention(
        Intention(agents[0], [agents[0], agents[2]], ["no_preference", "stay", "jump"])
    )
    env.add_intention(
        Intention(agents[2], [agents[0], agents[2]], ["no_preference", "stay", "jump"])
    )
    env.add_intention(
        Intention(agents[1], [agents[1], agents[3]], ["no_preference", "stay", "jump"])
    )
    env.add_intention(
        Intention(agents[3], [agents[1], agents[3]], ["no_preference", "stay", "jump"])
    )

    # stack the frames
    env: ParallelEnv = frame_stack_v3(env, 4)

    # must call reset!
    env.reset()

    logger.info("-" * 10 + "basic info starts" + "-" * 10)
    for agent in agents:
        logger.info(f"Agent {agent} observation space: {env.observation_space(agent)}")
        logger.info(f"Agent {agent} action space: {env.action_space(agent)}")
    logger.info("-" * 10 + "basic info ends" + "-" * 10)

    return env


def get_models(env):
    # create models
    models = make_models(env, device)
    for agent, model in models.items():
        param_num = sum(p.numel() for p in model.parameters())
        logger.info(f"Agent {agent} model: {model}")
        logger.info(f"Agent {agent} model parameter number: {param_num}")

    return models


def train(env, models):
    agents = env.agents
    observations, infos = env.reset()

    # the training loop here
    frame_num = 5
    for i in range(frame_num):
        actions = {}
        for agent in agents:
            observation = observations[agent]
            observation = np_to_torch(observation, device=device)
            action = models[agent](observation)
            action = torch_to_np(action)
            actions[agent] = action
        # actions = {
        #     agent: env.action_space(agent).sample() for agent in agents
        # }  # random actions
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # train network here; use the dictionary observation to get the observation for each agent

        logger.info(actions)
        logger.info("-" * 20)
        logger.info(f"Frame {i}")
        for agent in agents:
            logger.info(f"Agent {agent}")
            logger.info(
                f"\taction: {actions[agent]}\treward: {rewards[agent]}\ttermination: {terminations[agent]}\ttruncate: {truncations[agent]}\tinfo: {infos[agent]}"
            )


def main():
    env = create_env()
    models = get_models(env)
    train(env, models)


if __name__ == "__main__":
    main()
