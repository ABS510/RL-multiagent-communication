from pettingzoo.atari import volleyball_pong_v3
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import aec_to_parallel
from FrameStackV3 import frame_stack_v3
from EnvWrapper import EnvWrapper, Intention
from Logging import setup_logger
import numpy as np
from typing import Dict, Tuple, List

logger = setup_logger("VolleyballPongEnv")


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


# create the env
env = volleyball_pong_v3.env()

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
observations, info = env.reset()

# the training loop here
frame_num = 10
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
