from typing import Dict, Tuple, List
from argparse import Namespace
import random
import tqdm

from pettingzoo.atari import volleyball_pong_v3
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import aec_to_parallel
import numpy as np
import torch
import torch.nn as nn

from FrameStackV3 import frame_stack_v3
from EnvWrapper import EnvWrapper, Intention
from AECWrapper import AECWrapper
from Logging import setup_logger
from MakeModels import make_models
from utils import (
    np_to_torch,
    torch_to_np,
    get_torch_device,
    idx_to_action,
    action_to_idx,
)
from ExperienceReplay import ReplayBuffer


logger = setup_logger("VolleyballPongEnv", "test.log")
device = get_torch_device()


class VolleyballPongEnvWrapper(EnvWrapper):
    def __init__(self, base_env, penalty: float = 0.1):
        logger.info("Initializing VolleyballPongEnvWrapper")
        self.penalty = penalty
        super().__init__(base_env)
        self.accumulated_rewards = {agent: 0 for agent in self.agents}

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
        for reward in rewards:
            # set to numpy float
            rewards[reward] = np.float32(rewards[reward])
            self.accumulated_rewards[reward] += rewards[reward]
        for intention in intentions:
            src = intention.get_src_agent()
            intention_val = intention.get_intention()
            if intention_val == "stay" and action[src][0].astype(int) == 2:
                rewards[src] -= self.penalty
            elif intention_val == "jump" and action[src][0].astype(int) != 2:
                rewards[src] -= self.penalty
        return rewards

    def get_accumulated_rewards(self, agent: str) -> float:
        return self.accumulated_rewards[agent]


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


def update(agents, models, replay_buffer, params, criterion, optimizers, env):
    for agent in agents:
        models[agent].train()
        if len(replay_buffer[agent]) < params.batch_size:
            continue
        (
            actions,
            observations,
            rewards,
            terminations,
            truncations,
            infos,
            next_observations,
        ) = replay_buffer[agent].sample(params.batch_size)
        observations = np_to_torch(observations, device=device)
        next_observations = np_to_torch(next_observations, device=device)
        rewards = torch.tensor(list(rewards), device=device, dtype=torch.float32)
        done = (terminations == True) | (truncations == True)
        with torch.no_grad():
            next_q_values = models[agent](next_observations)
            max_next_q_values = torch.max(next_q_values, dim=1).values
            targets = rewards + params.gamma * max_next_q_values * (1 - done)
        q_values = models[agent](observations)
        action_idx = torch.tensor(
            [action_to_idx(a, env.action_space(agent)) for a in actions]
        ).to(device)
        q_values = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        loss = criterion(q_values, targets)
        models[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()


def train(env, models, params):
    agents = env.agents
    observations, infos = env.reset()

    replay_buffer = {
        agent: ReplayBuffer(params.replay_buffer_capacity) for agent in agents
    }

    criterion = nn.MSELoss()
    optimizers = {
        agent: torch.optim.Adam(model.parameters(), lr=params.lr)
        for agent, model in models.items()
    }

    # the training loop here
    frame_num = 10000
    for i in tqdm.tqdm(range(frame_num)):
        actions = {}
        for agent in agents:
            if random.random() < params.epsilon:
                action = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    observation = observations[agent]
                    observation = np_to_torch(observation, device=device)
                    q_values = models[agent](observation)
                    max_idx = torch.argmax(q_values).item()
                    action = idx_to_action(max_idx, env.action_space(agent))
            actions[agent] = action
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        termination = False
        for agent in agents:
            if terminations[agent] or truncations[agent]:
                termination = True
                break
        if termination:
            observations, infos = env.reset()
            logger.info(f"Game over! {infos}")
            for agent in agents:
                logger.info(
                    f"Agent {agent} accumulated reward: {env.get_accumulated_rewards(agent)}"
                )
            continue

        for agent in agents:
            replay_buffer[agent].push(
                actions[agent],
                observations[agent],
                rewards[agent],
                terminations[agent],
                truncations[agent],
                infos[agent],
                next_observations[agent],
            )

        # logger.info(actions)
        # logger.info("-" * 20)
        # logger.info(f"Frame {i}")
        # for agent in agents:
        #     logger.info(f"Agent {agent}")
        #     logger.info(
        #         f"\taction: {actions[agent]}\treward: {rewards[agent]}\ttermination: {terminations[agent]}\ttruncate: {truncations[agent]}\tinfo: {infos[agent]}"
        #     )

        observations = next_observations

        update(agents, models, replay_buffer, params, criterion, optimizers, env)


def main():
    param = Namespace(
        replay_buffer_capacity=10000, batch_size=32, lr=0.001, gamma=0.99, epsilon=0.1
    )
    env = create_env()
    models = get_models(env)
    train(env, models, param)


if __name__ == "__main__":
    main()
