from typing import Dict, Tuple, List
from argparse import Namespace
import random
import tqdm
import os

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

device = get_torch_device()


def evaluate(env, agents, models, game_nums, maximum_frame, logger, epsilon=0.05):
    for agent in agents:
        models[agent].eval()

    all_scores = {agent: [] for agent in agents}
    all_rewards = {agent: [] for agent in agents}

    all_intentions_followed = {agent: [] for agent in agents}
    all_intentions_not_followed = {agent: [] for agent in agents}
    all_no_intentions = {agent: [] for agent in agents}

    for game_num in range(game_nums):
        logger.info(f"Evaluation Game {game_num}")
        observations, infos = env.reset()
        for i in tqdm.tqdm(range(maximum_frame)):
            actions = {}
            for agent in agents:
                if random.random() < epsilon:
                    action = env.action_space(agent).sample()
                else:
                    with torch.no_grad():
                        observation = observations[agent]
                        observation = np_to_torch(observation, device=device)
                        q_values = models[agent](observation)
                        max_idx = torch.argmax(q_values).item()
                        action = idx_to_action(max_idx, env.action_space(agent))
                actions[agent] = action
            next_observations, rewards, terminations, truncations, infos = env.step(
                actions
            )

            termination = False
            if terminations == {} or truncations == {}:  # empty dict
                termination = True
            else:
                for agent in agents:
                    if truncations[agent] or terminations[agent]:
                        termination = True
                        break
            if termination:
                break

            observations = next_observations

        logger.info(f"Evaluation Game {game_num} finished in {i} frames")

        for agent in agents:
            score = env.get_accumulated_scores(agent)
            reward = env.get_accumulated_rewards(agent)
            intention_metrics = env.get_intention_metrics(agent)
            logger.info(f"Evaluation: Agent {agent} accumulated score: {score}")
            logger.info(f"Evaluation: Agent {agent} accumulated reward: {reward}")
            logger.info(
                f"Evaluation: Agent {agent} Intention Metrics: {intention_metrics}"
            )

            all_scores[agent].append(score)
            all_rewards[agent].append(reward)

            all_intentions_followed[agent].append(intention_metrics["followed"])
            all_intentions_not_followed[agent].append(intention_metrics["not_followed"])
            all_no_intentions[agent].append(intention_metrics["no_intention"])

    for agent in agents:
        scores = np.array(all_scores[agent])
        rewards = np.array(all_rewards[agent])

        intentions_followed = all_intentions_followed[agent]
        intentions_not_followed = all_intentions_not_followed[agent]
        no_intentions = all_no_intentions[agent]

        logger.info(f"Evaluation: Agent {agent} mean score: {np.mean(scores)}")
        logger.info(f"Evaluation: Agent {agent} score std: {np.std(scores)}")
        logger.info(f"Evaluation: Agent {agent} mean reward: {np.mean(rewards)}")
        logger.info(f"Evaluation: Agent {agent} reward std: {np.std(rewards)}")

        logger.info(
            f"Evaluation: Agent {agent} mean intention followed: {np.mean(intentions_followed)}"
        )
        logger.info(
            f"Evaluation: Agent {agent} intention followed std: {np.std(intentions_followed)}"
        )
        logger.info(
            f"Evaluation: Agent {agent} mean intention not followed: {np.mean(intentions_not_followed)}"
        )
        logger.info(
            f"Evaluation: Agent {agent} intention not followed std: {np.std(intentions_not_followed)}"
        )
        logger.info(
            f"Evaluation: Agent {agent} mean no-intentions: {np.mean(no_intentions)}"
        )
        logger.info(
            f"Evaluation: Agent {agent} no-intentions std: {np.std(no_intentions)}"
        )
