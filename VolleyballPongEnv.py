from typing import Dict, Tuple, List
from argparse import Namespace
import random
import tqdm
import os
from datetime import datetime

from pettingzoo.atari import volleyball_pong_v3
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import aec_to_parallel
import pygame
import numpy as np
import torch
import torch.nn as nn

from FrameStackV3 import frame_stack_v3
from EnvWrapper import EnvWrapper, Intention
from AECWrapper import AECWrapper
from Logging import setup_logger
from MakeModels import make_models
from Evaluate import evaluate
from utils import *
from ExperienceReplay import ReplayBuffer

import argparse
import importlib.util
import sys

logger = None
device = get_torch_device()


class VolleyballPongEnvWrapper(EnvWrapper):
    def __init__(self, base_env, penalty: float = 0.1):
        logger.info("Initializing VolleyballPongEnvWrapper")
        self.penalty = penalty
        super().__init__(base_env)
        self.accumulated_scores = {agent: 0 for agent in self.agents}
        self.accumulated_rewards = {agent: 0 for agent in self.agents}

        self.intention_followed = {agent: 0 for agent in self.agents}
        self.intention_not_followed = {agent: 0 for agent in self.agents}
        self.no_intention = {agent: 0 for agent in self.agents}

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
        # TODO(Akshay): record rewards for evaluation
        for reward in rewards:
            # set to numpy float
            rewards[reward] = np.float32(rewards[reward])
            self.accumulated_scores[reward] += max(rewards[reward], 0)
        # TODO: positive incentives for following intention?
        for intention in intentions:
            src = intention.get_src_agent()
            intention_val = intention.get_intention()
            if intention_val == "stay" and action[src][0].astype(int) == 2:
                rewards[src] -= self.penalty
                self.intention_not_followed[src] += 1
            elif intention_val == "jump" and action[src][0].astype(int) != 2:
                rewards[src] -= self.penalty
                self.intention_not_followed[src] += 1
            elif intention_val == "jump" or intention_val == "stay":
                self.intention_followed[src] += 1
            else:
                self.no_intention[src] += 1

        for agent in rewards:
            self.accumulated_rewards[agent] += rewards[agent]
        return rewards

    def get_accumulated_scores(self, agent: str) -> float:
        return self.accumulated_scores[agent]

    def get_accumulated_rewards(self, agent: str) -> float:
        return self.accumulated_rewards[agent]

    def get_intention_metrics(self, agent: str) -> float:
        return {
            "followed": self.intention_followed[agent],
            "not_followed": self.intention_not_followed[agent],
            "no_intention": self.no_intention[agent],
        }

    def get_accumulated_rewards(self, agent: str) -> float:
        return self.accumulated_rewards[agent]

    def reset(self, seed=None, options=None):
        res = super().reset(seed, options)
        self.accumulated_scores = {agent: 0 for agent in self.agents}
        self.accumulated_rewards = {agent: 0 for agent in self.agents}
        self.intention_followed = {agent: 0 for agent in self.agents}
        self.intention_not_followed = {agent: 0 for agent in self.agents}
        self.no_intention = {agent: 0 for agent in self.agents}
        return res


def create_env(params, intention_tuples):
    # create the env
    render_mode = None
    if params.evaluation_mode and params.render_game:
        pygame.init()
        pygame.display.set_mode((1, 1))
        render_mode = "human"

    env = volleyball_pong_v3.env(max_cycles=params.max_frame, render_mode=render_mode)

    env = AECWrapper(env)

    env = aec_to_parallel(env)

    env = VolleyballPongEnvWrapper(env, penalty=params.penalty)

    # add the intentions
    agents = env.agents
    logger.info(f"Agents: {env.agents}")
    # TODO: modify the intentions
    # TODO: config files

    for intent in intention_tuples:
        sender_idx = intent[0]
        receivers_idx = intent[1]

        sender = agents[sender_idx]
        receivers = [agents[idx] for idx in receivers_idx]
        env.add_intention(
            Intention(sender, receivers, ["no_preference", "stay", "jump"])
        )

    # stack the frames
    env: ParallelEnv = frame_stack_v3(env, params.stack_size)

    # must call reset!
    env.reset()

    logger.info("-" * 10 + "basic info starts" + "-" * 10)
    for agent in agents:
        logger.info(f"Agent {agent} observation space: {env.observation_space(agent)}")
        logger.info(f"Agent {agent} action space: {env.action_space(agent)}")
    logger.info("-" * 10 + "basic info ends" + "-" * 10)

    return env


def get_models(env, params):
    # create models
    models = make_models(
        env,
        device,
        hidden_sizes=params.hidden_sizes,
        stack_size=params.stack_size,
        model_path=params.model_path,
    )
    for agent, model in models.items():
        param_num = sum(p.numel() for p in model.parameters())
        logger.info(f"Agent {agent} model: {model}")
        logger.info(f"Agent {agent} model parameter number: {param_num}")

    return models


def update(agents, models, replay_buffer, params, criterion, optimizers, env):
    loss_per_agent = {}

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
        done = (terminations == True) | (truncations == True)
        rewards = torch.tensor(list(rewards), device=device, dtype=torch.float32)
        observations = np_to_torch(observations, device=device)
        if done:
            targets = rewards
        else:
            next_observations = np_to_torch(next_observations, device=device)
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
        loss_per_agent[agent] = loss.clone().detach().cpu().numpy()

        models[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()
    # TODO (Janny): Report the loss & values (tqdm?)
    return loss_per_agent


# TODO: DDQN?
def train(
    env: ParallelEnv,
    models,
    params: Namespace,
    eval_time=10,
    num_game_eval=50,
    eval_epsilon=0.05,
    save_model_time=10,
    log_dir=None,
):
    if log_dir is None:
        log_dir = f"outputs{datetime.now().strftime('%I:%M%p-%Y-%m-%d')}"

    agents = env.agents
    loss_csv = os.path.join(log_dir, "train_loss.csv")
    with open(loss_csv, "w") as f:
        col_names = ",".join(["Game"] + agents)
        f.write(col_names)
        f.write("\n")

    replay_buffer = {
        agent: ReplayBuffer(params.replay_buffer_capacity) for agent in agents
    }

    # TODO: different losses?
    criterion = nn.MSELoss()
    optimizers = {
        agent: torch.optim.Adam(model.parameters(), lr=params.lr)
        for agent, model in models.items()
    }

    epsilon = params.epsilon_init

    # the training loop here
    game_nums = params.game_nums
    maximum_frame = params.max_frame

    running_loss_per_agent = {}
    for game_num in range(game_nums):
        logger.info(f"Game {game_num}, epsilon: {epsilon}")
        observations, infos = env.reset()
        progress_bar = tqdm.tqdm(total=maximum_frame, position=0, leave=True)
        q_vals = {agent: [] for agent in agents}

        for i in range(maximum_frame):
            progress_bar.update(1)
            actions = {}
            for agent in agents:
                if random.random() < epsilon:
                    action = env.action_space(agent).sample()
                else:
                    with torch.no_grad():
                        # tuple of
                        # 0: scene np.array of shape (5, 2, stack_size)
                        # 1, 2: intentions np.array of shape (3 * stack_size, )
                        observation = observations[agent]
                        observation = np_to_torch(observation, device=device)
                        q_values = models[agent](observation)
                        max_idx = torch.argmax(q_values).item()
                        q_vals[agent].append(q_values[0, max_idx].item())
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
                for agent in agents:
                    replay_buffer[agent].push(
                        actions[agent],
                        observations[agent],
                        rewards[agent],
                        terminations[agent],
                        truncations[agent],
                        infos[agent],
                        None,
                    )
                break

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

            loss_per_agent = update(
                agents, models, replay_buffer, params, criterion, optimizers, env
            )

            for agent in agents:
                if agent not in loss_per_agent:
                    continue

                if agent not in running_loss_per_agent:
                    running_loss_per_agent[agent] = loss_per_agent[agent]
                else:
                    running_loss = running_loss_per_agent[agent] * i
                    running_loss_per_agent[agent] = (
                        running_loss + loss_per_agent[agent]
                    ) / (i + 1)

            progress_bar.set_postfix(
                {
                    a: (
                        "{:.3e}".format(running_loss_per_agent[a])
                        if a in running_loss_per_agent
                        else "NaN"
                    )
                    for a in agents
                },
                update=True,
            )

        del progress_bar

        for agent in agents:
            logger.info(
                f"Agent {agent} accumulated reward: {env.get_accumulated_scores(agent)}, accumulated penalty: {env.get_accumulated_rewards(agent)}"
            )

            if len(q_vals[agent]) != 0:
                logger.info(
                    f"Agent {agent} average q_val: {np.mean(q_vals[agent])}, std: {np.std(q_vals[agent])}"
                )

            if agent in running_loss_per_agent:
                logger.info(
                    f"Agent {agent} average loss: {running_loss_per_agent[agent]}"
                )

            # detect vanishing gradients
            param_num = sum(p.numel() for p in models[agent].parameters())
            avg_weight = (
                sum(p.abs().mean() for p in models[agent].parameters()) / param_num
            )
            for param in models[agent].parameters():
                if torch.isnan(param).any():
                    logger.error(f"Agent {agent} has NaN gradients")
                    break
            logger.info(f"Agent {agent} average weight: {avg_weight}")

        with open(loss_csv, "a") as f:
            loss_vals = [running_loss_per_agent.get(agent, "NaN") for agent in agents]
            loss_vals = [str(game_num)] + [format_loss_str(l) for l in loss_vals]
            f.write(",".join(loss_vals))
            f.write("\n")

        # save model
        if (game_num + 1) % save_model_time == 0:
            for agent, model in models.items():
                torch.save(
                    model.state_dict(),
                    f"{log_dir}/models/{agent}_model_checkpoint{game_num}.pth",
                )

        # update epsilon

        if params.epsilon_decay_type == "lin":
            epsilon = max(params.epsilon_min, epsilon - params.epsilon_decay)
        elif params.epsilon_decay_type == "mul":
            epsilon = max(params.epsilon_min, epsilon * params.epsilon_decay)

        if (game_num + 1) % eval_time == 0:
            # perform evaluations
            evaluate(
                env,
                agents,
                models,
                game_nums=num_game_eval,
                maximum_frame=params.max_frame,
                logger=logger,
                epsilon=eval_epsilon,
            )


def main(config):
    # TODO: hyperparameter tuning
    log_dir = config.log_dir

    if log_dir is None:
        log_dir = f"outputs{datetime.now().strftime('%I:%M%p-%Y-%m-%d')}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    global logger
    logger = setup_logger("VolleyballPongEnv", f"{log_dir}/test.log")

    if not os.path.exists(f"{log_dir}/models"):
        os.makedirs(f"{log_dir}/models")

    params = config.params
    intention_tuples = config.intentions_tuples
    env = create_env(params, intention_tuples)
    models = get_models(env, params)

    log_dir = config.log_dir

    if params.evaluation_mode == False:
        # Evaluate model every 10 games, save model every 10 games
        train(
            env,
            models,
            params,
            eval_time=10,
            save_model_time=10,
            log_dir=log_dir,
            num_game_eval=5,
        )
    else:
        evaluate(
            env,
            env.agents,
            models,
            game_nums=5,
            maximum_frame=500,
            logger=logger,
            epsilon=0.05,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Load a Python config file.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config .py file"
    )
    return parser.parse_args()


def import_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


if __name__ == "__main__":
    args = parse_args()
    config = import_config(args.config)
    main(config)


# TODO: visualization
# TODO: evaluation
