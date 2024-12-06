import math
from Network import NeuralNet
import torch
import torch.nn as nn
from typing import Dict

"""
Makes neural network models for the agents.
Note: Assumes state of each agent is a Tuple of Boxes!
"""


def make_models(env, device, hidden_sizes=[64, 128]) -> Dict[str, NeuralNet]:
    models = {}
    for agent in env.agents:
        observation_space = env.observation_space(agent)
        input_state_space_size = 0
        for item in observation_space:
            input_state_space_size += math.prod(item.shape)

        models[agent] = NeuralNet(
            input_state_space_size, hidden_sizes, env.action_space(agent)
        ).to(device)
    return models
