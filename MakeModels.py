import math
from Network import NeuralNet
import torch
import torch.nn as nn
from typing import Dict

"""
Makes neural network models for the agents.
Note: Assumes state of each agent is a Tuple of Boxes!
"""


# TODO (Yining): load models from file
def make_models(
    env, device, hidden_sizes=[64, 128], stack_size=30, model_path=None
) -> Dict[str, NeuralNet]:
    models = {}
    for agent in env.agents:
        observation_space = env.observation_space(agent)
        input_state_space_size = 0
        for item in observation_space:
            input_state_space_size += math.prod(item.shape) // stack_size
        output_size = 1
        for space in env.action_space(agent).spaces:
            output_size *= space.n
        models[agent] = NeuralNet(input_state_space_size, hidden_sizes, output_size).to(
            device
        )
    if model_path is not None:
        for agent in env.agents:
            models[agent].load_state_dict(torch.load(model_path[agent]))
    return models
