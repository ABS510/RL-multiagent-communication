import math
from dqn_nn import NeuralNet
import torch
import torch.nn as nn

'''
Makes neural network models for the agents.
Note: Assumes state of each agent is a Tuple of Boxes!
'''
def make_models(env):
    models = {}
    for agent in env.agents:
        observation_space = env.observation_space(agent)
        input_state_space_size = 0
        for item in observation_space:
            input_state_space_size += math.prod(item.shape)


        action_dim = env.action_space(agent)[0].n
        if len(env.action_space(agent)) > 1:
            communication_dim = env.action_space(agent)[1].n
        else:
            communication_dim = 0

        models[agent] = NeuralNet(input_state_space_size, [64, 128], action_dim, communication_dim, comm_act_fn=nn.Softmax(dim=1))
    return models