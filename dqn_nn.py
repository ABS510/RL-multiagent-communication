import torch.nn as nn
import torch.nn.functional as F
import torch

"""
Neural Network class for DQN
"""


class NeuralNet(nn.Module):
    """
    Params:
    input_space: size of observable space (including any communication visible to agent)

    hidden_sizes: Array of hidden layer sizes. There will be len(hidden_sizes) hidden layers

    num_actions: Size of action space (excludes communication)

    communication_dim: Size of communication output. Will be appended to output

    comm_act_fn: activation function (softmax/relu/sigmoid, etc.) for communication output.
                 Default value is identity function (Do nothing)

    op_act_fn: activation function (softmax/relu/sigmoid, etc.) for communication output.
                 Default value is identity function (Do nothing).
                 This is usual for Q-values

    All activations are ReLU for hidden layers.
    Over action space, op_act_fn is applied
    Over communication, comm_act_fn is applied.
    The results are concatenated.

    Input shape to model: (batch_size, input_space)
    Output shape from model: (batch_size, num_actions+communication_dim)

    For baseline model with no communication:
    set communication_dim=0
    """

    def __init__(
        self,
        input_space,
        hidden_sizes,
        # num_actions,
        # communication_dim,
        action_space,
        # op_act_fn=lambda x: x,
        # comm_act_fn=lambda x: x,
    ):
        super(NeuralNet, self).__init__()

        hidden_sizes = [input_space] + hidden_sizes

        lin_layers = []
        for i in range(len(hidden_sizes) - 1):
            lin_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self.lin_layers = nn.ModuleList(lin_layers)

        output_layers = []
        print(type(action_space))
        for space in action_space:
            output_layers.append(nn.Linear(hidden_sizes[-1], space.n))

        self.output_layers = nn.ModuleList(output_layers)

        self.normalize_output = nn.Softmax(dim=-1)
        # self.comm_act_fn = comm_act_fn

        # self.output_act = op_act_fn

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self._stack_tuple(x)
        for layer in self.lin_layers:
            x = F.relu(layer(x))

        output = []
        for layer in self.output_layers:
            output.append(self.normalize_output(layer(x)))

        # get the max index
        res = [torch.argmax(o, dim=-1) for o in output]
        return tuple(res)

        # x = self.output_layer(x)

        # x[:, 0 : self.num_actions] = self.output_act(x[:, 0 : self.num_actions])
        # x[:, self.num_actions :] = self.comm_act_fn(x[:, self.num_actions :])
        # return x

    def _stack_tuple(self, x):
        assert isinstance(x, tuple)
        return torch.cat(x, dim=-1)
