import torch.nn as nn
import torch.nn.functional as F
import torch

"""
Neural Network class for DQN
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# TODO: Different model arch
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
        output_size,
        # op_act_fn=lambda x: x,
        # comm_act_fn=lambda x: x,
        max_len,
    ):
        super(NeuralNet, self).__init__()

        print(input_space)
        embedding_size = 64

        self.input_embedding = nn.Linear(input_space, embedding_size)

        self.positional_encoding = PositionalEncoding(embedding_size, max_len=max_len)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_size, nhead=4, dim_feedforward=256, batch_first=True
            ),
            num_layers=2,
        )

        self.output_layer = nn.Linear(embedding_size, output_size)

        self.EOS = nn.Parameter(torch.randn(1, embedding_size))

        # hidden_sizes = [input_space] + hidden_sizes

        # lin_layers = []
        # for i in range(len(hidden_sizes) - 1):
        #     lin_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # self.lin_layers = nn.ModuleList(lin_layers)

        # self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self._stack_tuple(x)
        # swap the last two dimensions; x could be (batch_size, seq_len, input_size) or (seq_len, input_size)
        x = x.permute(0, 2, 1) if len(x.shape) == 3 else x.unsqueeze(0).permute(0, 2, 1)
        x = self.input_embedding(x)
        # add eos
        x = torch.cat([x, self.EOS.expand(x.shape[0], 1, -1)], dim=1)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x

        # if not isinstance(x, torch.Tensor):
        #     x = self._stack_tuple(x)
        # for layer in self.lin_layers:
        #     x = F.relu(layer(x))
        # x = self.output_layer(x)
        # return x

    def _stack_tuple(self, x):
        assert isinstance(x, tuple)
        if isinstance(x[0], torch.Tensor):
            return torch.cat([item.reshape(-1, item.shape[-1]) for item in x], dim=0)
        else:
            x = [self._stack_tuple(i) for i in x]

            x = torch.stack(x, dim=0)
            return x
