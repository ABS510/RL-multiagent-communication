from argparse import Namespace

# params = Namespace(
#     replay_buffer_capacity=30000,
#     batch_size=64,
#     lr=0.0001,
#     gamma=0.99,
#     max_frame=60 * 120,
#     game_nums=2000,
#     epsilon_init=1,
#     epsilon_decay=0.9 / 50,
#     epsilon_decay_type="lin",
#     epsilon_min=0.1,
#     penalty=0.1,
#     stack_size=60,
#     hidden_sizes=[1024, 512, 256, 64],
# )
params = Namespace(
    replay_buffer_capacity=100000,
    batch_size=64,
    lr=1e-4,
    gamma=0.99,
    max_frame=60 * 240,
    game_nums=2000,
    epsilon_init=1,
    epsilon_decay=0.9 / 50,
    epsilon_decay_type="lin",
    epsilon_min=0.1,
    penalty=0.01,
    stack_size=5,
    hidden_sizes=[1024, 256],
    render_game=False,
    evaluation_mode=False,
    model_path=None,  # or dict of model path if evaluation_mode=True
)

# intentions are tuples of form:
# (sender_idx, (receiever_idx1, receiver_idx2, ...))
# Example: (0, (0,2)) will be an intention from agent0 to agent0 and agent2
intentions_tuples = [(0, (0, 2)), (2, (0, 2)), (1, (1, 3)), (3, (1, 3))]
# intentions_tuples = []

log_dir = "no_intentions_1e-4_high_gamma"


# Example of config file for evaluation

# params = Namespace(
#     replay_buffer_capacity=30000,
#     batch_size=64,
#     lr=1e-5,
#     gamma=0.99,
#     max_frame=60 * 120,
#     game_nums=200,
#     epsilon_init=1,
#     epsilon_decay=0.9 / 50,
#     epsilon_decay_type="lin",
#     epsilon_min=0.1,
#     penalty=0.05,
#     stack_size=5,
#     hidden_sizes=[1024, 256],
#     evaluation_mode=True,
#     model_path={
#         "first_0": "models/first_0_model_checkpoint9.pth",
#         "second_0": "models/second_0_model_checkpoint9.pth",
#         "third_0": "models/third_0_model_checkpoint9.pth",
#         "fourth_0": "models/fourth_0_model_checkpoint9.pth",
#     },  # or dict of model path if evaluation_mode=True
# )

# intentions_tuples = [(0, (0, 2)), (2, (0, 2)), (1, (1, 3)), (3, (1, 3))]

# log_dir = "eval"
