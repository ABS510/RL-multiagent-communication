from argparse import Namespace

params = Namespace(
    replay_buffer_capacity=30000,
    batch_size=64,
    lr=0.0001,
    gamma=0.99,
    max_frame=60 * 120,
    game_nums=2000,
    epsilon_init=1,
    epsilon_decay=0.9 / 50,
    epsilon_decay_type="lin",
    epsilon_min=0.1,
    penalty=0.1,
    stack_size=60,
    hidden_sizes=[1024, 512, 256, 64],
)

# intentions are tuples of form:
# (sender_idx, (receiever_idx1, receiver_idx2, ...))
# Example: (0, (0,2)) will be an intention from agent0 to agent0 and agent2
intentions_tuples = [
    (0, (0, 2)),
    (2, (0, 2)),
    (1, (1,3)),
    (3, (1,3))
]

log_dir = "test_config_sample"