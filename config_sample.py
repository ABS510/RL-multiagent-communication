from argparse import Namespace

params = Namespace(
        replay_buffer_capacity=10000,
        batch_size=32,
        lr=0.001,
        gamma=0.99,
        max_frame=60 * 60,
        game_nums=200,
        epsilon_init=0.5,
        epsilon_decay=0.1,
        epsilon_decay_type="lin",
        epsilon_min=0.01,
        penalty=0.1,
        stack_size=4,
        hidden_sizes=[128, 64],
    )

# intentions are tuples of form:
# (sender_idx, (receiever_idx1, receiver_idx2, ...))
# Example: (0, (0,2)) will be an intention from agent0 to agent0 and agent2
intentions_tuples = [
    (0, (0,2)),
    (2, (0,2)),
    # (1, (1,3)),
    # (3, (1,3))
]
