from argparse import Namespace

params = Namespace(
    replay_buffer_capacity=100000,
    batch_size=64,
    lr=1e-4,
    gamma=0.99,
    max_frame=60 * 240,
    game_nums=80,
    epsilon_init=1,
    epsilon_decay=0.9 / 40,
    epsilon_decay_type="lin",
    epsilon_min=0.2,
    penalty=0.02,
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

log_dir = "final_part_intention2"
