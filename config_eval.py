from argparse import Namespace

params = Namespace(
    replay_buffer_capacity=100000,
    batch_size=64,
    lr=1e-4,
    gamma=0.99,
    max_frame=60 * 240 * 2,
    game_nums=2000,
    epsilon_init=1,
    epsilon_decay=0.9 / 40,
    epsilon_decay_type="lin",
    epsilon_min=0.1,
    penalty=0.02,
    stack_size=5,
    hidden_sizes=[1024, 256],
    render_game=True, # True for human-visible gameplay, False if not
    evaluation_mode=True,
    model_path={
        "first_0": "models/first_0_model_checkpoint79.pth",
        "second_0": "models/second_0_model_checkpoint79.pth",
        "third_0": "models/third_0_model_checkpoint79.pth",
        "fourth_0": "models/fourth_0_model_checkpoint79.pth",
    },  
)

# intentions are tuples of form:
# (sender_idx, (receiever_idx1, receiver_idx2, ...))
# Example: (0, (0,2)) will be an intention from agent0 to agent0 and agent2
# intentions_tuples = [(0, (0, 2)), (2, (0, 2)), (1, (1, 3)), (3, (1, 3))]
intentions_tuples = [(0, (0, 2)), (2, (0, 2))]
log_dir = "eval_output"
