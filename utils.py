import numpy as np
import torch

from typing import Tuple


def np_to_torch(x, device="cpu", dtype=torch.float32) -> Tuple[torch.Tensor, ...]:
    if isinstance(x, tuple):
        x = [np_to_torch(i, device=device, dtype=dtype) for i in x]
        return tuple(x)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def torch_to_np(x) -> Tuple[np.ndarray, ...]:
    if isinstance(x, tuple):
        x = [torch_to_np(i) for i in x]
        return tuple(x)
    return x.detach().cpu().numpy()


def get_torch_device() -> str:
    # # cuda or cpu or mps
    # if torch.cuda.is_available():
    #     return "cuda"
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def idx_to_action(idx: int, action_space) -> np.ndarray:
    action = []
    for space in action_space.spaces:
        action.append(idx % space.n)
        idx = idx // space.n
    return np.array(action)


def action_to_idx(action: np.ndarray, action_space) -> int:
    idx = 0
    for i, space in enumerate(action_space.spaces):
        idx += action[i] * np.prod([space.n for space in action_space.spaces[:i]])
    return int(idx)
