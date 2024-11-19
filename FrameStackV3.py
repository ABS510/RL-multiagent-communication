"""
This file is a modified version of the original FrameStack.py file from the supersuit library.
We made modifications so it should support Tuple observation spaces as well.
"""

from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper
from supersuit.utils.frame_stack import get_tile_shape
from gymnasium.spaces import Box, Discrete, Tuple
from pettingzoo.utils.env import AECEnv, ParallelEnv
from typing import Union
import numpy as np


def _stack_init(obs_space, stack_size, stack_dim=-1):
    if isinstance(obs_space, Box):
        tile_shape, new_shape = get_tile_shape(
            obs_space.low.shape, stack_size, stack_dim
        )
        return np.tile(np.zeros(new_shape, dtype=obs_space.dtype), tile_shape)
    elif isinstance(obs_space, Tuple):
        return tuple(
            [
                _stack_init(subspace, stack_size, stack_dim)
                for subspace in obs_space.spaces
            ]
        )
    else:
        return 0


def _stack_obs_space(obs_space, stack_size, stack_dim=-1):
    """
    obs_space_dict: Dictionary of observations spaces of agents
    stack_size: Number of frames in the observation stack
    Returns:
        New obs_space_dict
    """
    if isinstance(obs_space, Box):
        dtype = obs_space.dtype
        # stack 1-D frames and 3-D frames
        tile_shape, new_shape = get_tile_shape(
            obs_space.low.shape, stack_size, stack_dim
        )

        low = np.tile(obs_space.low.reshape(new_shape), tile_shape)
        high = np.tile(obs_space.high.reshape(new_shape), tile_shape)
        new_obs_space = Box(low=low, high=high, dtype=dtype)
        return new_obs_space
    elif isinstance(obs_space, Discrete):
        return Discrete(obs_space.n**stack_size)
    elif isinstance(obs_space, Tuple):
        return Tuple(
            [
                _stack_obs_space(subspace, stack_size, stack_dim)
                for subspace in obs_space.spaces
            ]
        )
    else:
        assert (
            False
        ), "Stacking is currently only allowed for Box, Discrete, and Tuple observation spaces. The given observation space is {}".format(
            obs_space
        )


def _stack_obs(frame_stack, obs, obs_space, stack_size, stack_dim=-1):
    """
    Parameters
    ----------
    frame_stack : if not None, it is the stack of frames
    obs : new observation
        Rearranges frame_stack. Appends the new observation at the end.
        Throws away the oldest observation.
    stack_size : needed for stacking reset observations
    """
    if isinstance(obs_space, Box):
        obs_shape = obs.shape
        agent_fs = frame_stack

        if len(obs_shape) == 1:
            size = obs_shape[0]
            agent_fs[:-size] = agent_fs[size:]
            agent_fs[-size:] = obs

        elif len(obs_shape) == 2:
            if stack_dim == -1:
                agent_fs[:, :, :-1] = agent_fs[:, :, 1:]
                agent_fs[:, :, -1] = obs
            elif stack_dim == 0:
                agent_fs[:-1] = agent_fs[1:]
                agent_fs[:-1] = obs

        elif len(obs_shape) == 3:
            if stack_dim == -1:
                nchannels = obs_shape[-1]
                agent_fs[:, :, :-nchannels] = agent_fs[:, :, nchannels:]
                agent_fs[:, :, -nchannels:] = obs
            elif stack_dim == 0:
                nchannels = obs_shape[0]
                agent_fs[:-nchannels] = agent_fs[nchannels:]
                agent_fs[-nchannels:] = obs
        return agent_fs

    elif isinstance(obs_space, Discrete):
        return (frame_stack * obs_space.n + obs) % (obs_space.n**stack_size)

    elif isinstance(obs_space, Tuple):
        res = []
        for idx, subspace in enumerate(obs_space.spaces):
            res.append(
                _stack_obs(frame_stack[idx], obs[idx], subspace, stack_size, stack_dim)
            )
        return tuple(res)


def frame_stack_v3(env, stack_size=4, stack_dim=-1) -> Union[AECEnv, ParallelEnv]:
    """Stacks observations in a frame-like manner. This is useful for training agents on environments that require temporal context.

    Args:
        env: The environment to be modified
        stack_size (int, optional): The stack window size. Defaults to 4.
        stack_dim (int, optional): The dimension to stack on. Defaults to -1.

    Returns:
        Union[AECEnv, ParallelEnv]: The modified environment
    """
    assert isinstance(stack_size, int), "stack size of frame_stack must be an int"
    assert f"stack_dim should be 0 or -1, not {stack_dim}"

    class _FrameStackModifier(BaseModifier):
        def modify_obs_space(self, obs_space):
            if isinstance(obs_space, Box):
                assert (
                    1 <= len(obs_space.shape) <= 3
                ), "frame_stack only works for 1, 2 or 3 dimensional observations"
            elif isinstance(obs_space, Discrete):
                pass
            elif isinstance(obs_space, Tuple):
                for subspace in obs_space.spaces:
                    if isinstance(subspace, Box):
                        assert (
                            1 <= len(subspace.shape) <= 3
                        ), "frame_stack only works for 1, 2 or 3 dimensional observations"
                    elif isinstance(subspace, Discrete):
                        pass
                    else:
                        assert (
                            False
                        ), "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
                            obs_space
                        )
            else:
                assert (
                    False
                ), "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
                    obs_space
                )

            self.old_obs_space = obs_space
            self.observation_space = _stack_obs_space(obs_space, stack_size, stack_dim)
            return self.observation_space

        def reset(self, seed=None, options=None):
            self.stack = _stack_init(self.old_obs_space, stack_size, stack_dim)
            self.reset_flag = True

        def modify_obs(self, obs):
            if self.reset_flag:
                for _ in range(stack_size):
                    self.stack = _stack_obs(
                        self.stack, obs, self.old_obs_space, stack_size, stack_dim
                    )
                self.reset_flag = False
            else:
                self.stack = _stack_obs(
                    self.stack, obs, self.old_obs_space, stack_size, stack_dim
                )

            return self.stack

        def get_last_obs(self):
            return self.stack

    return shared_wrapper(env, _FrameStackModifier)
