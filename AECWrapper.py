import time
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper
from pettingzoo.atari import volleyball_pong_v3
import numpy as np
import gymnasium as gym

from constants import *

"""
Simplified environment for Volley Ball Pong.
Initialize as AECWrapper(volleyball_pong_v3.env())
"""


class AECWrapper(OrderEnforcingWrapper):
    def __init__(self, env=None):

        super().__init__(env.env)
        if env is None:
            self.env = volleyball_pong_v3.env()
        else:
            self.env = env

        self.right_team_color = None
        self.left_team_color = None

        mask = np.zeros((210, 160), dtype=np.uint8)
        mask[:, 8:152] = 1
        mask[0:50, :] = 0
        mask[130:, 80 - 4 : 80 + 4] = 0
        mask[180:, :] = 0

        high = np.zeros(12)
        high[::2] = 210
        high[1::2] = 160
        high[-2] = 21
        high[-1] = 21
        self.high = high.reshape((6, 2)).astype(np.float64)
        self.ball_mask = mask
        self.accumulated_scores = {}

    def reset(self, seed=None, options=None):
        res = super().reset(seed, options)
        self.accumulated_scores = {agent: 0 for agent in self.agents}
        return res

    def observation_space(self, agent):
        high = np.zeros(12)
        high[::2] = 210
        high[1::2] = 160
        high[-2] = 21
        high[-1] = 21
        self.high = high.reshape((6, 2))
        return gym.spaces.Box(
            low=np.zeros((6, 2)), high=high.reshape((6, 2)), dtype=np.float64
        )

    def step(self, action):
        res = super().step(action)
        for agent in self.agents:
            self.accumulated_scores[agent] += max(
                0, int(self.env.env.env.rewards[agent])
            )
        return res

    def observe(self, agent):
        screen = super().observe(agent)

        if self.right_team_color is None or self.left_team_color is None:
            self.get_team_colors(screen)
        res = self.get_detections(screen)
        # res is 5 * 2, append [self.accumulated_scores[self.agents[0]], self.accumulated_scores[self.agents[1]]]
        # print all attributes of self
        res = np.concatenate(
            (
                res,
                [
                    [
                        self.accumulated_scores[agent],
                        self.accumulated_scores[agent],
                    ]
                ],
            )
        ).astype(np.float64)
        res /= self.high
        return res

    def find_rectangle(self, img, paddle, detections):
        M, N = img.shape
        m, n = paddle.shape

        min_diff = float("inf")
        min_i = 0
        min_j = 0
        for i in range(0, M - m):
            for j in range(0, N - n):
                diff = np.sum(np.abs(img[i : i + m, j : j + n] - paddle))
                if diff < min_diff:
                    if len(detections) > 0 and detections[-1][0] == i:
                        continue

                    if diff == 0:
                        return i, j
                    min_diff = diff
                    min_i = i
                    min_j = j

        return min_i, min_j

    def find_ball(self, img, ball_color):
        where_white = np.all(img == ball_color, axis=-1) & self.ball_mask
        ball_locs = np.where(where_white)
        if len(ball_locs[0]) == 0:
            return np.array([0, 0])
        else:
            return np.array([ball_locs[0][0], ball_locs[1][0]])

    def find_paddle(self, color, space, rows, col_start):
        mask = np.all(space - color, axis=-1)
        indices = np.where(np.logical_not(mask))
        row = indices[0][0]
        min_col = indices[1].min()

        return np.array([rows[row], col_start + min_col])

    def get_detections(self, screen):
        N = screen.shape[1]

        # in order, first_0, second_0, and others
        large_search_space = screen[LARGE_ROWS]
        small_search_space = screen[SMALL_ROWS]
        first_0_detection = self.find_paddle(
            self.right_team_color,
            large_search_space[:, N // 2 :, :],
            LARGE_ROWS,
            N // 2,
        )
        second_0_detection = self.find_paddle(
            self.left_team_color, large_search_space[:, : N // 2, :], LARGE_ROWS, 0
        )
        third_0_detection = self.find_paddle(
            self.right_team_color,
            small_search_space[:, N // 2 :, :],
            SMALL_ROWS,
            N // 2,
        )
        fourth_0_detection = self.find_paddle(
            self.left_team_color, small_search_space[:, : N // 2, :], SMALL_ROWS, 0
        )

        border_color = np.array([236, 236, 236])
        detected_ball = self.find_ball(screen, border_color)

        paddles = np.array(
            [
                first_0_detection,
                second_0_detection,
                third_0_detection,
                fourth_0_detection,
            ]
        )

        state_vector = np.concatenate((paddles, detected_ball.reshape((1, -1))), axis=0)

        return state_vector

    def get_team_colors(self, observation):
        N = observation.shape[1]

        if self.right_team_color is None:
            self.right_team_color = np.zeros((3,), dtype=np.uint8)
        if self.left_team_color is None:
            self.left_team_color = np.zeros((3,), dtype=np.uint8)

        for c in range(3):
            right_colors = np.unique(observation[SEARCH_ROWS, N // 2 :, c])
            left_colors = np.unique(observation[SEARCH_ROWS, : N // 2, c])

            right_paddle_colors = np.setdiff1d(right_colors, left_colors)
            left_paddle_colors = np.setdiff1d(left_colors, right_colors)

            self.right_team_color[c] = right_paddle_colors[0]
            self.left_team_color[c] = left_paddle_colors[0]
