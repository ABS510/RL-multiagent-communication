from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper
from pettingzoo.atari import volleyball_pong_v3
import numpy as np
import gymnasium as gym

'''
Simplified environment for Volley Ball Pong.
Initialize as AECWrapper(volleyball_pong_v3.env())
'''
class AECWrapper(OrderEnforcingWrapper):
    def __init__(self, env=None):
        
        super().__init__(env.env)
        if env is None:
            self.env = volleyball_pong_v3.env()
        else:
            self.env = env

    def observation_space(self, agent):
        high = np.zeros(10)
        high[::2] = 210
        high[1::2] = 160
        return gym.spaces.Box(low=np.zeros(10), high=high, dtype=np.int64)

    def observe(self, agent):
        BOARD_TOP = 24
        screen = super().observe(agent)
        res = self.get_detections(screen[BOARD_TOP:, :, 0])
        return res

    def find_rectangle(self, img, paddle, detections):
        M, N = img.shape
        m, n = paddle.shape

        min_diff = float('inf')
        min_i = 0
        min_j = 0
        for i in range(0, M-m):
            for j in range(0, N-n):
                diff = np.sum(np.abs(img[i:i+m, j:j+n] - paddle))
                if diff < min_diff:
                    if len(detections) > 0 and detections[-1][0] == i:
                        continue
                    # print(detections)

                    if diff == 0:
                        return i,j
                    min_diff = diff
                    min_i = i
                    min_j = j

        return min_i, min_j

    def find_ball(self, img, ball_color):

        SMALL_SHAPE = (4, 8)
        LARGE_SHAPE = (4, 16)
        BALL_SHAPE = (4, 2)
        BORDER_CORNER_SHAPE = (10, 8)
        BOARD_TOP = 24
        VIDEO_FRAMES = 500

        ball_region = np.pad(
            np.ones(BALL_SHAPE),
            2
        )

        bin_img = (img == ball_color).astype(int)

        candidate =  self.find_rectangle(bin_img, ball_region, [])
        if candidate == BORDER_CORNER_SHAPE:
            # :)
            return None
        return candidate

    def get_detections(self, observation_R):
      SMALL_SHAPE = (4, 8)
      LARGE_SHAPE = (4, 16)
      BALL_SHAPE = (4, 2)
      BORDER_CORNER_SHAPE = (10, 8)
      BOARD_TOP = 24
      VIDEO_FRAMES = 500
      # detect per frame
      detections = []
      team_candidates = (101, 223)
      for candidate_color in team_candidates:
          for shapes in LARGE_SHAPE, SMALL_SHAPE:
              paddle_candidate = self.find_rectangle(
                  observation_R,
                  candidate_color * np.ones(shapes),
                  detections
              )

              detections.append(paddle_candidate)
      border_color = 236
      detected_ball = self.find_ball(observation_R, border_color)

      paddles = np.array(detections).reshape(-1)
      ball = np.array(detected_ball).reshape(-1) if detected_ball is not None else np.array([0,0]).reshape(-1)

      state_vector = np.concatenate((paddles, ball))

      return state_vector