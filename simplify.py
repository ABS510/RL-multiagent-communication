import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from typing import Tuple
from Detection import Detection

COLORS = ['blue', 'blue', 'orange', 'orange', 'black']
SMALL_SHAPE = (4, 8)
LARGE_SHAPE = (4, 16)
BALL_SHAPE = (4, 2)
BORDER_CORNER_SHAPE = (10, 8)
BOARD_TOP = 24
VIDEO_FRAMES = 500

# first_0 = large RHS player
# second_0 = large LHS player

def plot_initial_detections(detections):
    fig, ax = plt.subplots()
    sc = ax.scatter(detections[:, 1], detections[:, 0], c=COLORS)
    plt.show()

    return sc

def plot_detections(all_detections):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    x = all_detections[0][:, 1]
    y = all_detections[0][:, 0] + BOARD_TOP

    ax.set_xlim(left=0, right=160)  # Flip x-axis
    ax.set_ylim(top=0, bottom=210)  # Flip y-axis
    sc = ax.scatter(x, y, c=COLORS[:len(x)])

    # Function to update both lines in each frame
    def update(frame):
        x = all_detections[frame][:, 1]
        y = all_detections[frame][:, 0] + BOARD_TOP

        # sc = ax.scatter(x, y, c=COLORS[:len(x)])
        sc.set_offsets(np.c_[x, y])        
        return sc,

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=len(all_detections), interval=50, blit=True)

    # Save the animation as a .mp4 file
    # ani.save('test.mp4', writer='ffmpeg', fps=30)
    writergif = PillowWriter(fps=30)
    ani.save('test.gif',writer=writergif)

    # Show the plot (optional)
    plt.show()

def find_rectangle(
    img: np.ndarray, 
    paddle: np.ndarray, 
    detections: list[Tuple[int]],
) -> Tuple[int]:
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

def find_rectangle_from_prev(
    img: np.ndarray, 
    paddle: np.ndarray,
    teammate_detection: Detection = None,
    prev_detection: Detection = None
) -> Tuple[int]:
    M, N = img.shape
    m, n = paddle.shape

    min_diff = float('inf')

    if prev_detection is None:
        find_rectangle(img, paddle, [])

    prev_i = prev_detection.x
    prev_j = prev_detection.y
    
    # find in row
    col_candidates = [j for neighbors in zip(range(prev_j, -1, -1), range(prev_j+1, N)) for j in neighbors]
    for j in col_candidates:
        diff = np.sum(np.abs(img[prev_i:prev_i+m, j:j+n] - paddle))
        if diff < min_diff:
            if teammate_detection is not None and teammate_detection.x == prev_i:
                continue
            if diff == 0:
                return prev_i, j
            min_diff = diff
        
    # find in col
    row_candidates = [i for neighbors in zip(range(prev_i, -1, -1), range(prev_i+1, M)) for i in neighbors]
    for i in row_candidates:
        diff = np.sum(np.abs(img[i:i+m, prev_j:prev_j+n] - paddle))
        if diff < min_diff:
            if teammate_detection is not None and teammate_detection.x == prev_i:
                continue
            if diff == 0:
                return i, prev_j
            min_diff = diff

    # search all if not found
    return find_rectangle(img, paddle, teammate_detection)
    
def find_ball(img, ball_color):
    ball_region = np.pad(
        np.ones(BALL_SHAPE),
        2
    )

    bin_img = (img == ball_color).astype(int)
    
    candidate =  find_rectangle(bin_img, ball_region, []) 
    if candidate == BORDER_CORNER_SHAPE:
        # :) 
        return None
    return candidate

class SimplifiedVolleyballPong():
    def __init__(self, debug=False):
        # init tracked variables
        self.prev_count_map = None
        self.background_color = 0
        self.border_color = 236
        self.team_candidates = (101, 223)
        self.colors_encoding = []

        self.debug = debug

        # detections
        self.paddles = {}
        self.ball = None
    
        if debug:
            self.all_detections = []

    def _update_colors(self, observation_R):
        unique, counts = np.unique(observation_R, return_counts=True)

        count_map = sorted(
            list(zip(unique.tolist(), counts.tolist())),
            key=lambda x: x[1]
        )
        
        # update color
        if count_map != self.prev_count_map:
            self.prev_count_map = count_map
            
            self.background_color = count_map[-1][0]
            self.border_color = count_map[-2][0]
            self.team_candidates = (
                min(count_map[0][0], count_map[1][0]),
                max(count_map[0][0], count_map[1][0])
            )

            new_colors_encoding = [
                self.background_color, 
                self.border_color, 
                *(self.team_candidates)
            ]
            if new_colors_encoding != self.colors_encoding:
                self.colors_encoding = new_colors_encoding
                return True
        return False

    def _assign_detection_agent(self, c, shape, observation_R):
        M, N = observation_R.shape

        if c > N/2:
            # right side
            if shape == LARGE_SHAPE:
                return 'first_0'
            else:
                return 'third_0'
        else:
            # left side
            if shape == LARGE_SHAPE:
                return 'second_0'
            else:
                return 'fourth_0'
    
    def _get_detections(self, observation_R):
        # detect per frame
        detections_arr = []
        for candidate_color in self.team_candidates:
            for shapes in LARGE_SHAPE, SMALL_SHAPE:
                paddle_r, paddle_c = find_rectangle(
                    observation_R,
                    candidate_color * np.ones(shapes),
                    detections_arr
                )

                agent_name = self._assign_detection_agent(
                    paddle_c,
                    shapes,
                    observation_R
                )

                detections_arr.append((paddle_r, paddle_c))
                self.paddles[agent_name] = Detection(agent_name, paddle_r, paddle_c, shapes, candidate_color)
                

        detected_ball = find_ball(observation_R, self.border_color)
        
        paddles = np.array(detections_arr)

        ball = np.array(detected_ball).reshape((1,2)) if detected_ball is not None else np.zeros((1, 2))

        self.detections = np.concatenate((paddles, ball), axis=0)
        if self.debug:
            self.all_detections.append(self.detections)

            if len(self.all_detections) >= VIDEO_FRAMES:
                self.get_observation_video()
                self.all_detections = []

        return self.detections

    def _get_detections_from_previous(self, observation_R):
        # detect per frame
        detections = []

        for candidate_color in self.team_candidates:
            for shapes in LARGE_SHAPE, SMALL_SHAPE:
                paddle_candidate = find_rectangle(
                    observation_R,
                    candidate_color * np.ones(shapes),
                    detections
                )

                detections.append(paddle_candidate)
        detected_ball = find_ball(observation_R, self.border_color)
        
        paddles = np.array(detections)

        ball = np.array(detected_ball).reshape((1,2)) if detected_ball is not None else np.zeros((1, 2))

        self.detections = np.concatenate((paddles, ball), axis=0)
        if self.debug:
            self.all_detections.append(self.detections)

            if len(self.all_detections) >= VIDEO_FRAMES:
                self.get_observation_video()
                self.all_detections = []

        return paddles, ball

    def observe(self, observation):
        observation_R = observation[BOARD_TOP:, :, 0]
        # hack: the items are after the 24th line
        self._update_colors(observation_R)
        # if (self._update_colors(observation_R)):
            # return self._get_detections(observation_R)
        return self._get_detections_from_previous(observation_R)

    def get_observation_video(self):
        if not self.debug:
            print("Not debug")

        plot_detections(self.all_detections)

