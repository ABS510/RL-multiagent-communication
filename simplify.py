import numpy as np
import matplotlib.pyplot as plt

COLORS = ['blue', 'blue', 'orange', 'orange']
SMALL_SHAPE = (4, 8)
LARGE_SHAPE = (4, 16)
BALL_SHAPE = (4, 2)
BORDER_CORNER_SHAPE = (10, 8)

# first_0 = large RHS player
# second_0 = large LHS player

def plot_initial_detections(detections):
    fig, ax = plt.subplots()
    sc = ax.scatter(detections[:, 1], detections[:, 0], c=COLORS)
    # plt.ioff()
    plt.show()

    return sc

def plot_detections(detections, sc=None):
    # if sc is None:
    return plot_initial_detections(detections)

    # sc.set_offsets([detections[:, 1], detections[:, 0]])  # Update the data points
    # plt.draw()
    # plt.show()
    # return sc

def find_rectangle(img, paddle, detections):
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
