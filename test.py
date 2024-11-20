from pettingzoo.atari import volleyball_pong_v3
import pygame  # to capture keyboard inputs
import time  # to slow down the game for better visualization
import numpy as np

from simplify import *

human_mode = False

# Initialize the environment
env = volleyball_pong_v3.env(render_mode="human")
env.reset(seed=42)

# Initialize pygame to capture keyboard inputs
pygame.init()
screen = pygame.display.set_mode((1, 1))  # Small window for capturing events only

# Define key mappings for actions
KEYS = {
    pygame.K_SPACE: 1,  # Spike
    pygame.K_UP: 2,  # Move up
    pygame.K_RIGHT: 3,  # Move right
    pygame.K_LEFT: 4,  # Move left
    pygame.K_DOWN: 5,  # Move down
}

KEYS_NAMES = {
    pygame.K_UP: 'UP',  # Move up
    pygame.K_DOWN: 'DOWN',  # Move down
    pygame.K_LEFT: 'LEFT',  # Move left
    pygame.K_RIGHT: 'RIGHT',  # Move right
    pygame.K_SPACE: 'SPACE',  # Spike
}

HUMAN_AGENTS = ['first_0', 'second_0', 'third_0', 'fourth_0']

def get_human_action():
    """Capture keyboard inputs and return the corresponding action."""
    pygame.event.pump()  # Process event queue
    keys = pygame.key.get_pressed()

    for key, action in KEYS.items():
        if keys[key]:
            return action  # Return the action corresponding to the key press
    return 0  # No action if no key is pressed

prev_human_agent = 0
prev_action = 0

prev_count_map = None
background_color = 0
border_color = 236
team_candidates = (101, 223)

colors_encoding = []
sc = None

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    # np.savetxt("observation_ball.txt", observation[:, :, 0], fmt='%d') 

    observation_R = observation[24:, :, 0]
    # hack: the items are after the 24th line
    unique, counts = np.unique(observation_R, return_counts=True)
    count_map = sorted(
        list(zip(unique.tolist(), counts.tolist())),
        key=lambda x: x[1]
    )
    
    # update color
    if count_map != prev_count_map:
        prev_count_map = count_map
        
        background_color = count_map[-1][0]
        border_color = count_map[-2][0]
        team_candidates = (
            min(count_map[0][0], count_map[1][0]),
            max(count_map[0][0], count_map[1][0])
        )

        new_colors_encoding = [background_color, border_color, *team_candidates]
        if new_colors_encoding != colors_encoding:
            colors_encoding = new_colors_encoding
            
    # detect per frame
    detections = []
    for candidate_color in team_candidates:
        for shapes in LARGE_SHAPE, SMALL_SHAPE:
            paddle_candidate = find_rectangle(
                observation_R,
                candidate_color * np.ones(shapes),
                detections
            )

            detections.append(paddle_candidate)

    detected_ball = find_ball(observation_R, border_color)
    if detected_ball is not None:
        print("ball: ", detected_ball)

        # print(np.array(detections)) 
        # sc = plot_detections(np.array(detections), sc)
        # plot_updating_visualization(np.array(detections))

    if termination or truncation:
        action = None
        env.step(action)
    elif human_mode:
        # check if continued action
        action = get_human_action()

        # if action ends, switch to next human agent
        if prev_action != action:
            if action == 0:
                prev_human_agent = (prev_human_agent + 1) % 4
                # print("switching to ", prev_human_agent)

            prev_action = action
            if agent == HUMAN_AGENTS[prev_human_agent]:
                env.step(action)
            else:
                env.step(0)

        # if action persists, dispatch only to current human agent
        elif prev_action == action and HUMAN_AGENTS[prev_human_agent] == agent:
            env.step(action)
        
        else:
            env.step(0)
    else:
        action = env.action_space(agent).sample() 
        env.step(action)


env.close()
pygame.quit()
