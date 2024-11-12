from pettingzoo.atari import volleyball_pong_v3
import pygame  # to capture keyboard inputs
import time  # to slow down the game for better visualization

# Initialize the environment
env = volleyball_pong_v3.env(render_mode="human")
env.reset(seed=42)

# Initialize pygame to capture keyboard inputs
pygame.init()
screen = pygame.display.set_mode((1, 1))  # Small window for capturing events only

# Define key mappings for actions
KEYS = {
    pygame.K_UP: 1,  # Move up
    pygame.K_DOWN: 3,  # Move down
    pygame.K_LEFT: 4,  # Move left
    pygame.K_RIGHT: 5,  # Move right
    pygame.K_SPACE: 2,  # Spike
}


def get_human_action():
    """Capture keyboard inputs and return the corresponding action."""
    pygame.event.pump()  # Process event queue
    keys = pygame.key.get_pressed()

    for key, action in KEYS.items():
        if keys[key]:
            return action  # Return the action corresponding to the key press

    return 0  # No action if no key is pressed


for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        if agent == "first_0":  # Assign human control to 'first_0' agent
            action = env.action_space(agent).sample()
        else:
            action = env.action_space(œ
                agent
            ).sample()  # Random action for the other agent

    env.step(action)
    # time.sleep(0.001)  # Slow down the game for better visualization

env.close()
pygame.quit()
