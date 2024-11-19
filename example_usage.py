from pettingzoo.atari import volleyball_pong_v3
from pettingzoo.utils.env import ParallelEnv
from FrameStackV3 import frame_stack_v3
from EnvWrapper import EnvWrapper, Intention
from Logging import setup_logger

logger = setup_logger("Example")

# create the env
env = EnvWrapper(volleyball_pong_v3.parallel_env())

# add the intentions
agents = env.agents
logger.info(f"Agents: {env.agents}")
env.add_intention(Intention(agents[0], agents[0:2], ["no_preference", "stay", "jump"]))
env.add_intention(Intention(agents[1], agents[0:2], ["no_preference", "stay", "jump"]))

# stack the frames
env: ParallelEnv = frame_stack_v3(env, 4)

# must call reset!
observations, info = env.reset()

logger.info("-" * 20)
# check the observation and action spaces
logger.info(f"Agents: {agents}")
for agent in agents:
    logger.info(f"Agent {agent} observation space: {env.observation_space(agent)}")
    logger.info(f"Agent {agent} action space: {env.action_space(agent)}")
    logger.info(f"Agent {agent} random action: {env.action_space(agent).sample()}")
    logger.info(
        f"Agent {agent} random action type: {type(env.action_space(agent).sample())}"
    )
    for observation in observations[agent]:
        logger.info(f"Agent {agent} observation shape: {observation.shape}")

# the training loop here
frame_num = 1
for i in range(frame_num):
    # insert policy here; use the dictionary observation to get the observation for each agent
    actions = {
        agent: env.action_space(agent).sample() for agent in agents
    }  # random actions
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # train network here; use the dictionary observation to get the observation for each agent

    logger.info("-" * 20)
    logger.info(f"Frame {i}")
    for observation in observations[agents[0]]:
        logger.info(f"Agent {agents[0]} observation shape: {observation.shape}")
    logger.info(f"Rewards: {rewards}")
    logger.info(f"Terminations: {terminations}")
    logger.info(f"Truncations: {truncations}")
    logger.info(f"Infos: {infos}")
