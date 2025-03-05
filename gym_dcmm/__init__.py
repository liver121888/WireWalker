from gymnasium.envs.registration import register

register(
    id="gym_dcmm/DcmmVecWorld-v0",
    entry_point="gym_dcmm.envs:DcmmVecEnv",
    #  max_episode_steps=300,
)

register(
    id="gym_dcmm/WireWalkerVecWorld-v0",
    entry_point="gym_dcmm.envs:WireWalkerVecWorld",
    #  max_episode_steps=300,
)