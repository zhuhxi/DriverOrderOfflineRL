from gymnasium.envs.registration import register

register(
    id='cyborg',
    entry_point='cyborg_env.envs:CyborgEnv'
)