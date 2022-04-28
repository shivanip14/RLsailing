import gym
from gym_basic.envs.sailing_env import SailingEnv
from stable_baselines.common.env_checker import check_env

env = SailingEnv()
check_env(env)