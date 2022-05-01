from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO
from gym_basic.envs.sailing_env import SailingEnv

env = SailingEnv()
env = DummyVecEnv([lambda: env])
model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save(r'gym_basic/models/SailingOptimization_TRPO')
env.close()
del model