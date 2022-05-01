from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from gym_basic.config.algos import rl_agos
from gym_basic.envs.sailing_env import SailingEnv
import time

algo_name = "DQN"
algo = rl_agos.get(algo_name)

env = SailingEnv()
env = DummyVecEnv([lambda: env])
model = algo(DQNMlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save(r'gym_basic/models/SailingOptimization_' + algo_name)
model.save(r'gym_basic/models/history/SailingOptimization_' + algo_name + "_" + str(time.time()))
env.close()
del model