from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from gym_basic.config.algos import rl_agos
from gym_basic.envs.sailing_env import SailingEnv
import time

def train(algo_name):
    now = str(time.time())
    env = SailingEnv()
    env = DummyVecEnv([lambda: env])
    if algo_name == "DQN":
        model = rl_agos.get(algo_name)(policy=DQNMlpPolicy, env=env, verbose=1, exploration_fraction=0.15, exploration_final_eps=0.01, tensorboard_log="gym_basic/results/tensorboard/")
    else:
        model = rl_agos.get(algo_name)(MlpPolicy, env, verbose=1, tensorboard_log="gym_basic/results/tensorboard/")
    model.learn(total_timesteps=50000)
    model.save(r'gym_basic/models/SailingOptimization_' + algo_name)
    model.save(r'gym_basic/models/history/SailingOptimization_' + algo_name + "_" + now)
    env.close()
    del model