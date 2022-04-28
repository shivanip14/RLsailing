from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym_basic.envs.sailing_env import SailingEnv
from gym_basic.envs.sailing_copied import SailingEnvCopied

#Training
env = SailingEnvCopied()
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save(r'gym_basic/models/SailingCopiedOptimization')
del model
