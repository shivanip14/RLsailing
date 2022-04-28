from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym_basic.envs.sailing_env import SailingEnv
from gym_basic.envs.sailing_copied import SailingEnvCopied

env = SailingEnvCopied()
env = DummyVecEnv([lambda: env])

model = PPO2.load(r'gym_basic/models/SailingCopiedOptimization')
obs = env.reset()

total_reward = 0

for trial in range(2000):
    done = False
    env.reset()
    while not done:
        done = env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
    total_reward += reward
print('Average reward = ', total_reward/10)
