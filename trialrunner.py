from gym_basic.envs.sailing_env import SailingEnv
from gym_basic.envs.sailing_copied import SailingEnvCopied

env = SailingEnv()
total_steps = 0
total_reward = 0
for trial in range(100):
    done = False
    env.reset()
    while not done:
        done = env.render(trial_no=trial)
        obs, reward, done, info = env.step(env.action_space.sample())
        total_steps += 1
    total_reward += reward
print('Average # steps per trial = ', total_steps/10)
print('Average reward = ', total_reward/10)