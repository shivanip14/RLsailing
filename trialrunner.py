from gym_basic.envs.sailing_env import SailingEnv
from gym_basic.config.world_config import MAX_TEST_TRIALS

env = SailingEnv()
total_steps = 0
total_reward = 0
highest_reward = 0
highest_reward_trial_no = 0
for trial in range(MAX_TEST_TRIALS):
    done = False
    env.reset()
    print('Running trial #' + str(trial))
    while not done:
        done = env.render(trial_no = trial, highest_reward_trial_no = highest_reward_trial_no, highest_reward = highest_reward, max_trials=MAX_TEST_TRIALS)
        obs, reward, done, info = env.step(env.action_space.sample(), trial)
        total_steps += 1
    if reward >= highest_reward:
        highest_reward = reward
        highest_reward_trial_no = trial
    total_reward += reward
print('Average # steps per trial = ', total_steps/MAX_TEST_TRIALS)
print('Average reward = ', total_reward/MAX_TEST_TRIALS)