from gym_basic.envs.sailing_env import SailingEnv
from gym_basic.config.world_config import MAX_TEST_TRIALS
from gym_basic.config.algos import rl_agos
from tqdm import tqdm

def test(algo_name):
    print('##### Testing with {} #####'.format(algo_name))
    env = SailingEnv()

    model = rl_agos.get(algo_name).load(r'gym_basic/models/SailingOptimization_' + algo_name)
    obs = env.reset()

    total_steps = 0
    total_reward = 0
    highest_reward = 0
    highest_reward_trial_no = 0
    for trial in tqdm(range(MAX_TEST_TRIALS), desc='Testing trials'):
        episodic_reward = 0
        done = False
        env.reset()
        while not done:
            done = env.render(trial_no = trial, highest_reward_trial_no = highest_reward_trial_no, highest_reward = highest_reward, max_trials=MAX_TEST_TRIALS)
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action, trial)
            total_steps += 1
            episodic_reward += reward
        if episodic_reward >= highest_reward:
            highest_reward = episodic_reward
            highest_reward_trial_no = trial
        total_reward += episodic_reward
    print('Average # steps per trial = ', total_steps/MAX_TEST_TRIALS)
    print('Average reward = ', total_reward/MAX_TEST_TRIALS)