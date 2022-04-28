import numpy as np
from gym_basic.algos.discretizer import Discretizer
from gym_basic.envs.sailing_env import SailingEnv
from collections import deque

# Set parameters for learning
alpha = 0.2
epsilon = 0.5
gamma = 1
discr_vector = (60, 60, 180, 30,)

env = SailingEnv()

# Create the discretizer with maxs and mins from the enviroment
d = Discretizer(discr_vector, env.observation_space.low, env.observation_space.high)

# Create and initialize Q-value table to 0
Q = np.zeros(discr_vector + (env.action_space.n,))

# Just to store the long-term-reward of the last 100 experiments
scores = deque(maxlen=100)
lrews = []
lr = []

def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])

for episode in range(1,10001):
    done = False
    R, reward = 0,0
    state = d.Discretize(env.reset())
    while done != True:
            action = choose_action(state, epsilon)
            obs, reward, done, info = env.step(action)
            new_state = d.Discretize(obs)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action]) #3
            R = gamma * R + reward
            state = new_state
    lr.append(R)
    scores.append(R)
    mean_score = np.mean(scores)
    lrews.append(np.mean(scores))
    if mean_score >= 250 and episode >= 100:
        print('Ran {} episodes. Solved after {} trials ✔'.format(episode, episode - 100))
        print('Mean score = {}, R = {}'.format(mean_score, R))

        break
    if episode % 100 == 0:
        print('Episode {} Total Reward: {} Average Reward: {}'.format(episode,R,np.mean(scores)))

