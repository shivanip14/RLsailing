from gym_basic.config.algos import rl_agos
from train import train
from test import test
import os

for algorithm in rl_agos:
    train(algorithm)
    test(algorithm)
os.system("tensorboard dev upload --logdir gym_basic/results/tensorboard/")