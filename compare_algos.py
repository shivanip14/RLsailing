from gym_basic.config.algos import rl_agos
from test import test

for algorithm in rl_agos:
    test(algorithm)