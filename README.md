## Advanced Topics in Computational Intelligence - Reinforcement Learning (UPC-MAI)
### Introduction
The objective of this project is to implement a sailboat agent which learns to sail towards a given target, in the prevalent environmental conditions, using reinforcement learning with a custom OpenAI Gym environment.

### Run instructions
1.  Prerequisites:
    1. Installing msi from https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi (to use the TRPO algorithm from stable_baseline)
	2. Install mpi4py (a message passing API) with `pip install mpi4py`
	3. Install `gym==0.19.0` compatible with `tensorflow==1.15.0`, `stable_baselines==2.9.0`, `numpy==1.21.5`
2. Either:
    1. Run `compare_algos.py` - if you want to train and test all algorithms. Currently supporting DQN, TRPO, A2C, PPO2, and ACER.
        1. If tensorboard upload is creating issues with respect to auth token (it most probably will, especially for the first time or if running after a while), revoke creds explcitly with `tensorboard dev auth revoke` and attempt upload again. It will prompt for a fresh login this time.
        2. If you're still facing issues, comment out the line #9 in _compare_algos.py_.
	2. Run the methods in `train.py` and then `test.py`, if you want to run a specific algorithm - by passing the algo name as argument:
	    `python -c "import train; train.train(\"TRPO\")"`
	    `python -c "import test; test.test(\"TRPO\")"`
3. If individual algorithms were run or if you want to visualise the results of a new run of an existing algorithm (maybe after changing the world configuration), upload the logs created after training to TensorBoard.dev using the following command: `tensorboard dev upload --logdir gym_basic/results/tensorboard/`, and follow the experiment link created to view the stats.

### Misc
- The source code along with the report of this project could be found at https://github.com/shivanip14/RLsailing 
- Word config can be changed via _gym_basic/config/world_config.py_.
- History of the models created during every run can be found at _gym_basic/models/history_.
- More info for OpenAI + _stable_baselines_: https://towardsdatascience.com/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82
