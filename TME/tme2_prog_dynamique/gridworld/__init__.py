import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/RLADL/TME/tme2/env')

from gridworld.gridworld_env import GridworldEnv

from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gridworld:GridworldEnv',
)
