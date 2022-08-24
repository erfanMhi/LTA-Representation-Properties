import os
import sys


import pytest
import torch
import numpy as np


CURRENT_DIR = sys.path[0] + '/' + '..'
print(sys.path)
sys.path.insert(0, CURRENT_DIR)
print(sys.path)

from core.utils.test_funcs import online_dead_neurons_ratio

print("Change dir to", os.getcwd())
os.chdir(CURRENT_DIR)
print(os.getcwd())

class DummyLogger:

    def __init__(self):
        pass

    def info(self, log):
        print(log)


class DummyConfig:

    def __init__(self):
        self.logger = DummyLogger()
    
    def state_normalizer(self, state):
        return state

class DummyAgent:
    def __init__(self):
        self.cfg = DummyConfig()
        self.total_steps = 100
        self.episode_rewards = [10, 20]

    def rep_net(self, states):
        return states


def pytest_configure(): # TODO: This function uses dummy data for now. These dummy inputs should be replaced by the main inputs sometime soon and the agent should be initialized properly as well.

    pass    


def test_dead_neurons_ratio():
    
    label = None

    approximate_dead_neurons_ratio = 0.2
    error_margin = 0.05


    property_measure_data = (torch.rand(1, 1024) > approximate_dead_neurons_ratio).type(torch.float32)

    print(property_measure_data)

    agent = DummyAgent()

    assert approximate_dead_neurons_ratio - error_margin < online_dead_neurons_ratio(agent, property_measure_data, label) < approximate_dead_neurons_ratio + error_margin