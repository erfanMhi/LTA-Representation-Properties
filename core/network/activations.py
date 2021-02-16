import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional

class LTA:
    def __init__(self, cfg):
        # self.tiling = cfg.activation['tiling']
        # 1 tiling, binning
        self.n_tilings = 1
        self.n_tiles = cfg.activation_config['tile']
        self.bound_low, self.bound_high = cfg.activation_config['bound_low'], cfg.activation_config['bound_high']
        self.delta = (self.bound_high - self.bound_low) / self.n_tiles
        self.c_mat = torch.as_tensor(np.array([self.delta * i for i in range(self.n_tiles)]) + self.bound_low, dtype=torch.float32)
        self.eta = cfg.activation_config['eta']
        self.d = cfg.activation_config['input']

    def __call__(self, reps):
        # temp = torch.tanh(reps)
        # print(reps.shape)
        # print(self.c_mat)
        #temp = torch.clamp(reps, self.bound_low, self.bound_high)
        # print(temp)
        temp = reps
        # temp = torch.tanh(reps)
        # temp = torch.clamp(reps, self.bound_low, self.bound_high)
        temp = temp.reshape([-1, self.d, 1, 1])
        onehots = 1.0 - self.i_plus_eta(self.sum_relu(self.c_mat, temp))
        out = torch.reshape(torch.cat([v for v in onehots], axis=1), [-1, int(self.d * self.n_tiles * self.n_tilings)])
        # print('out: ', torch.sum(out>0))
        return out

    def sum_relu(self, c, x):
        out = functional.relu(c - x) + functional.relu(x - self.delta - c)
        # print('sum_relu: ', out)
        return out

    def i_plus_eta(self, x):
        out = (x <= self.eta).type(torch.float32) * x + (x > self.eta).type(torch.float32)
        return out


class ActvFactory:

    @classmethod
    def get_activation_fn(cls, cfg):
        # Creates a function for constructing the value value_network
        if cfg.activation_config['name'] == 'None':
            return lambda x:x
        elif cfg.activation_config['name'] == 'LTA':
            return LTA(cfg)
        else:
            raise NotImplementedError
