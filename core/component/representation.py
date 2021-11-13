import os
import numpy as np
import torch
import torch.nn as nn

from core.network import network_architectures
from core.utils import torch_utils

class DefaultRepresentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        net = self.create_network(cfg)
        self.net = net
        self.output_dim = cfg.rep_config['out_dim']

    def forward(self, x):
        return self.net(x)


class OnlineRepresentation(DefaultRepresentation):
    def __init__(self, cfg):
        super().__init__(cfg)

    def create_network(self, cfg):
        if cfg.rep_config['network_type'] == 'fc':
            return network_architectures.FCBody(cfg.device, np.prod(cfg.rep_config['in_dim']),
                                                   cfg.rep_config['hidden_units'], cfg.rep_config['out_dim'])

        elif cfg.rep_config['network_type'] == 'conv': 
            return network_architectures.ConvBody(cfg.device, cfg.rep_config['in_dim'],
                                                     cfg.rep_config['out_dim'], cfg.rep_config['conv_architecture'])
        elif cfg.rep_config['network_type'] is None:
            return lambda x: x
        else:
            raise NotImplementedError


class ModularRepresentation(DefaultRepresentation):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.activation_config['name'] == "None":
            self.output_dim = cfg.rep_config['out_dim']
        elif cfg.activation_config['name'] == 'LTA':
            self.output_dim = cfg.rep_config['out_dim'] * cfg.activation_config['tile']
        else:
            raise NotImplementedError

    def create_network(self, cfg):
        if cfg.rep_config['network_type'] == 'fc':
            return network_architectures.FCNetwork(cfg.device, np.prod(cfg.rep_config['in_dim']),
                                                   cfg.rep_config['hidden_units'], cfg.rep_config['out_dim'],
                                                   head_activation=cfg.rep_activation_fn)

        elif cfg.rep_config['network_type'] == 'conv':
            return network_architectures.ConvNetwork(cfg.device, cfg.rep_config['in_dim'],
                                                     cfg.rep_config['out_dim'], cfg.rep_config['conv_architecture'],
                                                  head_activation=cfg.rep_activation_fn)
        elif cfg.rep_config['network_type'] is None:
            return lambda x: x
        else:
            raise NotImplementedError


class IdentityRepresentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_dim = cfg.rep_config['out_dim']
        self.device = cfg.device

    def forward(self, x):
        return torch_utils.tensor(x.reshape(-1, 675), self.device)


class RepFactory:
    @classmethod
    def get_rep_fn(cls, cfg):
        # Creates a function for constructing the value value_network
        if cfg.rep_config['rep_type'] == 'online':
            return lambda: OnlineRepresentation(cfg)
        elif cfg.rep_config['rep_type'] == 'modular':
            return lambda: ModularRepresentation(cfg)
        elif cfg.rep_config['rep_type'] == 'identity':
            return lambda: IdentityRepresentation(cfg)
        else:
            raise NotImplementedError