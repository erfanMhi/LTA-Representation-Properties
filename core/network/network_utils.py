import torch.nn as nn


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def layer_init_zero(layer):
    nn.init.constant_(layer.weight, 0)
    return layer


def layer_init_xavier(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_lta(layer):
    nn.init.uniform_(layer.weight, -0.003, 0.003)
    nn.init.constant_(layer.bias.data, 0)
    return layer
