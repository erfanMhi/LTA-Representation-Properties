import torch

from core.network.activations import LTA
import numpy as np 

cfg = {
        "epsilon": 0.1,
        "rep_config": {
            "rep_type": "modular",
            "network_type": "conv",
            "conv_architecture": {
                "conv_layers": [
                   {"in": 3,  "out": 16, "kernel": 4, "stride": 2, "pad": 2},
                   {"in": 16,  "out": 16, "kernel": 4, "stride": 2, "pad": 2},
                   {"in": 16,  "out": 8, "kernel": 4, "stride": 2, "pad": 2}
                ]
            },
            "in_dim": [15, 15, 3],
            "out_dim": 32,
        },

        "activation_config": {
            "name": "LTA",
            "input": 1,
            "tile": 20,
            "eta": 2.0,
            "bound_high": 20,
            "bound_low": -20
        },
        "val_fn_config": {
            "val_fn_type": "linear",
            "init_type": "lta"
        },
        "memory_size": 10000,
        "batch_size": 1,
    }

class CFG(object):
    pass

c = CFG()
c.__dict__.update(cfg)
cfg = c

print(cfg.rep_config)

lta = LTA(cfg)

rep = torch.tensor([[4.5]])

print(lta(rep))
