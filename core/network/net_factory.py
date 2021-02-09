import numpy as np

from core.network import network_architectures


class NetFactory:

    @classmethod
    def get_val_fn(cls, cfg):
        # Creates a function for constructing the value value_network
        if cfg.val_fn_config['val_fn_type'] == 'fc':
            return lambda: network_architectures.FCNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                           cfg.val_fn_config['hidden_units'], cfg.action_dim)
        elif cfg.val_fn_config['val_fn_type'] == 'conv':
            return lambda: network_architectures.ConvNetwork(cfg.device, cfg.state_dim,
                                                             cfg.action_dim, cfg.val_fn_config['conv_architecture'])
        elif cfg.val_fn_config['val_fn_type'] == 'linear':
            return lambda: network_architectures.LinearNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                               cfg.action_dim)
        else:
            raise NotImplementedError
