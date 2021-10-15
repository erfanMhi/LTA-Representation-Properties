import torch
from core.network.constraint import *


class OptFactory:
    @classmethod
    def get_optimizer_fn(cls, cfg):
        if cfg.optimizer_type == 'SGD':
            return lambda params: torch.optim.SGD(params, cfg.learning_rate)
        elif cfg.optimizer_type == 'Adam':
            return lambda params: torch.optim.Adam(params, cfg.learning_rate)
        elif cfg.optimizer_type == 'RMSProp':
            return lambda params: torch.optim.RMSprop(params, cfg.learning_rate)
        else:
            raise NotImplementedError

    @classmethod
    def get_vf_loss_fn(cls, cfg):
        if cfg.vf_loss == 'mse':
            return torch.nn.MSELoss
        elif cfg.vf_loss == 'huber':
            return torch.nn.SmoothL1Loss
        else:
            raise NotImplementedError

    @classmethod
    def get_constr_fn(cls, cfg):
        if cfg.vf_constraint is None:
            return lambda: NullConstraint(0)
        elif cfg.vf_constraint['type'] == "sparse":
            assert cfg.val_fn_config["val_fn_type"] == "linear", "Sparse constraint only used when the value function is linear"
            return lambda: Sparse(cfg.vf_constraint['weight'], cfg.sparse_level)
        elif cfg.vf_constraint['type'] == "diverse":
            return lambda: Diverse(cfg.vf_constraint['weight'], cfg.vf_constraint['group_size'])
        else:
            raise NotImplementedError
