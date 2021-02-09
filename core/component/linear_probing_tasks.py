import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import seaborn as sns

from core.network import network_architectures
from core.utils import torch_utils
from core.network import optimizer

def get_linear_probing_task(cfg):
    lp_tasks = []
    for task in cfg.linearprob_tasks:
        fns = {}
        if task["loss"] == "mse":
            loss = torch.nn.MSELoss()
            # loss = torch.nn.functional.mse_loss
            linear_fn = network_architectures.LinearNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                   task["truth_end"] - task["truth_start"])
        elif task["loss"] == "cross-entropy":
            loss = torch.nn.CrossEntropyLoss()
            linear_fn = network_architectures.LinearNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                   (task["truth_end"] - task["truth_start"]) * task["num_class"])
        else:
            raise NotImplementedError


        fns["loss_fn"] = loss
        fns["linear_fn"] = linear_fn
        fns["task"] = task["task"]
        fns["truth_start"] = task["truth_start"]
        fns["truth_end"] = task["truth_end"]
        if task["task"] == "color":
            fns["num_class"] = task["num_class"]

        lp_tasks.append(fns)
    return lp_tasks
