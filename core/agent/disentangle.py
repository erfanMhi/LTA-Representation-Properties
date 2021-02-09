import os
import torch
import math
import numpy as np
from scipy import stats

def disentangle(cfg):
    class Disentangle():
        def __init__(self, cfg):
            self.cfg = cfg
            self.linear_fn = self.cfg.linear_fn()

        def load_linear_probing(self):
            parameters_dir = self.cfg.get_parameters_dir()
            path = os.path.join(parameters_dir, "linear_probing")
            self.linear_fn.load_state_dict(torch.load(path))

        def check_disentangle(self):
            w = np.absolute(self.linear_fn.fc_head.weight.data.numpy())
            sum_r = np.sum(w, axis=1)
            p = w / (sum_r.reshape((-1, 1)) * np.ones((1, w.shape[1])))
            rho = np.sum(w, axis=0) / np.sum(w)
            d = 1 + np.sum(np.multiply(p, np.log(p) / np.log(p.shape[0])), axis=0)
            score = np.multiply(rho, d).sum()
            print("Disentangle score", score)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(-1 * np.multiply(p, np.log(p) / np.log(p.shape[0])))
            plt.colorbar()
            plt.show()

    return Disentangle(cfg)
