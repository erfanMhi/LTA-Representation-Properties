import torch
import numpy as np
from core.utils import torch_utils

class NullConstraint():
    def __init__(self, weight):
        self.weight = weight
        return

    def __call__(self, *argv):
        return 0


class Sparse(NullConstraint):
    def __init__(self, weight, beta):
        super().__init__(weight)
        self.beta = beta
        self.log_beta = np.log(self.beta)

    # def __call__(self, q, target, phi):
    #     beta_hat = torch.mean(phi, dim=0)
    #     non_zeros = torch.where(beta_hat > self.beta)[0]
    #     skl = torch.zeros(phi.size()[1])
    #     skl[non_zeros] = torch.log(beta_hat[non_zeros]) + self.beta/beta_hat[non_zeros] - self.log_beta - 1
    #     skl = skl.sum()
    #     constr = self.weight * skl
    #     return constr

    def __call__(self, value, target, phi):
        beta_computed = torch.mean(phi, 0)
        mask = beta_computed.gt(self.beta)
        beta_hat = torch.masked_select(beta_computed, mask)
        beta_vec = torch.zeros([len(beta_hat)]).fill_(self.beta)
        beta_div = torch.div(beta_vec, beta_hat)
        beta_div_forLog = torch.log(beta_div)
        beta_val = torch.sub(beta_div, beta_div_forLog)
        one_vec = torch.zeros([len(beta_hat)]).fill_(1.)
        beta_final = torch.sub(beta_val, one_vec)
        sparsity_loss = self.weight * beta_final.sum()
        return sparsity_loss

class Diverse(NullConstraint):
    def __init__(self, weight, group_size):
        super().__init__(weight)
        self.group_size = group_size

    def __call__(self, q_all, target_all, phi_all):
        repeat = int(np.ceil(len(q_all) / self.group_size))
        rmd = np.remainder(len(q_all), self.group_size)
        combs = int(self.group_size * (self.group_size - 1) / 2 * np.floor(len(q_all) / self.group_size) + \
                    rmd * (rmd - 1) / 2)
        diff_v_all = torch.zeros(combs)
        diff_phi_all = torch.zeros(combs)
        count = 0
        for loop in range(repeat):
            # random = np.random.choice(list(range(len(phi_all))), size=self.group_size, replace=False)
            idx = [k for k in range(self.group_size*loop, min(len(phi_all), self.group_size*(loop+1)))]
            phis = phi_all[idx]
            qs = q_all[idx]

            for i in range(len(phis)):
                for j in range(i + 1, len(phis)):
                    phi_i, phi_j = phis[i], phis[j]
                    qi, qj = qs[i], qs[j]
                    diff_v_all[count] = torch.abs(qi - qj)
                    diff_phi_all[count] = torch.norm(phi_i - phi_j)
                    count += 1
        normalized_dv = torch.nn.functional.normalize(diff_v_all, dim=0)
        normalized_dphi = torch.nn.functional.normalize(diff_phi_all, dim=0)

        # Removing the indexes with zero value of representation difference
        nonzero_idx = normalized_dphi!=0
        normalized_dv = normalized_dv[nonzero_idx]
        normalized_dphi = normalized_dphi[nonzero_idx]

        specialize = torch.mean(torch.clamp(torch.div(normalized_dv, normalized_dphi), 0, 1))
        # print(normalized_dv, normalized_dphi, specialize,"\n")
        return self.weight * specialize
        # normalized_div = 1 - specialize  # 1 - specialization
        # return self.weight * normalized_div
