import torch
import numpy as np

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

    def __call__(self, q, target, phi):
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