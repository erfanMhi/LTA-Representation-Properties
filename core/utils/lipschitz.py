import os
import numpy as np
import torch

from core.utils.torch_utils import tensor


def compute_lipschitz(cfg, rep_net, val_net, env):
    try:
        _tensor = lambda x: tensor(x, cfg.device)
        states, _, _, _ = env.get_visualization_segment()
        states = cfg.state_normalizer(states)

        with torch.no_grad():
            phi_s = _tensor(rep_net(states))
            values = val_net(phi_s)

        num_states = len(states)
        N = num_states * (num_states - 1) // 2
        diff_v = np.zeros(N)
        diff_phi = np.zeros(N)

        idx = 0
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                phi_i, phi_j = phi_s[i], phi_s[j]
                vi, vj = values[i], values[j]
                diff_v[idx] = torch.abs(vi - vj).max().item()
                diff_phi[idx] = np.linalg.norm((phi_i - phi_j).numpy())
                idx += 1
        ratio_dv_dphi = np.divide(diff_v, diff_phi, out=np.zeros_like(diff_phi), where=diff_phi != 0)
        return val_net.compute_lipschitz_upper(), ratio_dv_dphi, np.corrcoef(diff_v, diff_phi)[0][1]
    except NotImplementedError:
        return val_net.compute_lipschitz_upper(), 0.0, 0.0


def compute_dynamics_awareness(cfg, rep_net):
    def dist_difference(base_rep, similar_rep, different_idx):
        if type(base_rep) == torch.Tensor:
            base_rep = base_rep.data.numpy()
        if type(similar_rep) == torch.Tensor:
            similar_rep = similar_rep.data.numpy()
        similar_dist = np.linalg.norm(similar_rep - base_rep, axis=1).mean()
        diff_rep1 = base_rep[different_idx[:, 0]]
        diff_rep2 = base_rep[different_idx[:, 1]]
        diff_dist = np.linalg.norm(diff_rep1 - diff_rep2, axis=1).mean()
        prop = (diff_dist - similar_dist) / diff_dist
        if np.isinf(prop) or np.isnan(prop) or prop < 0:
            prop = 0
        return prop
    base_obs = np.load(os.path.join(cfg.data_root, cfg.distance_path["current"]))
    similar_obs = np.load(os.path.join(cfg.data_root, cfg.distance_path["next"]))
    samples = len(base_obs)
    different_idx = np.random.randint(samples, size=samples*2).reshape((samples, 2))
    with torch.no_grad():
        base_rep = rep_net(cfg.state_normalizer(base_obs))
        similar_rep = rep_net(cfg.state_normalizer(similar_obs))
    prop = dist_difference(base_rep, similar_rep, different_idx)
    return prop


def compute_decorrelation(cfg, rep_net, env):
    _tensor = lambda x: tensor(x, cfg.device)
    states, _, _, _ = env.get_visualization_segment()
    states = cfg.state_normalizer(states)

    with torch.no_grad():
        representations = rep_net(states).numpy()
        correlation_matrix = np.corrcoef(representations.transpose(1, 0))
        correlation_matrix[np.tril_indices(32)] = 0.0
        correlation_matrix = np.abs(correlation_matrix)
        total_correlation = np.sum(np.abs(correlation_matrix))
        total_off_diag_upper = 32 * 31 / 2 # N(N-1)/2
        average_correlation = total_correlation / total_off_diag_upper
    return 1 - average_correlation





    # try:
    #     _tensor = lambda x: tensor(x, cfg.device)
    #     states, _, _, _ = env.get_visualization_segment()
    #     states = cfg.state_normalizer(states)
    #
    #     with torch.no_grad():
    #         phi_s = _tensor(rep_net(states))
    #         values = val_net(phi_s)
    #
    #     num_states = len(states)
    #     N = num_states * (num_states - 1) // 2
    #     diff_v = np.zeros(N)
    #     diff_phi = np.zeros(N)
    #
    #     idx = 0
    #     for i in range(len(states)):
    #         for j in range(i + 1, len(states)):
    #             phi_i, phi_j = phi_s[i], phi_s[j]
    #             vi, vj = values[i], values[j]
    #             diff_v[idx] = torch.abs(vi - vj).max().item()
    #             diff_phi[idx] = np.linalg.norm((phi_i - phi_j).numpy())
    #             idx += 1
    #     ratio_dv_dphi = np.divide(diff_v, diff_phi, out=np.zeros_like(diff_phi), where=diff_phi != 0)
    #     return val_net.compute_lipschitz_upper(), ratio_dv_dphi, np.corrcoef(diff_v, diff_phi)[0][1]
    # except NotImplementedError:
    #     return val_net.compute_lipschitz_upper(), 0.0, 0.0



# def compute_lipschitz(cfg, rep_net, val_net, env):
#     N = 10000
#     rng = np.random.RandomState(0)
#
#     def generate_perturbation(r=1):
#         u = rng.normal(size=rep_net.output_dim)
#         # u = rng.normal(size=np.prod(cfg.state_dim))
#         norm = np.linalg.norm(u)
#         if norm == 0.0:
#             return u
#         r = r * rng.rand() ** (1./np.prod(rep_net.output_dim))
#         # r = r * rng.rand() ** (1. / np.prod(cfg.state_dim))
#         u = u * r / norm
#         return u
#
#     _tensor = lambda x: tensor(x, cfg.device)
#
#     states, _, _, _ = env.get_visualization_segment()
#     states = cfg.state_normalizer(states)
#
#     phi_s = _tensor(rep_net.phi(states))
#     values = val_net(phi_s)
#     R = torch.max(torch.sqrt(torch.sum((phi_s ** 2), dim=1))).item()
#
#     ratio_dv_dphi = np.zeros(N)
#     diff_v = np.zeros(N)
#     diff_phi = np.zeros(N)
#
#     for i in range(N):
#         k = rng.randint(len(states))
#         p = phi_s[k]
#         vp = values[k]
#
#         perturb = _tensor(generate_perturbation(R))
#         # q = rep_net.phi(_tensor(states[k]) + perturb)
#         q = _tensor(p) + perturb
#         vq = val_net(q.unsqueeze(0))
#
#         diff_v[i] = torch.abs(vp - vq).max().item()
#         diff_phi[i] = np.linalg.norm((q - p).numpy())
#         ratio_dv_dphi[i] = diff_v[i]/diff_phi[i]
#     return val_net.compute_lipschitz_upper(), ratio_dv_dphi, np.corrcoef(diff_v, diff_phi)[0][1]





