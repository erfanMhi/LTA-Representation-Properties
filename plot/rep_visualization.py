import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap.umap_ as umap
import umap.plot

sys.path.insert(0, '../')
from plot.plot_utils import *

from core.environment import env_factory
from core.network import activations
from core.utils import normalizer
from core.component import representation
from experiment.sweeper import Sweeper

# os.chdir("..")
print("Change dir to", os.getcwd())

def load_cfg(json, param, run_num):
    project_root = os.path.abspath(os.path.dirname(__file__))
    cfg = Sweeper(project_root, json).parse(param)
    cfg.rep_config["path"] = cfg.exp_name+"/{}_run/{}_param_setting/parameters/rep_net_earlystop".format(run_num, param)
    cfg.rep_config["load_params"] = True
    cfg.rep_activation_fn = activations.ActvFactory.get_activation_fn(cfg)
    cfg.rep_fn = representation.RepFactory.get_rep_fn(cfg)
    cfg.env_fn = env_factory.EnvFactory.create_env_fn(cfg)
    cfg.state_normalizer, cfg.reward_normalizer = normalizer.NormalizerFactory.get_normalizer(cfg)
    return cfg

def load_rep_fn(cfg):
    path = os.path.join(cfg.data_root, cfg.rep_config['path'])
    rep_net = cfg.rep_fn()
    rep_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return rep_net

def get_reps(cfg, rep_fn):
    segment = cfg.env_fn().get_visualization_segment()
    states, state_coords, goal_states, goal_coords = segment
    assert len(states) == len(state_coords)

    f_s = rep_fn(cfg.state_normalizer(states)).detach().numpy()
    f_g = rep_fn(cfg.state_normalizer(goal_states)).detach().numpy()
    f_s = f_s.reshape((len(f_s), -1))
    f_g = f_g.reshape((len(f_g), -1))
    return states, state_coords, goal_states, goal_coords, f_s, f_g

def distance_to_goal(f_states, f_goal, size_x, size_y, state_coords):
    # l2_vec = ((f_states - f_goal) ** 2).sum(axis=1)  # .sum(axis=1).sum(axis=1)
    l2_vec = [np.linalg.norm(f_states[i] - f_goal) for i in range(len(f_states))]
    distance_map = np.zeros((size_x, size_y))
    for k, xy_coord in enumerate(state_coords):
        x, y = xy_coord
        distance_map[x][y] = l2_vec[k]
    return distance_map

def pair_distance(f_states, f_goal, size_x, size_y, state_coords):
    l2_vec = lambda f1, f2: np.linalg.norm(f1 - f2)
    distance_map = np.zeros((size_x, size_y))
    for k, xy_coord in enumerate(state_coords):
        x, y = xy_coord
        neighboor_dist = []
        for neighbor in [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]:
            idx = np.where(np.all(state_coords == neighbor, axis=1) == True)[0]
            if len(idx)>0:
                assert len(idx) == 1
                idx = idx[0]
                d = l2_vec(f_states[k], f_states[idx])
                neighboor_dist.append(d)
        distance_map[x][y] = np.average(np.array(neighboor_dist))
    return distance_map

def distance_heatmap(mode, state_coords, goal_states, goal_coords, reps, goal_rep, title):
    if mode=="to_goal":
        dist_fn = distance_to_goal
    elif mode=="pair":
        dist_fn = pair_distance
    else:
        raise NotImplementedError

    max_x = max(state_coords, key=lambda xy: xy[0])[0]
    max_y = max(state_coords, key=lambda xy: xy[1])[1]

    if mode == "to_goal":
        fig, ax = plt.subplots(nrows=1, ncols=len(goal_states), figsize=(3 * len(goal_states), 2.5))
        ax = [ax] if len(goal_states) == 1 else ax
        max_dist = 0
        for g_k in range(len(goal_states)):
            _distance_map = dist_fn(reps, goal_rep[g_k], max_x + 1, max_y + 1, state_coords)
            max_dist = np.max(_distance_map) if np.max(_distance_map) > max_dist else max_dist
        for g_k in range(len(goal_states)):
            _distance_map = dist_fn(reps, goal_rep[g_k], max_x + 1, max_y + 1, state_coords)
            im = ax[g_k].imshow(_distance_map, cmap="Blues", vmin=0, vmax=max_dist)
            ax[g_k].set_title('Goal: {}'.format(goal_coords[g_k]))
            ax[g_k].text(goal_coords[g_k][1], goal_coords[g_k][0], "x", ha="center", va="center", color="red")
            # ax[g_k].axis('off')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5))
        _distance_map = dist_fn(reps, goal_rep[0], max_x + 1, max_y + 1, state_coords)
        im = ax.imshow(_distance_map, cmap="Blues")

    fig.colorbar(im)
    fig.tight_layout(w_pad=4.0)
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    return

def draw_umap(state_coords, goal_coords, reps, coord2id, id2rank, ax, seed=0):
    # mapper = umap.UMAP().fit(reps)
    # p = umap.plot.points(mapper)
    # umap.plot.show(p)

    # for si,seed in enumerate(seeds):
    fit = umap.UMAP(random_state=seed, min_dist=0.01, spread=0.75)
    u = fit.fit_transform(reps)
    im = ax.scatter(u[:, 0], u[:, 1], c=np.array([id2rank[coord2id[(s[0], s[1])]] for s in state_coords]),
                        s=3, cmap="winter", vmin=-1, vmax=172)
    for g in range(len(goal_coords)):
    # for g in range(1):
        gidx = np.where(np.all(state_coords == goal_coords[g], axis=1) == True)[0]
        print("Show goal {} with a different color".format(gidx))
        assert len(gidx) == 1
        gidx = gidx[0]
        ax.scatter(u[gidx, 0], u[gidx, 1], c="orange", s=3)
        ax.text(u[gidx, 0], u[gidx, 1], "{}".format(id2rank[coord2id[tuple(state_coords[gidx])]]),color="orangered",size=7)
    return im

def exchange_kv(dictionary):
    res = dict((v, k) for k, v in dictionary.items())
    return res

def visualize_distance(json, param, run_num, title, mode="to_goal"):
    cfg = load_cfg(json, param, run_num)
    rep_fn = load_rep_fn(cfg)
    states, state_coords, goal_states, goal_coords, reps, goal_rep = get_reps(cfg, rep_fn)
    distance_heatmap(mode, state_coords, goal_states, goal_coords, reps, goal_rep, title)

def visualize_umap(json, param, runs=[0], title="umap"):
    id2rank = np.load("../data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for id in id2rank:
        id2rank[id] += 1  # format rank
    id2coord = np.load("../data/dataset/gridhard/srs/goal(9, 9)_id2coord.npy", allow_pickle=True).item()
    coord2id = exchange_kv(id2coord)

    fig, ax = plt.subplots(nrows=1, ncols=len(runs), figsize=(3 * len(runs), 2.5))
    for ri, run_num in enumerate(runs):
        cfg = load_cfg(json, param, run_num)
        rep_fn = load_rep_fn(cfg)
        states, state_coords, goal_states, goal_coords, reps, goal_rep = get_reps(cfg, rep_fn)
        im = draw_umap(state_coords, goal_coords, reps, coord2id, id2rank, ax[ri], seed=32)
        ax[ri].set_title("Rep Seed {}".format(run_num))

    fig.colorbar(im)
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # visualize_distance("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn/sweep.json", 1, 0, "distance2goal-relu")
    # visualize_distance("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/successor_as/sweep.json", 1, 4, "distance2goal-relu+sf")
    # visualize_distance("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_sweep.json", 1, 0, "distance2goal-fta")
    # visualize_distance("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep.json", 2, 0, "distance2goal-fta+sf")

    # visualize_distance("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn/sweep.json", 1, 0, "distancepair-relu", mode="pair")
    # visualize_distance("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/successor_as/sweep.json", 1, 4, "distancepair-relu+sf", mode="pair")
    # visualize_distance("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_sweep.json", 1, 0, "distancepair-fta", mode="pair")
    # visualize_distance("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep.json", 2, 0, "distancepair-fta+sf", mode="pair")

    runs = list(range(5))
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn/sweep.json", 1, runs, "umap-relu")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_1g_gamma0.9.json", 1, runs, "umap-relu+VirtualVF1")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/aux_control/sweep_5g_gamma0.9.json", 0, runs, "umap-relu+VirtualVF5")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/info/sweep.json", 1, runs, "umap-relu+info")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/input_decoder/sweep.json", 1, runs, "umap-relu+decoder")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/nas_v2_delta/sweep.json", 1, runs, "umap-relu+nas")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/reward/sweep.json", 0, runs, "umap-relu+rwd")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_aux/successor_as/sweep.json", 1, runs, "umap-relu+sf")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_sweep.json", 1, runs, "umap-fta")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.4_sweep.json", 1, runs, "umap-fta0.4")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.6_sweep.json", 1, runs, "umap-fta0.6")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_sweep.json", 2, runs, "umap-fta0.8")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_1g_gamma0.9.json", 2, runs, "umap-fta+VirtualVF5")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/aux_control/sweep_5g_gamma0.9.json", 2, runs, "umap-fta+VirtualVF1")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/info/sweep.json", 1, runs, "umap-fta+info")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/sweep.json", 2, runs, "umap-fta+decoder")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/sweep.json", 2, runs, "umap-fta+nas")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/reward/sweep.json", 1, runs, "umap-fta+rwd")
    # visualize_umap("experiment/config/test_v13/gridhard/linear_vf/original_0909/online_property/dqn_lta_aux/successor_as/sweep.json", 2, runs, "umap-fta+sf")
    
    visualize_umap("../experiment/config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep.json", 1, runs, "umap-relu")