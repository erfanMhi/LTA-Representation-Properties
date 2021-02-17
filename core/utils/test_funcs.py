import time
import numpy as np
import pickle as pkl
import torch
import os
from itertools import combinations
from core.environment.gridworlds import GridHardXY
import copy
from core.utils import torch_utils
import string


def dqn_rep_distance_viz(agent):
    agent.visualize()


def generate_noisy_dataset(env):
    # state_coords = [[x, y] for x in range(15)
    #                 for y in range(15) if not int(env.obstacles_map[x][y])]
    # states = [env.generate_state(coord) for coord in state_coords]

    space_x, space_y = np.where(env.obstacles_map[:15, :15] == 0)
    space = np.concatenate((space_x.reshape(-1, 1), space_y.reshape(-1, 1)), axis=1)
    noise1, noise2 = [], []
    for coord in space:
        noise1.append(env.generate_state(coord))
        noise2.append(env.generate_state(coord))
    noise1, noise2 = np.array(noise1), np.array(noise2)
    return noise1, noise2


def noisy_difference(rep1, rep2):
    diff = np.mean(np.linalg.norm(rep1 - rep2, axis=1)) / np.mean(np.linalg.norm(rep1, axis=1))
    # if np.isnan(diff):
    #     diff = 1
    return diff


# def test_laplace_changeNoise(agent):
#     agent.load(os.path.join(agent.cfg.data_root, agent.cfg.rep_config['path']))
#     env = agent.env
#     noise1, noise2 = generate_noisy_dataset(env)
#     with torch.no_grad():
#         rep1 = agent.rep_net(agent.cfg.state_normalizer(noise1))
#         rep2 = agent.rep_net(agent.cfg.state_normalizer(noise2))
#     diff = noisy_difference(rep1, rep2)
#     with open(os.path.join(agent.cfg.get_parameters_dir(), "../denoise.txt"), "w") as f:
#         f.write("Parameter Beta={}, Delta={}, Alpha={}. Change {:.8f}"
#           .format(agent.cfg.beta, agent.cfg.delta, agent.cfg.learning_rate, diff))

def test_dqn_changeNoise(agent):
    env = agent.env
    noise1, noise2 = generate_noisy_dataset(env)
    with torch.no_grad():
        rep1 = agent.rep_net(agent.cfg.state_normalizer(noise1))
        rep2 = agent.rep_net(agent.cfg.state_normalizer(noise2))
    diff = noisy_difference(rep1, rep2)
    # agent.cfg.logger.info("Parameter Sync={}, Alpha={}. Change {:.8f}"
    #       .format(agent.cfg.target_network_update_freq, agent.cfg.learning_rate, diff))
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../denoise.txt"), "w") as f:
        f.write("Parameter Sync={}, Alpha={}. Change {:.8f}"
          .format(agent.cfg.target_network_update_freq, agent.cfg.learning_rate, diff))

def test_randomNN_changeNoise(agent):
    env = agent.env
    noise1, noise2 = generate_noisy_dataset(env)
    with torch.no_grad():
        rep1 = agent.rep_net(agent.cfg.state_normalizer(noise1))
        rep2 = agent.rep_net(agent.cfg.state_normalizer(noise2))
    diff = noisy_difference(rep1, rep2)
    # agent.cfg.logger.info("Random representation baseline. Change {:.8f}".format(diff))
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../denoise.txt"), "w") as f:
        f.write("Random representation baseline. Change {:.8f}".format(diff))

def test_input_changeNoise(agent):
    env = agent.env
    noise1, noise2 = generate_noisy_dataset(env)
    rep1 = agent.cfg.state_normalizer(noise1)
    rep2 = agent.cfg.state_normalizer(noise2)
    diff = noisy_difference(rep1, rep2)
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../denoise.txt"), "w") as f:
        f.write("Random representation baseline. Change {:.8f}".format(diff))

def generate_distance_dataset(cfg):
    # base_obs = np.load(os.path.join(cfg.data_root, cfg.distance_path["current"].replace("distance_current_states.npy", "distance_current_states_sameEP.npy")))
    # similar_obs = np.load(os.path.join(cfg.data_root, cfg.distance_path["next"].replace("distance_next_states.npy", "distance_next_states_sameEP.npy")))
    # actions = np.load(os.path.join(cfg.data_root, cfg.distance_path["action"].replace("distance_actions.npy", "distance_actions_sameEP.npy")))
    # rewards = np.load(os.path.join(cfg.data_root, cfg.distance_path["reward"].replace("distance_rewards.npy", "distance_rewards_sameEP.npy")))
    # terminal = np.load(os.path.join(cfg.data_root, cfg.distance_path["terminal"].replace("distance_terminals.npy", "distance_terminals_sameEP.npy")))
    base_obs = np.load(os.path.join(cfg.data_root, cfg.distance_path["current"]))
    similar_obs = np.load(os.path.join(cfg.data_root, cfg.distance_path["next"]))
    actions = np.load(os.path.join(cfg.data_root, cfg.distance_path["action"]))
    rewards = np.load(os.path.join(cfg.data_root, cfg.distance_path["reward"]))
    terminal = np.load(os.path.join(cfg.data_root, cfg.distance_path["terminal"]))
    samples = len(base_obs)
    different_idx = np.random.randint(samples, size=samples*2).reshape((samples, 2))
    return base_obs, similar_obs, different_idx, actions, rewards, terminal

def generate_distance_dataset_random(cfg, env):

    random_set = os.path.join(cfg.data_root,
                 cfg.distance_path["current"].replace("distance_current_states.npy", "random_data.pkl"))
    if os.path.isfile(random_set):
        with open(random_set, "rb") as f:
            savef = pkl.load(f)
        base_obs = savef[0]
        similar_obs = savef[1]
        different_idx_all = savef[2]
    else:
        xyenv = env
        timeout = 1000
        base = []
        s = xyenv.reset()
        sequence = [s]
        for _ in range(timeout):
            s, _, terminal, _ = xyenv.step([np.random.randint(len(xyenv.actions))])
            if not terminal:
                sequence.append(s)
            else:
                base.append(sequence)
                s = xyenv.reset()
                sequence = [s]
        base.append(sequence)

        base_obs = np.concatenate([np.array(seq[:-1]) for seq in base], axis=0)
        similar_obs = np.concatenate([np.array(seq[1:]) for seq in base], axis=0)
        base_len = 0
        different_idx_all = np.zeros((0, 2))
        for i in range(len(base)):
            given_idx = base_len + np.array(range(len(base[i]) - 1)).reshape((-1, 1))
            different_idx = base_len + np.random.randint(len(base[i]) - 1, size=len(base[i]) - 1).reshape((-1, 1))
            diff_idx = np.concatenate((given_idx, different_idx), axis=1)
            base_len += len(base[i]) - 1
            different_idx_all = np.concatenate((different_idx_all, diff_idx), axis=0).astype(int)
        savef = [base_obs, similar_obs, different_idx_all]
        with open(random_set, "wb") as f:
            pkl.dump(savef, f)

    return base_obs, similar_obs, different_idx_all, None, None, None

# def generate_distance_dataset(env):
#     xyenv = env#GridHardXY(2048)
#     timeout = 1
#     samples = 1000
#     tau_probs = [0.9**(x-1)-0.9**x for x in range(1, timeout+1)]
#     tau_probs = [x / np.sum(tau_probs) for x in tau_probs]
#     base = []
#     similar = []
#     for _ in range(samples):
#         xyenv.reset()
#         base.append(xyenv.get_useful())
#         sequence = []
#         for _ in range(timeout):
#             xyenv.step([np.random.randint(len(xyenv.actions))])
#             sequence.append(xyenv.get_useful())
#         similar.append(sequence[np.random.choice(list(range(len(sequence))), p=tau_probs)])
#     different_idx = np.random.randint(samples, size=samples*2).reshape((samples, 2))
#     # base = [[9, 9], [14, 0], [0, 0], [3, 14], [3, 4]]
#     # similar = [[9, 10], [14, 1], [1, 0], [4, 14], [4, 4]]
#     # different_idx = np.array(list(combinations(list(range(len(base))), 2)))
#
#     base_obs = []
#     similar_obs = []
#     for coord in base:
#         base_obs.append(env.generate_state(coord))
#     for coord in similar:
#         similar_obs.append(env.generate_state(coord))
#     base_obs, similar_obs = np.array(base_obs), np.array(similar_obs)
#     return base_obs, similar_obs, different_idx, None, None, None



def generate_linear_probing_dataset(env, cfg):
    observations = np.load(os.path.join(cfg.data_root, cfg.linearprob_path["test"]))
    info = []
    for o in observations:
        info.append(env.get_useful(o))
    info = np.array(info)
    validation_idx = list(range(0, len(observations), 2))
    test_idx = list(range(1, len(observations), 2))
    validation_obs = observations[validation_idx]
    validation_if = info[validation_idx]
    test_obs = observations[test_idx]
    test_if = info[test_idx]
    return validation_obs, validation_if, test_obs, test_if
# def generate_linear_probing_dataset(env, action_dim, retain):
    # state, coord, action, next_state, next_coord = [], [], [], [], []
    # reset = True
    # for i in range(10000):
    #     if reset:
    #         s = env.reset()
    #         xy = env.get_useful()
    #     a = np.random.randint(0, action_dim)
    #     sp, _, reset, _ = env.step([a])
    #     xy_p = env.get_useful()
    #     state.append(s)
    #     coord.append(xy)
    #     action.append(a)
    #     next_state.append(sp)
    #     next_coord.append(xy_p)
    #     s = sp
    #     xy = xy_p
    #
    # rand_idx = np.random.choice(list(range(len(state))), size=100, replace=False)
    # state = np.array(state)[rand_idx]
    # action = np.array(action)[rand_idx]
    # coord = np.array(coord)[rand_idx]
    # next_coord = np.array(next_coord)[rand_idx]
    # if retain == "next":
    #     return [state, action], next_coord
    # elif retain == "current":
    #     return state, coord


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
def dist_difference_v2(base_rep, similar_rep, different_idx):
    if type(base_rep) == torch.Tensor:
        base_rep = base_rep.data.numpy()
    if type(similar_rep) == torch.Tensor:
        similar_rep = similar_rep.data.numpy()
    similar_dist = np.linalg.norm(similar_rep - base_rep, axis=1)
    diff_rep1 = base_rep[:]
    diff_rep2 = base_rep[different_idx[:, 1]]
    diff_dist = np.linalg.norm(diff_rep1 - diff_rep2, axis=1)
    dn = np.max(np.concatenate((diff_dist.reshape(-1, 1), similar_dist.reshape(-1, 1)), axis=1), axis=1) + 1e-05
    prop = np.nan_to_num((diff_dist - similar_dist) / dn).mean()
    if np.isinf(prop) or np.isnan(prop):# or prop < 0:
        prop = 0
    return prop


# def test_laplace_distance(agent):
#     agent.load(os.path.join(agent.cfg.data_root, agent.cfg.rep_config['path']))
#     env = agent.env
#     # base_obs, similar_obs, different_idx = generate_distance_dataset(env)
#     base_obs, similar_obs, different_idx, _ = generate_distance_dataset(agent.cfg)
#     with torch.no_grad():
#         base_rep = agent.rep_net(agent.cfg.state_normalizer(base_obs))
#         similar_rep = agent.rep_net(agent.cfg.state_normalizer(similar_obs))
#     prop = dist_difference(base_rep, similar_rep, different_idx)
#     # if prop > 0.95:
#     # agent.cfg.logger.info("Parameter Beta={}, Delta={}, Alpha={}. Distance {:.8f}"
#     #       .format(agent.cfg.beta, agent.cfg.delta, agent.cfg.learning_rate, prop))
#     # print("Parameter Beta={}, Delta={}, Alpha={}. Distance {:.8f}"
#     #       .format(agent.cfg.beta, agent.cfg.delta, agent.cfg.learning_rate, prop))
#     with open(os.path.join(agent.cfg.get_parameters_dir(), "../distance.txt"), "w") as f:
#         f.write("Parameter Beta={}, Delta={}, Alpha={}. Distance {:.8f}"
#           .format(agent.cfg.beta, agent.cfg.delta, agent.cfg.learning_rate, prop))
#     return


def test_dqn_distance(agent):
    base_obs, similar_obs, different_idx, _, _, _ = generate_distance_dataset(agent.cfg)
    # base_obs, similar_obs, different_idx, _, _, _ = generate_distance_dataset_random(agent.cfg, agent.env)
    with torch.no_grad():
        base_rep = agent.rep_net(agent.cfg.state_normalizer(base_obs))
        similar_rep = agent.rep_net(agent.cfg.state_normalizer(similar_obs))

    prop = dist_difference(base_rep, similar_rep, different_idx)
    # prop = dist_difference_v2(base_rep, similar_rep, different_idx)
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../distance.txt"), "w") as f:
        f.write("Alpha={}. Distance {:.8f}"
          .format(agent.cfg.learning_rate, prop))
    return
def online_distance(agent, base_obs, similar_obs, different_idx):
    with torch.no_grad():
        base_rep = agent.rep_net(agent.cfg.state_normalizer(base_obs))
        similar_rep = agent.rep_net(agent.cfg.state_normalizer(similar_obs))
    prop = dist_difference(base_rep, similar_rep, different_idx)
    log_str = 'total steps %d, total episodes %3d, ' \
              'Distance: %.8f/'
    agent.cfg.logger.info(log_str % (agent.total_steps, len(agent.episode_rewards), prop))
    return

def run_linear_probing(agent):
    env = agent.env
    validation, truth, _, _ = generate_linear_probing_dataset(env, agent.cfg)
    done = False
    while True:
        agent.step()
        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            done = agent.eval_linear_probing(validation, truth, validate=True)
        if done or (agent.cfg.max_steps and agent.total_steps >= agent.cfg.linear_max_steps):
            agent.save_linear_probing()
            break
    return

def test_linear_probing(agent):
    agent.load_linear_probing()
    env = agent.env
    _, _, testset, truth = generate_linear_probing_dataset(env, agent.cfg)
    agent.eval_linear_probing(testset, truth)
    return


def check_aux(agent):
    t0 = time.time()
    agent.populate_returns()
    while True:
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            t0 = time.time()
        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            agent.eval_episodes()
            if agent.cfg.visualize:
                agent.visualize()
                agent.visualize_distance()
            if agent.cfg.save_params:
                agent.save()
            if agent.cfg.evaluate_lipschitz:
                agent.log_lipschitz()
            t0 = time.time()
        if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
            agent.save()
            break
        agent.step()


def test_orthogonality(agent):
    model = agent.rep_net
    # states = agent.cfg.state_normalizer(traj["states"])
    states, _, _, _, _, _ = generate_distance_dataset(agent.cfg)
    states = agent.cfg.state_normalizer(states)
    rhos = []
    for i in range(10):
        random = np.random.choice(list(range(len(states))), size=100,  replace=False)
        s = states[random]
        reps = model(s)

        # Vincent's thesis
        reps = reps.detach().numpy()
        dot_prod = np.matmul(reps, reps.T)
        norm = np.linalg.norm(reps, axis=1).reshape((-1, 1))
        norm_prod = np.matmul(norm, norm.T)
        normalized = np.abs(np.divide(dot_prod, norm_prod))
        rho = (normalized.sum() - np.diagonal(normalized).sum()) / (normalized.shape[0] * (normalized.shape[0]-1))
        rhos.append(rho)
        # rho = np.sum(reps1 * reps2, axis=1).mean() / (np.linalg.norm(reps1, axis=1).mean() * np.linalg.norm(reps2, axis=1).mean())
    rho = 1 - np.array(rhos).mean()
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../orthogonality.txt"), "w") as f:
        f.write("Orthogonality: {:.8f}".format(rho))
def online_orthogonality(agent, states):
    states = agent.cfg.state_normalizer(states)
    rhos = []
    for i in range(10):
        random = np.random.choice(list(range(len(states))), size=100,  replace=False)
        s = states[random]
        with torch.no_grad():
            reps = agent.rep_net(s)
        # Vincent's thesis
        reps = reps.detach().numpy()
        dot_prod = np.matmul(reps, reps.T)
        norm = np.linalg.norm(reps, axis=1).reshape((-1, 1))
        norm_prod = np.matmul(norm, norm.T)
        normalized = np.abs(np.divide(dot_prod, norm_prod))
        rho = (normalized.sum() - np.diagonal(normalized).sum()) / (normalized.shape[0] * (normalized.shape[0]-1))
        rhos.append(rho)
        # rho = np.sum(reps1 * reps2, axis=1).mean() / (np.linalg.norm(reps1, axis=1).mean() * np.linalg.norm(reps2, axis=1).mean())
    rho = 1 - np.array(rhos).mean()
    log_str = 'total steps %d, total episodes %3d, ' \
              'Orthogonality: %.8f/'
    agent.cfg.logger.info(log_str % (agent.total_steps, len(agent.episode_rewards), rho))


def test_robustness(agent):
    rs = np.random.RandomState(0)
    img, _, _, _, _, _ = generate_distance_dataset(agent.cfg)
    ns_std = 1
    noise1 = rs.normal(loc=0, scale=ns_std, size=np.product(img.shape)).reshape(img.shape)
    noise2 = rs.normal(loc=0, scale=ns_std, size=np.product(img.shape)).reshape(img.shape)
    img_n1 = img + noise1
    img_n2 = img + noise2

    # draw(img_n1[10])
    # draw(img_n2[10])
    # draw(img[10])

    with torch.no_grad():
        img_n1 = agent.cfg.state_normalizer(img_n1)
        img_n2 = agent.cfg.state_normalizer(img_n2)
        rep_n1 = agent.rep_net(img_n1)
        rep_n2 = agent.rep_net(img_n2)
    change = 1 - np.linalg.norm(rep_n1 - rep_n2) / np.linalg.norm(rep_n1)

    with open(os.path.join(agent.cfg.get_parameters_dir(), "../robustness.txt"), "w") as f:
        f.write("Robustness: {:.8f}".format(change))
def online_robustness(agent, img):
    rs = np.random.RandomState(0)
    ns_std = 1
    noise1 = rs.normal(loc=0, scale=ns_std, size=np.product(img.shape)).reshape(img.shape)
    noise2 = rs.normal(loc=0, scale=ns_std, size=np.product(img.shape)).reshape(img.shape)
    img_n1 = img + noise1
    img_n2 = img + noise2
    with torch.no_grad():
        img_n1 = agent.cfg.state_normalizer(img_n1)
        img_n2 = agent.cfg.state_normalizer(img_n2)
        rep_n1 = agent.rep_net(img_n1)
        rep_n2 = agent.rep_net(img_n2)
    change = 1 - np.linalg.norm(rep_n1 - rep_n2) / np.linalg.norm(rep_n1)
    log_str = 'total steps %d, total episodes %3d, ' \
              'Robustness: %.8f/'
    agent.cfg.logger.info(log_str % (agent.total_steps, len(agent.episode_rewards), change))


def test_sparsity(agent):
    img, _, _, _, _, _ = generate_distance_dataset(agent.cfg)
    with torch.no_grad():
        img = agent.cfg.state_normalizer(img)
        rep = agent.rep_net(img).detach().numpy()

    rep = rep.reshape((rep.shape[0], rep.shape[1], 1))
    zeros = np.all(rep==0, axis=2).astype(int)
    # print(np.sum(zeros))
    # print(len(np.where(rep==0)[0]))

    # lifetime sparsity
    lifetime_inact = np.sum(zeros, axis=0)
    num_sample = rep.shape[0]
    lifetime_sparsity = (lifetime_inact / num_sample).mean()

    # instance sparsity
    feature_inact = np.sum(zeros, axis=1)
    num_f = rep.shape[1]
    instance_sparsity = (feature_inact / num_f).mean()

    with open(os.path.join(agent.cfg.get_parameters_dir(), "../sparsity_instance.txt"), "w") as f:
        f.write("Instance sparsity: {:.8f}".format(instance_sparsity))
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../sparsity_lifetime.txt"), "w") as f:
        f.write("Lifetime sparsity: {:.8f}".format(lifetime_sparsity))
    print("Instance sparsity: {:.8f}".format(instance_sparsity))
    print("Lifetime sparsity: {:.8f}".format(lifetime_sparsity))


# def test_dense_sparsity(agent):
#     img, _, _, _, _, _ = generate_distance_dataset(agent.cfg)
#     with torch.no_grad():
#         img = agent.cfg.state_normalizer(img)
#         rep = agent.rep_net(img).detach().numpy()
#
#     rep = rep.reshape((rep.shape[0], rep.shape[1], 1))
#     zeros = np.all(abs(rep) < 0.1, axis=2).astype(int)
#
#     # instance sparsity
#     feature_inact = np.sum(zeros, axis=1)
#     num_f = rep.shape[1]
#     instance_sparsity = (feature_inact / num_f).mean()
#
#     # lifetime sparsity
#     lifetime_inact = np.sum(zeros, axis=0)
#     num_sample = rep.shape[0]
#     lifetime_sparsity = (lifetime_inact / num_sample).mean()
#
#     with open(os.path.join(agent.cfg.get_parameters_dir(), "../sparsity_instance_dense.txt"), "w") as f:
#         f.write("Instance sparsity: {:.8f}".format(instance_sparsity))
#     with open(os.path.join(agent.cfg.get_parameters_dir(), "../sparsity_lifetime_dense.txt"), "w") as f:
#         f.write("Lifetime sparsity: {:.8f}".format(lifetime_sparsity))


def test_dynamic_interference(agent):
    state_all, next_s_all, _, action_all, reward_all, terminal_all = generate_distance_dataset(agent.cfg)
    state_all = agent.cfg.state_normalizer(state_all)
    next_s_all = agent.cfg.state_normalizer(next_s_all)

    action_all = action_all.reshape([-1, 1])
    sample_idx = np.random.choice(list(range(len(state_all))), size=200)
    state_batch = state_all[sample_idx]
    action_batch = action_all[sample_idx]
    next_s_batch = next_s_all[sample_idx]
    reward_batch = reward_all[sample_idx]
    terminal_batch = terminal_all[sample_idx]

    all_param = agent.val_net.parameters()
    param_num = 0
    for p in all_param:
        param_num += np.product(p.size())

    t0 = time.time()
    agent.populate_returns()
    non_interf = {
        "non-interf": [],
        "generalize-total": [],
        "generalize-count": [],
        "interference-total": [],
        "interference-count": []
    }
    non_interf = eval_noninterference(agent, param_num, state_batch, next_s_batch, action_batch, reward_batch, terminal_batch, non_interf)
    while True:
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            t0 = time.time()
        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            agent.eval_episodes()
            non_interf = eval_noninterference(agent, param_num, state_batch, next_s_batch, action_batch, reward_batch, terminal_batch, non_interf)
            if agent.cfg.visualize:
                agent.visualize()
            t0 = time.time()
        if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
            agent.save()
            break
        agent.step()
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../interference_log.pkl"), "wb") as f:
        pkl.dump(non_interf, f)

def test_trainedVF_interf(agent):
    state_all, next_s_all, _, action_all, reward_all, terminal_all = generate_distance_dataset(agent.cfg)
    state_all = agent.cfg.state_normalizer(state_all)
    next_s_all = agent.cfg.state_normalizer(next_s_all)
    action_all = action_all.reshape([-1, 1])

    agent.load(agent.cfg.get_parameters_dir())

    all_param = agent.val_net.parameters()
    param_num = 0
    for p in all_param:
        param_num += np.product(p.size())
    non_interf = {
        "non-interf": [],
        "generalize-total": [],
        "generalize-count": [],
        "interference-total": [],
        "interference-count": []
    }
    itr = 1
    batchsize = 100
    for i in range(itr):
        sample_idx = np.random.choice(list(range(len(state_all))), size=batchsize)
        state_batch = state_all[sample_idx]
        action_batch = action_all[sample_idx]
        next_s_batch = next_s_all[sample_idx]
        reward_batch = reward_all[sample_idx]
        terminal_batch = terminal_all[sample_idx]
        non_interf = eval_noninterference(agent, param_num, state_batch, next_s_batch, action_batch, reward_batch, terminal_batch, non_interf)

    # with open(os.path.join(agent.cfg.get_parameters_dir(), "../interference_test.pkl"), "wb") as f:
    #     pkl.dump(non_interf, f)
    count_g = np.array(non_interf["generalize-count"]).sum()
    avg_g = np.array(non_interf["generalize-total"]).sum() / count_g
    count_i = np.array(non_interf["interference-count"]).sum()
    avg_i = np.array(non_interf["interference-total"]).sum() / count_i
    avg_noninterf = np.array(non_interf["non-interf"]).mean()
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../interference_test.txt"), "w") as f:
        f.write("Percentage generalization: {:.8f}\n".format(count_g / (itr*batchsize*(batchsize-1))))
        f.write("Percentage interference: {:.8f}\n".format(count_i / (itr*batchsize*(batchsize-1))))
        f.write("Expected generalization: {:.8f}\n".format(avg_g))
        f.write("Expected interference: {:.8f}\n".format(avg_i))
        f.write("Non-interference score: {:.8f}\n".format(avg_noninterf))

def eval_noninterference(agent, param_num, state, next_s, action, reward, terminal, record):
    def loss(agent, states, actions, next_states, rewards, terminals):
        # if not agent.cfg.rep_config['train_rep']:
        #     with torch.no_grad():
        #         phi = agent.rep_net(states)
        # else:
        #     phi = agent.rep_net(states)
        with torch.no_grad():
            phi = agent.rep_net(states)

        q = agent.val_net(phi)[list(range(len(actions))), actions]

        # Constructing the target
        with torch.no_grad():
            q_next = agent.targets.val_net(agent.targets.rep_net(next_states))
            q_next = q_next.max(1)[0]
            terminals = torch_utils.tensor(terminals, agent.cfg.device)
            rewards = torch_utils.tensor(rewards, agent.cfg.device)
            target = agent.cfg.discount * q_next * (1 - terminals).float()
            target.add_(rewards.float())
            target = target.view((-1, 1))
        loss = agent.vf_loss(q, target)  # (target - q).pow(2).mul(0.5).mean()
        return loss

    data_size = state.shape[0]
    grad_mat = np.zeros([data_size, param_num])
    for i in range(data_size):
        agent.val_net.zero_grad()
        agent.rep_net.zero_grad()
        l = loss(agent, state[i:(i + 1)], action[i:(i + 1)], next_s[i:(i + 1)], reward[i:(i + 1)], terminal[i:(i + 1)])
        l.backward()
        grad_list = []
        for para in agent.val_net.parameters():
            if para.grad is not None:
                grad_list.append(para.grad.flatten().numpy())
        grad_mat[i] = np.concatenate(grad_list)

    agent.val_net.zero_grad()
    agent.rep_net.zero_grad()
    agent.targets.val_net.zero_grad()
    agent.targets.rep_net.zero_grad()

    grad_mat = np.nan_to_num(grad_mat)
    ntk_mat = np.matmul(grad_mat, grad_mat.T)
    sample_norm = np.linalg.norm(grad_mat, axis=1).reshape((-1, 1))
    norm = np.matmul(sample_norm, sample_norm.T)
    ntk_mat = np.divide(ntk_mat, norm)
    ntk_mat = np.nan_to_num(ntk_mat) # set 0 to where norm=0
    np.fill_diagonal(ntk_mat, 0) # remove diagonal

    generalize = ntk_mat[np.where(ntk_mat > 0)]
    interfer = ntk_mat[np.where(ntk_mat < 0)]
    g_count = len(generalize)
    i_count = len(interfer)

    ntk_mat = np.clip(ntk_mat * (-1), 0, np.inf)
    rho = 1 - (np.sum(ntk_mat) - np.trace(ntk_mat)) / (data_size * (data_size - 1))
    # agent.cfg.logger.tensorboard_writer.add_scalar('dqn/interference', rho, agent.total_steps)
    record["non-interf"].append(rho)
    if g_count > 0:
        g_total = generalize.sum()
        record["generalize-total"].append(g_total)
        record["generalize-count"].append(g_count)
    if i_count > 0:
        i_total = interfer.sum()
        record["interference-total"].append(i_total)
        record["interference-count"].append(i_count)
    return record

def test_noninterference(agent):
    def loss(val_net, rep_net, states, actions, next_states, rewards, terminals):#, true_val=None):
        # if true_val is None:
        #     true_val = val_net
        q = val_net(rep_net(states))[np.array(range(len(actions))), actions[:, 0]]
        # Constructing the target
        with torch.no_grad():
            q_next = val_net(rep_net(next_states))
            # q_next = true_val(rep_net(next_states))
            q_next = q_next.max(1)[0]
            terminals = torch_utils.tensor(terminals, agent.cfg.device)
            rewards = torch_utils.tensor(rewards, agent.cfg.device)
            target = agent.cfg.discount * q_next * (1 - terminals).float()
            target.add_(rewards.float())
            # print("{:.4f}, \t{:.4f}, \t{:.4f}".format(q_next.item(), q.item(), target.item(), rewards.item()))
        return torch.nn.functional.mse_loss(q, target)
        # return target - q

    # if agent.cfg.rep_config["load_params"]:
    #     path = os.path.join(agent.cfg.data_root, agent.cfg.rep_config["path"])
    #     agent.rep_net.load_state_dict(torch.load(path))
    #     path = path[:-7] + "val_net"
    #     true_val = copy.deepcopy(agent.val_net)
    #     true_val.load_state_dict(torch.load(path))

    state_all, next_s_all, _, action_all, reward_all, terminal_all = generate_distance_dataset(agent.cfg)
    state_all = agent.cfg.state_normalizer(state_all)
    next_s_all = agent.cfg.state_normalizer(next_s_all)
    action_all = action_all.reshape([-1, 1])
    rhos = []
    for i in range(10):
        sample_idx = np.random.choice(list(range(len(state_all))), size=100)
        state_batch = state_all[sample_idx]
        action_batch = action_all[sample_idx]
        next_s_batch = next_s_all[sample_idx]
        reward_batch = reward_all[sample_idx]
        terminal_batch = terminal_all[sample_idx]

        model = agent.cfg.val_fn() # use cfg to deal with the case when Laplace object does not have val_net
        rep_net = agent.rep_net
        # print(list(agent.val_net.parameters())[0])
        data_size = state_batch.shape[0]
        all_param = model.parameters()
        param_num = 0
        for p in all_param:
            param_num += np.product(p.size())
        grad_mat = np.zeros([data_size, param_num])
        for i in range(data_size):
            model.zero_grad()
            rep_net.zero_grad()
            l = loss(model, rep_net, state_batch[i:(i + 1)], action_batch[i:(i + 1)], next_s_batch[i:(i + 1)], reward_batch[i:(i + 1)], terminal_batch[i:(i + 1)])
            l.backward()
            grad_list = []
            for para in model.parameters():
                if para.grad is not None:
                    grad_list.append(para.grad.flatten().numpy())
            grad_mat[i] = np.concatenate(grad_list)

        ntk_mat = np.matmul(grad_mat, grad_mat.T)
        sample_norm = np.linalg.norm(grad_mat, axis=1).reshape((-1, 1))
        norm = np.matmul(sample_norm, sample_norm.T)
        ntk_mat = np.divide(ntk_mat, norm)
        ntk_mat = np.clip(ntk_mat * (-1), 0, np.inf)
        rho = 1 - (np.sum(ntk_mat) - np.trace(ntk_mat)) / (data_size * (data_size - 1))
        rhos.append(rho)
    # print(np.array(rhos).mean())
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../interference.txt"), "w") as f:
        f.write("Interference: {:.8f}".format(np.array(rhos).mean()))
def online_noninterference(agent, state_all, next_s_all, action_all, reward_all, terminal_all):
    def loss(val_net, rep_net, states, actions, next_states, rewards, terminals):
        q = val_net(rep_net(states))[np.array(range(len(actions))), actions[:, 0]]
        with torch.no_grad():
            q_next = val_net(rep_net(next_states))
            q_next = q_next.max(1)[0]
            terminals = torch_utils.tensor(terminals, agent.cfg.device)
            rewards = torch_utils.tensor(rewards, agent.cfg.device)
            target = agent.cfg.discount * q_next * (1 - terminals).float()
            target.add_(rewards.float())
        return torch.nn.functional.mse_loss(q, target)

    state_all = agent.cfg.state_normalizer(state_all)
    next_s_all = agent.cfg.state_normalizer(next_s_all)
    action_all = action_all.reshape([-1, 1])
    rhos = []
    for i in range(10):
        sample_idx = np.random.choice(list(range(len(state_all))), size=100)
        state_batch = state_all[sample_idx]
        action_batch = action_all[sample_idx]
        next_s_batch = next_s_all[sample_idx]
        reward_batch = reward_all[sample_idx]
        terminal_batch = terminal_all[sample_idx]
        data_size = state_batch.shape[0]
        all_param = agent.val_net.parameters()
        param_num = 0
        for p in all_param:
            param_num += np.product(p.size())
        grad_mat = np.zeros([data_size, param_num])
        for i in range(data_size):
            agent.val_net.zero_grad()
            agent.rep_net.zero_grad()
            l = loss(agent.val_net, agent.rep_net, state_batch[i:(i + 1)], action_batch[i:(i + 1)], next_s_batch[i:(i + 1)], reward_batch[i:(i + 1)], terminal_batch[i:(i + 1)])
            l.backward()
            grad_list = []
            for para in agent.val_net.parameters():
                if para.grad is not None:
                    grad_list.append(para.grad.flatten().numpy())
            grad_mat[i] = np.concatenate(grad_list)
        ntk_mat = np.matmul(grad_mat, grad_mat.T)
        sample_norm = np.linalg.norm(grad_mat, axis=1).reshape((-1, 1))
        norm = np.matmul(sample_norm, sample_norm.T)
        ntk_mat = np.divide(ntk_mat, norm)
        ntk_mat = np.clip(ntk_mat * (-1), 0, np.inf)
        rho = 1 - (np.sum(ntk_mat) - np.trace(ntk_mat)) / (data_size * (data_size - 1))
        rhos.append(rho)
    agent.val_net.zero_grad()
    agent.rep_net.zero_grad()
    log_str = 'total steps %d, total episodes %3d, ' \
              'Noninterference: %.8f/'
    agent.cfg.logger.info(log_str % (agent.total_steps, len(agent.episode_rewards), np.array(rhos).mean()))

def test_decorrelation(agent):
    state_all, next_s_all, _, action_all, reward_all, terminal_all = generate_distance_dataset(agent.cfg)
    with torch.no_grad():
        representations = agent.rep_net(agent.cfg.state_normalizer(state_all)).numpy()

        # remove dead features
        std_test = np.std(representations, axis=0)
        zeros = np.where(std_test == 0)[0]
        representations = np.delete(representations, zeros, 1)

        correlation_matrix = np.corrcoef(representations.transpose(1, 0))
        # correlation_matrix = np.nan_to_num(correlation_matrix, nan=0)

        # assert representations.shape[1] == 32
        dim = representations.shape[1]
        correlation_matrix[np.tril_indices(dim)] = 0.0
        correlation_matrix = np.abs(correlation_matrix)
        total_correlation = np.sum(np.abs(correlation_matrix))
        total_off_diag_upper = dim * (dim-1) / 2 # N(N-1)/2
        average_correlation = total_correlation / total_off_diag_upper
        decorr = 1 - average_correlation
    with open(os.path.join(agent.cfg.get_parameters_dir(), "../decorrelation.txt"), "w") as f:
        f.write("Decorrelation: {:.8f}".format(np.array(decorr).mean()))

def online_decorrelation(agent, state_all):
    with torch.no_grad():
        representations = agent.rep_net(agent.cfg.state_normalizer(state_all)).numpy()
        # remove dead features
        std_test = np.std(representations, axis=0)
        zeros = np.where(std_test == 0)[0]
        representations = np.delete(representations, zeros, 1)

        correlation_matrix = np.corrcoef(representations.transpose(1, 0))
        dim = representations.shape[1]
        correlation_matrix[np.tril_indices(dim)] = 0.0
        correlation_matrix = np.abs(correlation_matrix)
        total_correlation = np.sum(np.abs(correlation_matrix))
        total_off_diag_upper = dim * (dim-1) / 2 # N(N-1)/2
        average_correlation = total_correlation / total_off_diag_upper
        decorr = 1 - average_correlation
    log_str = 'total steps %d, total episodes %3d, ' \
              'Decorrelation: %.8f/'
    agent.cfg.logger.info(log_str % (agent.total_steps, len(agent.episode_rewards), decorr))#np.array(rhos).mean()))


def run_steps_onlineProperty(agent):
    t0 = time.time()
    agent.populate_returns()
    state_all, next_s_all, different_idx, action_all, reward_all, terminal_all = generate_distance_dataset(agent.cfg)
    while True:
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            t0 = time.time()
        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            agent.eval_episodes()
            if agent.cfg.visualize:
                agent.visualize()
            if agent.cfg.save_params:
                agent.save()
            if agent.cfg.evaluate_lipschitz:
                agent.log_lipschitz()
            if agent.cfg.evaluate_distance:
                online_distance(agent, state_all, next_s_all, different_idx)
            if agent.cfg.evaluate_orthogonality:
                online_orthogonality(agent, state_all)
            if agent.cfg.evaluate_noninterference:
                online_noninterference(agent, state_all, next_s_all, action_all, reward_all, terminal_all)
            if agent.cfg.evaluate_decorrelation:
                online_decorrelation(agent, state_all)
            t0 = time.time()
        if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
            agent.save()
            break
        agent.step()




def draw(state):
    import matplotlib.pyplot as plt
    frame = state.astype(np.uint8)
    figure, ax = plt.subplots()
    ax.imshow(frame)
    plt.axis('off')
    plt.show()
    plt.close()
