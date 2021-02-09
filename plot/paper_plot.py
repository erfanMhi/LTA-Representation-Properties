import os
import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from core.environment.gridworlds_goal import *
from core.environment.collectworlds import *

os.chdir("..")
print(os.getcwd())

def check_supervision():
    env = GridHardRGB(0)
    reset = True
    traj = []
    xys = []
    steps = 300000
    for i in range(steps):
        if reset:
            state = env.reset()
            xy = env.get_useful()
            reset = False
        action = np.random.randint(0, 4)
        next_state, reward, reset, _ = env.step([action])
        traj.append(next_state)
        xys.append(env.get_useful())
    srs = np.zeros([15, 15, 15, 15, 3])
    count = np.zeros((15, 15))
    sr = np.array(traj[steps-1])
    for i in range(steps-2, -1, -1):
        sr += traj[i + 1] * 0.9
        x, y = xys[i]
        srs[x, y] += sr
        count[x, y] += 1

    dist = np.zeros((15, 15))
    goal = srs[9, 9] / count[9, 9]
    for x in range(15):
        for y in range(15):
            if count[x, y] != 0:
                avg = srs[x, y] / count[x, y]
                dist[x, y] = np.linalg.norm(avg - goal)

    plt.figure()
    plt.imshow(dist)
    plt.show()

def plot_maze():
    fig, axs = plt.subplots(2, 3)
    order = [0, 4, 3, 1, 2, 5]
    for i in order:
        env = GridHardRGBGoal(i, 0)
        obs = env.reset()
        obs[env.goal_x, env.goal_y] = 255
        pos = order.index(i)
        axs[pos//3, pos%3].imshow(obs.astype(np.uint8))
    plt.savefig("img/env_maze.pdf", dpi=300, format='pdf', bbox_inches='tight')

def plot_picky_eater():
    env = CollectTwoColorRGB(0)
    obs = env.reset()
    plt.imshow(obs.astype(np.uint8))
    plt.savefig("img/env_picky_eater.pdf", dpi=300, format='pdf', bbox_inches='tight')

# def plot_env():
#     maze = GridHardRGBGoal(0, 0)
#     obs_m = maze.reset()
#     # obs_m[maze.goal_x, maze.goal_y] = 255
#     eater = CollectTwoColorRGB(0)
#     obs_e = eater.reset()
#     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#     ax1.imshow(obs_m.astype(np.uint8))
#     ax2.imshow(obs_e.astype(np.uint8))
#     plt.savefig("../../Pictures/environments.pdf", dpi=300, format='pdf', bbox_inches='tight')


def all_visualization(root_path, total_comb, name="visualization"):
    num_row = 2
    fig, axs = plt.subplots(num_row, total_comb//num_row+1)
    for i in range(total_comb):
        file = "{}/{}_param_setting/visualizations/{}.png".format(root_path, i, name)
        vis = plt.imread(file)
        axs[i%num_row, i//num_row].imshow(vis)
        axs[i%num_row, i//num_row].set_title(i)
        axs[i%num_row, i//num_row].axis('off')
    plt.show()

def check_sweep(results, env, exp, num_param, runs, file_name):

    xlabel = [x[1] for x in results]
    results = [x[0] for x in results]
    for res in results:
        avgs = []
        for i in range(num_param):
            avgs.append(confidence_interval(res.format(env, exp), file_name, runs, param_idx=i)[0])
        avgs = np.array(avgs)
        # print(res, avgs, avgs.argmin())
        print(res, avgs, avgs.argmax())


def confidence_interval(root_path, file_name, total_comb, start=0, param_idx=0, line=-1):
    all_res = []
    for i in range(start, total_comb):
        path = "{}/{}_run/{}_param_setting/{}.txt".format(root_path, i, param_idx, file_name)
        if os.path.isfile(path):
            with open(path, "r") as f:
                string = f.readlines()[line]
            res = float(string.split(" ")[-1])
            all_res.append(res)
        else:
            print(path, "not exist")

    all_res = np.array(all_res)
    mu = all_res.mean()
    std = all_res.std()
    # interval = [np.max([0, mu-std*2]), mu+std*2]
    interval = [mu-std*2, mu+std*2]
    # print("{}: {:.4f}, [{:.6f}, {:.6f}]".format(root_path.split("/")[-3:-1], mu, interval[0], interval[1]))
    print("{}: {:.3f}, [{:.3f}]".format(root_path.split("/")[-3:-1], mu, std))
    return mu, interval[0], interval[1], all_res


# def add_label(violin, label, labels):
#     color = violin["bodies"][0].get_facecolor().flatten()
#     labels.append((mpatches.Patch(color=color), label))
#     return labels

def violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name, normalize=False):
    if normalize:
        all_d_nomalize = []
        min = np.inf
        max = -1*np.inf
        for d in all_d:
            mn = d.min()
            mx = d.max()
            if mn < min:
                min = mn
            if mx > max:
                max = mx
        for d in all_d:
            d_nomalize = (d - min) / (max - min)
            all_d_nomalize.append(d_nomalize)
        all_d = all_d_nomalize

    violin_parts = ax1.violinplot(all_d, showmeans=False, showextrema=False)
    means = [np.mean(all_d[i]) for i in range(len(all_d))]#np.mean(np.array(all_d), axis=1)
    maxs = [np.max(all_d[i]) for i in range(len(all_d))]#np.max(np.array(all_d), axis=1)
    mins = [np.min(all_d[i]) for i in range(len(all_d))]#np.min(np.array(all_d), axis=1)

    for i in range(len(violin_parts['bodies'])):
        violin_parts['bodies'][i].set_facecolor(colors[i])
        ax1.scatter([i+1], means[i], marker='_', color=colors[i], s=150, zorder=10)
        ax1.scatter([i+1], maxs[i], marker='.', color=colors[i], s=50, zorder=10)
        ax1.scatter([i+1], mins[i], marker='.', color=colors[i], s=50, zorder=10)
        ax1.vlines([i+1], mins[i], maxs[i], colors[i], linestyle='-', lw=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.xticks(np.arange(1, len(all_d) + 1), xlabel, rotation=30, visible=True)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # plt.savefig("/home/han/Pictures/{}_{}_{}.pdf".format(exp, env, file_name), dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig("/home/han/Pictures/{}_{}_{}.png".format(exp, env, file_name), bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_10k_distance(exp, env, results, yaxis):
    colors = {
        "End2End (from scratch)": "black",
        "Pre-train DQN-Rep": "purple",
        "Aux XY": "green",
        "Aux XY Count": "green",
        "Aux InputDecoder": "lime",
        "Aux InputDecoderLong": "lime",
        "Aux NAS": "orange",
        "Successor AS": "wheat",
        "Successor AS large": "wheat",
        "Successor AS reward": "gold",
        "AuxControl": "cornflowerblue",
        "AuxControl 1G": "cornflowerblue",
        "AuxControl 5G": "royalblue",
        "RandomRep": "crimson",
        "NoRep": "slategray"
    }
    # xlabel = ['aux AE-rand', 'aux AE', 'pre-train DQN', 'aux sr-rand', 'aux sr', 'aux sr+reward', 'aux control1g', 'aux control5g',  'aux model', 'aux xy']
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    xlabel = [x[1] for x in results]
    results = [x[0] for x in results]
    colors = [colors[k] for k in xlabel]

    figsize = (1 * len(results), 5)
    # figsize = (5, 5)
    num_runs = 30

    # ===========================================================================================
    plt.figure(0, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    plt.title("State Similarity: larger the better")
    dist = []
    print("\nDistance-all")
    file_name = "10k_distance"
    plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    for res in results:
        if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
            dist.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
        else:
            dist.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    dist = np.array(dist)
    all_d = [dist[i][3] for i in range(len(dist))]
    violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)
    # ===========================================================================================
    plt.figure(1, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    plt.title("State Similarity: larger the better")
    dist = []
    print("\nDistance-green")
    file_name = "10k_distance_green"
    plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    for res in results:
        if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
            dist.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
        else:
            dist.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    dist = np.array(dist)
    all_d = [dist[i][3] for i in range(len(dist))]
    violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)

    # ===========================================================================================
    plt.figure(2, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    plt.title("State Similarity: larger the better")
    dist = []
    print("\nDistance-red")
    file_name = "10k_distance_red"
    plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    for res in results:
        if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
            dist.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
        else:
            dist.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    dist = np.array(dist)
    all_d = [dist[i][3] for i in range(len(dist))]
    violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)


def plot_interval(exp, env, results, yaxis, num_runs = 30):

    colors = {
        "Baseline (from scratch)": "black",
        "DQN": "black",
        "No auxiliary": "purple",
        "Expert-xy": "bisque",
        "Expert-(xy+color)": "goldenrod",
        "Expert-(xy+count)": "burlywood",

        "Input-decoder": "brown",

        "Aux Next-agent-state": "orange",
        "Next-agent-state": "orange",

        "Successor-feature": "green",
        "Successor-rep": "lime",

        "Pick-red control": "steelblue",
        "Single-goal control": "steelblue",
        "All-goals control": "dodgerblue",
        "control": "dodgerblue",

        "Random": "crimson",
        "NoRep": "slategray",

        "Laplace": "gray",
        "Laplace-rwd": "gray",
        "Laplace-xy": "gray",
        "Laplace-xy+color": "gray",
        "Laplace-xy+count": "gray",
        
        "Sparse": "darkcyan"
    }

    xlabel = [x[1] for x in results]
    results = [x[0] for x in results]
    colors = [colors[k] for k in xlabel]

    figsize = (1 * len(results), 5)
    # figsize = (5, 5)

    # ===========================================================================================
    plt.figure(0, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    den = []
    if exp == "denoising":
        plt.title("Denoising")
        print("\nDenoising")
        file_name = "denoise"
    elif exp in ["linear_probing", "linear_probing_color", "linear_probing_count"]:
        # plt.title("Linear Probing Accuracy")
        print("\nRetain XY")
        file_name = "linear_probing"
    else:
        raise NotImplementedError
    plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    for res in results:
        if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
            den.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
        else:
            den.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    den = np.array(den)
    all_d = [den[i][3] for i in range(len(den))]
    violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)

    # ===========================================================================================
    plt.figure(1, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    # plt.title("Dynamics Awareness")
    dist = []
    print("\nDistance")
    file_name = "distance"
    plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    for res in results:
        if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
            dist.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
        else:
            dist.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    dist = np.array(dist)
    all_d = [dist[i][3] for i in range(len(dist))]
    violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)

    #===========================================================================================
    plt.figure(2, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    # plt.title("Orthogonality")
    interf = []
    print("\nOrthogonality")
    file_name = "orthogonality"
    plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    for res in results:
        if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
            interf.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
        else:
            interf.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    interf = np.array(interf)
    all_d = [interf[i][3] for i in range(len(interf))]
    violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)

    #===========================================================================================
    plt.figure(3, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    # plt.title("Interference")
    interf = []
    print("\nInterference")
    file_name = "interference"
    plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    for res in results:
        if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
            interf.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
        else:
            interf.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    interf = np.array(interf)
    all_d = [interf[i][3] for i in range(len(interf))]
    violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)

    #===========================================================================================
    plt.figure(10, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    # plt.title("Interference")
    decorr = []
    print("\nDecorrelation")
    file_name = "decorrelation"
    plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    for res in results:
        if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
            decorr.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
        else:
            decorr.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    decorr = np.array(decorr)
    all_d = [decorr[i][3] for i in range(len(decorr))]
    violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)

    # plt.figure(9, figsize=figsize)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # fig.set_figwidth(figsize[0])
    # fig.tight_layout(pad=6)
    # plt.title("Expected interference")
    # print("Expected interference")
    # interf_only = []
    # for res in results:
    #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    #         interf_only.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1, line=-2))
    #     else:
    #         interf_only.append(confidence_interval(res.format(env, exp), file_name, num_runs, line=-2))
    # interf_only = np.array(interf_only)
    # all_d = [interf_only[i][3] for i in range(len(interf_only))]
    # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name+"_iExpect")
    #
    # plt.figure(10, figsize=figsize)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # fig.set_figwidth(figsize[0])
    # fig.tight_layout(pad=6)
    # plt.title("Expected generalization")
    # print("Expected generalization")
    # generalize_only = []
    # for res in results:
    #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    #         generalize_only.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1, line=-3))
    #     else:
    #         generalize_only.append(confidence_interval(res.format(env, exp), file_name, num_runs, line=-3))
    # generalize_only = np.array(generalize_only)
    # all_d = [generalize_only[i][3] for i in range(len(generalize_only))]
    # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name+"_gExpect")
    #
    # plt.figure(11, figsize=figsize)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # fig.set_figwidth(figsize[0])
    # fig.tight_layout(pad=6)
    # plt.title("Percentage of interference pairs")
    # print("Percentage of interference pairs")
    # interf_only = []
    # for res in results:
    #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    #         interf_only.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1, line=-4))
    #     else:
    #         interf_only.append(confidence_interval(res.format(env, exp), file_name, num_runs, line=-4))
    # interf_only = np.array(interf_only)
    # all_d = [interf_only[i][3] for i in range(len(interf_only))]
    # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name+"_iCount")
    #
    # plt.figure(12, figsize=figsize)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # fig.set_figwidth(figsize[0])
    # fig.tight_layout(pad=6)
    # plt.title("Percentage of generalization pairs")
    # print("Percentage of generalization pairs")
    # generalize_only = []
    # for res in results:
    #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    #         generalize_only.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1, line=-5))
    #     else:
    #         generalize_only.append(confidence_interval(res.format(env, exp), file_name, num_runs, line=-5))
    # generalize_only = np.array(generalize_only)
    # all_d = [generalize_only[i][3] for i in range(len(generalize_only))]
    # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name+"_gCount")

    # #===========================================================================================
    # plt.figure(4, figsize=figsize)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # fig.set_figwidth(figsize[0])
    # fig.tight_layout(pad=6)
    # plt.title("Robustness: smaller the better")
    # interf = []
    # print("\nRobustness")
    # file_name = "robustness"
    # plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    # for res in results:
    #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
    #     else:
    #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    # interf = np.array(interf)
    # all_d = [interf[i][3] for i in range(len(interf))]
    # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)
    #
    # #===========================================================================================
    # plt.figure(5, figsize=figsize)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # fig.set_figwidth(figsize[0])
    # fig.tight_layout(pad=6)
    # plt.title("Instance sparsity (h2)")
    # interf = []
    # print("\nInstance sparsity h2")
    # file_name = "sparsity_instance_h2"
    # plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    # for res in results:
    #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
    #     else:
    #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    # interf = np.array(interf)
    # all_d = [interf[i][3] for i in range(len(interf))]
    # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)
    #
    # # plt.figure(6, figsize=figsize)
    # # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # # fig.set_figwidth(figsize[0])
    # # fig.tight_layout(pad=6)
    # # plt.title("Lifetime sparsity (h2)")
    # # interf = []
    # # print("\nLifetime sparsity h2")
    # # file_name = "sparsity_lifetime_h2"
    # # plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    # # for res in results:
    # #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    # #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
    # #     else:
    # #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    # # interf = np.array(interf)
    # # all_d = [interf[i][3] for i in range(len(interf))]
    # # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)
    #
    # #===========================================================================================
    # plt.figure(7, figsize=figsize)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # fig.set_figwidth(figsize[0])
    # fig.tight_layout(pad=6)
    # plt.title("Instance sparsity (dense)")
    # interf = []
    # print("\nInstance sparsity dense")
    # file_name = "sparsity_instance_dense"
    # plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    # for res in results:
    #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
    #     else:
    #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    # interf = np.array(interf)
    # all_d = [interf[i][3] for i in range(len(interf))]
    # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)
    #
    # # plt.figure(8, figsize=figsize)
    # # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # # fig.set_figwidth(figsize[0])
    # # fig.tight_layout(pad=6)
    # # plt.title("Lifetime sparsity (dense)")
    # # interf = []
    # # print("\nLifetime sparsity dense")
    # # file_name = "sparsity_lifetime_dense"
    # # plt.ylim(yaxis[file_name][0], yaxis[file_name][1])
    # # for res in results:
    # #     if res == "data/output/paper_results/{}/{}/dqn_aux/successor_as/best_reward/":
    # #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs, start=1))
    # #     else:
    # #         interf.append(confidence_interval(res.format(env, exp), file_name, num_runs))
    # # interf = np.array(interf)
    # # all_d = [interf[i][3] for i in range(len(interf))]
    # # violin_plot(ax1, colors, all_d, xlabel, exp, env, file_name)


# all_visualization("data/output/denoising/gridrgb-ns/laplace/sweep_1conv/0_run", 39, name="visualization")

# plot_interval("dynamic_noninterference", "gridhard",
#               results=[
#                     ["data/output/paper_results/{}/{}/random/default/", "Random representation"],
#                     ["data/output/paper_results/{}/{}/dqn/best/", "No auxiliary"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/input_decoder/best_longer/", "Input-decoder prediction"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/xy/best/", "Expert-xy prediction"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/aux_control/best_1g/", "Single-goal control"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/aux_control/best_5g/", "All-goals control"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/nas_v2_delta/best/", "Next-agent-state prediction"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/successor_as/best_v2/", "Successor-feature prediction"],
#               ],
#               yaxis={
#                   "interference_test": [0.7, 0.95],
#               }, num_runs = 60
#               )
# plot_interval("dynamic_noninterference", "collect_two",
#               results=[
#                     ["data/output/paper_results/{}/{}/random/default/", "Random representation"],
#                     ["data/output/paper_results/{}/{}/dqn/best/", "No auxiliary"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/input_decoder/best/", "Input-decoder prediction"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/xy/best_xy/", "Expert-xy prediction"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/xy/best/", "Expert-(xy+color) prediction"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/xy/best_count/", "Expert-(xy+count) prediction"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/aux_control/best/", "Pick-red control"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/nas_v2_delta/best/", "Next-agent-state prediction"],
#                     ["data/output/paper_results/{}/{}/dqn_aux/successor_as/best_v2/", "Successor-feature prediction"],
#               ],
#               yaxis={
#                   "interference_test": [0.8, 1.0],
#               }, num_runs = 30
#               )

# plot_interval("linear_probing", "gridhard",
#               results=[
#                   ["data/output/paper_results/{}/{}/random/default/", "Random representation"],
#                   ["data/output/paper_results/{}/{}/dqn/best/", "No auxiliary"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/input_decoder/best_longer/", "Input-decoder prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best/", "Expert-xy prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/aux_control/best_1g/", "Single-goal control"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/aux_control/best_5g/", "All-goals control"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/nas_v2_delta/best/", "Next-agent-state prediction"],
#                   ["data/output/tests/{}/{}/dqn_aux/successor_as/best_v3/", "Successor-feature prediction"],
#                   ["data/output/tests/{}/{}/dqn_aux/successor_as/vanilla/", "Successor-rep prediction"],
#               ],
#               yaxis={
#                   "linear_probing": [0.6, 1.0],
#                   "distance": [0.2, 0.9],
#                   "distance_random": [0.2, 0.9],
#                   "distance_v2": [0.2, 0.9],
#                   "orthogonality": [0, 0.8],
#                   "interference": [-0.03, 1.03],
#                   "decorrelation": [0.4, 0.95],
#                   "robustness": [-0.03, 1.03],
#                   "sparsity_instance_h2": [0, 1],
#                   "sparsity_lifetime_h2": [0, 1],
#                   "sparsity_instance_dense": [0, 1],
#                   "sparsity_lifetime_dense": [0, 1]
#               }, num_runs = 60
#               )
# plot_interval("linear_probing", "collect_two",
#               results = [
#                   ["data/output/paper_results/{}/{}/random/default/", "Random representation"],
#                   ["data/output/paper_results/{}/{}/dqn/best/", "No auxiliary"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/input_decoder/best/", "Input-decoder prediction"],
#                   # ["data/output/paper_results/{}/{}/dqn_aux/xy/best_xy/", "Expert-xy prediction"],
#                   ["data/output/tests/{}/{}/dqn_aux/xy/best_xy/", "Expert-xy prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best/", "Expert-(xy+color) prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best_count/", "Expert-(xy+count) prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/aux_control/best/", "Pick-red control"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/nas_v2_delta/best/", "Next-agent-state prediction"],
#                   # ["data/output/paper_results/{}/{}/dqn_aux/successor_as/best_v2/", "Successor-feature prediction"],
#                   ["data/output/tests/{}/{}/dqn_aux/successor_as/best_v2/", "Successor-feature prediction"],
#                   ["data/output/tests/{}/{}/dqn_aux/successor_as/vanilla/", "Successor-rep prediction"],
#               ],
#               yaxis= {
#                   "linear_probing": [0.6, 1.0],
#                   "distance": [0.6, 1.0],
#                   "distance_random": [0, 1.0],
#                   "distance_sameEp": [0.6, 1.0],
#                   "distance_v2": [0.6, 1.0],
#                   "orthogonality": [0, 0.6],
#                   "interference": [-0.03, 1.03],
#                   "decorrelation": [0.3, 0.9],
#                   "robustness": [-0.03, 1.03],
#                   "sparsity_instance_h2": [0, 1],
#                   "sparsity_lifetime_h2": [0, 1],
#                   "sparsity_instance_dense": [0, 1],
#                   "sparsity_lifetime_dense": [0, 1]
#               })

# plot_interval("linear_probing_count", "collect_two",
#               results = [
#                   ["data/output/paper_results/{}/{}/random/default_together/", "Random representation"],
#                   ["data/output/paper_results/{}/{}/dqn/best_together/", "No auxiliary"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/input_decoder/best_together/", "Input-decoder prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best_xy_together/", "Expert-xy prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best_together/", "Expert-(xy+color) prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best_count_together/", "Expert-(xy+count) prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/aux_control/best_together/", "Pick-red control"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/nas_v2_delta/best_together/", "Next-agent-state prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/successor_as/best_v2_together/", "Successor-feature prediction"],
#                   ["data/output/tests/{}/{}/dqn_aux/successor_as/vanilla/", "Successor-rep prediction"],
#               ],
#               yaxis= {
#                   "linear_probing": [0.5, 1.01],
#                   "distance": [-0.03, 1.03],
#                   "orthogonality": [-0.03, 1.03],
#                   "interference": [-0.03, 1.03],
#                   "robustness": [-0.03, 1.03],
#                   "sparsity_instance_h2": [0, 1],
#                   "sparsity_lifetime_h2": [0, 1],
#                   "sparsity_instance_dense": [0, 1],
#                   "sparsity_lifetime_dense": [0, 1]
#               })
#
# plot_interval("linear_probing_color", "collect_two",
#               results = [
#                   ["data/output/paper_results/{}/{}/random/default/", "Random representation"],
#                   ["data/output/paper_results/{}/{}/dqn/best/", "No auxiliary"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/input_decoder/best/", "Input-decoder prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best_xy/", "Expert-xy prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best/", "Expert-(xy+color) prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/xy/best_count/", "Expert-(xy+count) prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/aux_control/best/", "Pick-red control"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/nas_v2_delta/best/", "Next-agent-state prediction"],
#                   ["data/output/paper_results/{}/{}/dqn_aux/successor_as/best_v2/", "Successor-feature prediction"],
#                   ["data/output/tests/{}/{}/dqn_aux/successor_as/vanilla/", "Successor-rep prediction"],
#               ],
#               yaxis= {
#                   "linear_probing": [0.45, 1.01],
#                   "distance": [-0.03, 1.03],
#                   "orthogonality": [-0.03, 1.03],
#                   "interference": [-0.03, 1.03],
#                   "robustness": [-0.03, 1.03],
#                   "sparsity_instance_h2": [0, 1],
#                   "sparsity_lifetime_h2": [0, 1],
#                   "sparsity_instance_dense": [0, 1],
#                   "sparsity_lifetime_dense": [0, 1]
#               })

# check_supervision()

# results = [
#     ["data/output/tests/{}/{}/dqn_aux/successor_as_orth/sweep", "Aux"]
# ]
# check_sweep(results, "representations", "collect_two", "linear_probing")

# plot_maze()
# plot_picky_eater()
# plot_env()


#============================== after submission

# results = [
#     ["data/output/linear_vf/{}/{}/dqn/sweep", "DQN"],
#     ["data/output/linear_vf/{}/{}/dqn_aux/aux_control/sweep_1g", "DQN-control1g"],
#     ["data/output/linear_vf/{}/{}/dqn_aux/aux_control/sweep_5g", "DQN-control5g"],
#     ["data/output/linear_vf/{}/{}/dqn_aux/info/sweep", "DQN-xy"],
#     ["data/output/linear_vf/{}/{}/dqn_aux/input_decoder/sweep", "DQN-id"],
#     ["data/output/linear_vf/{}/{}/dqn_aux/nas_v2_delta/sweep", "DQN-nas"],
#     ["data/output/linear_vf/{}/{}/dqn_aux/successor_as/sweep", "DQN-sf"],
#     ["data/output/linear_vf/{}/{}/dqn_sparse/sweep", "DQN-sparse"],
#     # ["data/output/linear_vf/{}/{}/laplace/sweep", "Laplace"]
#     ["data/output/linear_vf/{}/{}/random/sweep", "Random"],
#     ["data/output/linear_vf/{}/{}/input/sweep", "Input"],
#
#     ["data/output/linear_vf/{}/{}/dqn_sparse_highdim/sparse0.1/sweep", "DQN-sparse"],
#     ["data/output/linear_vf/{}/{}/dqn_sparse_highdim/sparse0.2/sweep", "DQN-sparse"],
#     ["data/output/linear_vf/{}/{}/dqn_sparse_highdim/sparse0.4/sweep", "DQN-sparse"],
# ]
# check_sweep(results, "gridhard", "property", 4, 5, "linear_probing_xy")

results = [
    ["data/output/linear_vf/{}/{}/dqn/sweep", "DQN"],
    ["data/output/linear_vf/{}/{}/dqn_aux/aux_control/sweep", "DQN-control1g"],
    ["data/output/linear_vf/{}/{}/dqn_aux/info/sweep_xy", "DQN-xy"],
    ["data/output/linear_vf/{}/{}/dqn_aux/info/sweep_xy+color", "DQN-xy+color"],
    ["data/output/linear_vf/{}/{}/dqn_aux/info/sweep_xy+count", "DQN-xy+ccount"],
    ["data/output/linear_vf/{}/{}/dqn_aux/input_decoder/sweep", "DQN-id"],
    ["data/output/linear_vf/{}/{}/dqn_aux/nas_v2_delta/sweep", "DQN-nas"],
    ["data/output/linear_vf/{}/{}/dqn_aux/successor_as/sweep", "DQN-sf"],
    ["data/output/linear_vf/{}/{}/dqn_sparse_highdim/sparse0.1/sweep", "DQN-sparse"],
    ["data/output/linear_vf/{}/{}/dqn_sparse_highdim/sparse0.2/sweep", "DQN-sparse"],
    ["data/output/linear_vf/{}/{}/dqn_sparse_highdim/sparse0.4/sweep", "DQN-sparse"],

    ["data/output/linear_vf/{}/{}/random/sweep", "Random"],
    ["data/output/linear_vf/{}/{}/input/sweep", "Input"]
]
check_sweep(results, "collect_two", "property", 4, 5, "linear_probing_xy")
check_sweep(results, "collect_two", "property", 4, 5, "linear_probing_color")
check_sweep(results, "collect_two", "property", 4, 5, "linear_probing_count")