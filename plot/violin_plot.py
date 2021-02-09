import os
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt

from plot.plot_utils import *
from plot.plot_dicts import *

os.chdir("..")
print("Change dir to", os.getcwd())


def plot_interval_complexity_reduction(path_dict_all, env):
    keyword = "Lipschitz:"

    figsize = (5 * len(path_dict_all), 5)

    plt.figure(0, figsize=figsize)
    fig, axs = plt.subplots(nrows=1, ncols=len(path_dict_all))
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)

    for j in range(len(path_dict_all)):
        path_dict = path_dict_all[j]
        xlabel = [x["label"] for x in path_dict]
        paths = [x["control"] for x in path_dict]
        colors = [violin_colors[k] for k in xlabel]
        den = []

        for path in paths:
            den.append(confidence_interval(path, 0, "log", keyword))
        # den = np.array(den)

        all_d = [den[i][3] for i in range(len(den))]
        flatten = lambda t: [item for sublist in t for item in sublist]

        max_d = np.max(np.array(flatten(all_d)))#, dtype=object))
        for i in range(len(all_d)):
            all_d[i] = 1 - all_d[i] / max_d
        violin_plot(axs[j], colors, all_d)
        axs[j].set_xticks(list(range(1, len(xlabel)+1)))
        axs[j].set_xticklabels(xlabel, rotation=90)
    plt.title(keyword)
    plt.savefig("plot/img/{}_complexityReduction.png".format(env), bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_interval(path_dict, file_name, yaxis, env):

    xlabel = [x["label"] for x in path_dict]
    paths = [x["property"] for x in path_dict]
    colors = [violin_colors[k] for k in xlabel]

    figsize = (1 * len(paths), 5)

    plt.figure(0, figsize=figsize)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)
    den = []

    plt.ylim(yaxis[0], yaxis[1])
    for path in paths:
        den.append(confidence_interval(path, 0, file_name, target_keywords[file_name]))
    # den = np.array(den)
    all_d = [den[i][3] for i in range(len(den))]
    violin_plot(ax1, colors, all_d)
    plt.xticks(np.arange(1, len(all_d) + 1), xlabel, rotation=30, visible=True)
    plt.title(file_name.strip(".txt"))
    plt.savefig("plot/img/{}_{}.png".format(env, file_name.strip(".txt")), bbox_inches='tight')
    plt.close()
    plt.clf()

def maze_violin():
    plot_interval(maze_same, "decorrelation.txt", [0.4,1], "maze")
    plot_interval(maze_same, "distance.txt", [0,1], "maze")
    plot_interval(maze_same, "interference.txt", [0.95,1], "maze")
    plot_interval(maze_same, "linear_probing_xy.txt", [0.7,1], "maze")
    plot_interval(maze_same, "orthogonality.txt", [0,0.8], "maze")
    plot_interval(maze_same, "sparsity_instance.txt", [0,1.0], "maze")

    plot_interval_complexity_reduction([maze_same, maze_similar, maze_different], "maze")

def collect_violin():
    # plot_interval(eater_same, "decorrelation.txt", [0.4,1], "collect")
    # plot_interval(eater_same, "distance.txt", [0.2,1], "collect")
    # plot_interval(eater_same, "interference.txt", [0.94,0.98], "collect")
    # plot_interval(eater_same, "linear_probing_xy.txt", [0.65,1], "collect")
    # plot_interval(eater_same, "linear_probing_color.txt", [0.4,1], "collect")
    # plot_interval(eater_same, "linear_probing_count.txt", [0.4,1], "collect")
    # plot_interval(eater_same, "orthogonality.txt", [0,0.4], "collect")
    # plot_interval(eater_same, "sparsity_instance.txt", [0,0.8], "collect")

    plot_interval_complexity_reduction([eater_same, eater_different_fix, eater_different_tune], "collect")

def mazexy_violin():
    plot_interval(mazexy_same, "sparsity_instance.txt", [0, 1], "maze-xy")

if __name__ == '__main__':
    # maze_violin()
    collect_violin()
    # mazexy_violin()