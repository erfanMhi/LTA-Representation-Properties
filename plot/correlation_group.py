import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from plot.correlation import *


def corr_changes_one_property(rep_lst, property, perc_lst, relationship):
    corrs = []
    for perc in perc_lst:
        c = calculation(rep_lst, property, perc=perc, relationship=relationship)
        corrs.append(c)
    return corrs

def corr_changes_complexity_reduction(rep_lst, perc_lst, relationship):
    corrs = []
    for perc in perc_lst:
        c = calculation_complexity_reduction(rep_lst, perc=perc, relationship=relationship)
        corrs.append(c)
    return corrs

def correlation_block(group_all, relationship):
    # perc_lst = [
    #     [0, 0.5],
    #     [0.5, 1],
    #     [0, 1],
    # ]
    # perc_lst = [
    #     [0, 0.2],
    #     [0.2, 0.4],
    #     [0.4, 0.6],
    #     [0.6, 0.8],
    #     [0.8, 1],
    #     [0,1]
    # ]
    perc_lst = [
        [0, 1]
    ]

    deco_changes = corr_changes_one_property(group_all, "decorrelation.txt", perc_lst, relationship)
    dist_changes = corr_changes_one_property(group_all, "distance.txt", perc_lst, relationship)
    interf_changes = corr_changes_one_property(group_all, "interference.txt", perc_lst, relationship)
    lp_changes = corr_changes_one_property(group_all, "linear_probing_xy.txt", perc_lst, relationship)
    ortho_changes = corr_changes_one_property(group_all, "orthogonality.txt", perc_lst, relationship)
    sparse_changes = corr_changes_one_property(group_all, "sparsity_instance.txt", perc_lst, relationship)
    cr_changes = corr_changes_complexity_reduction(group_all, perc_lst, relationship)

    labels = ["decorrelation", "dynamic-aware", "non-interference", "info-retain", "orthogonality", "sparsity", "complexity_reduction"]
    data = [deco_changes, dist_changes, interf_changes, lp_changes, ortho_changes, sparse_changes, cr_changes]
    x = np.arange(len(labels))
    width = 0.5  # the width of the bars

    bar_w = width / len(perc_lst)
    fig, ax = plt.subplots()
    for i in range(len(data[0])):
        temp = x if bar_w==width else x - width/2+bar_w*i
        ax.bar(temp, [d[i] for d in data], bar_w, label=perc_lst[i])
        print("Bar {}, data {}".format(i, [d[i] for d in data]))

    ax.set_ylabel('Correlation')
    ax.set_title('Correlation of performance blocks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="center")
    ax.plot([x[0]-width]+list(x)+[x[-1]+width], [0]*(len(x)+2), color="black")

    # ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    group_all = [
        maze_same,
        maze_similar,
        maze_different_fix,
        # maze_different_tune,
        eater_same,
        eater_different_fix,
        # eater_different_tune,
    ]


    correlation_block([maze_same, eater_same], None)
    # correlation_block([maze_similar], None)
    # correlation_block([maze_different_fix, eater_different_fix], None)
    # correlation_block([maze_different_tune, eater_different_tune], None)

    # correlation_block(group_all, None)
    # # correlation_block(group_all, "log")
    # # correlation_block(group_all, "sqrt")
