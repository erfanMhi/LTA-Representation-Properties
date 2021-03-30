import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
sys.path.insert(0, '..')

from plot.plot_utils import *
from plot.plot_paths import *

# from plot.curves_property import learning_curve_mean
os.chdir("..")
print("Change dir to", os.getcwd())


def load_property(group, property_key, targets, early_stopped):
    print("\n"+property_key)
    ordered = []
    for target in targets:
        for i in group:
            if i["label"] == target:
                ordered.append(i)
    group = ordered

    # print(all_groups, "\n\n", all_group_dict, "\n")
    reverse = True if property_key in ["lipschitz", "interf"] else False  # normalize interference and lipschitz, for non-interference and complexity reduction measure
    normalize = True if property_key in ["return"] else False
    model_saving = load_info(group, 0, "model", path_key="online_measure") if early_stopped else None
    properties = load_online_property(group, property_key, reverse=reverse, normalize=normalize, cut_at_step=model_saving)
    labels = [i["label"] for i in group]
    assert labels == targets


    # all reps:
    all_property = []
    all_color = []
    all_marker = []
    indexs = [0]
    for idx in range(len(labels)):
        rep = labels[idx]
        color = violin_colors[rep]
        marker = marker_styles[rep]
        prop = arrange_list(properties[rep])
        all_property.append(prop)
        all_color.append(color)
        all_marker.append(marker)

    all_property = np.array(all_property)
    return all_property

def arrange_list(dict1):
    l1 = []
    for i in dict1.keys():
        v1 = dict1[i]
        if np.isnan(v1):
            print("run {} is nan".format(i))
            v1 = 0
        l1.append(v1)
    return l1


def simple_maze_check_aux(targets, title, early_stopped):

    print("\nDifferent task - fix")
    diff_lip = load_property(gh_diff_early, property_key="lipschitz", targets=targets, early_stopped=early_stopped)
    diff_dist = load_property(gh_diff_early, property_key="distance", targets=targets, early_stopped=early_stopped)
    diff_ortho = load_property(gh_diff_early, property_key="ortho", targets=targets, early_stopped=early_stopped)
    diff_interf = load_property(gh_diff_early, property_key="interf", targets=targets, early_stopped=early_stopped)
    diff_diversity = load_property(gh_diff_early, property_key="diversity", targets=targets, early_stopped=early_stopped)
    diff_spars = load_property(gh_diff_early, property_key="sparsity", targets=targets, early_stopped=early_stopped)
    diff_return = load_property(gh_diff_early, property_key="return", targets=targets, early_stopped=early_stopped)

    # labels = ["complexity reduction", "dynamics awareness", "orthogonality", "noninterference", "diversity", "sparsity", "transfer performance"]
    # diff_aux = {}
    # for idx, t in enumerate(targets):
    #     diff_aux[t] = [diff_lip[idx], diff_dist[idx], diff_ortho[idx], diff_interf[idx], diff_diversity[idx], diff_spars[idx], diff_return[idx]]
    #
    # x = np.arange(len(labels))  # the label locations
    # width = 0.1  # the width of the bars
    # fig, ax = plt.subplots()
    # for i,t in enumerate(targets):
    #     # ax.bar(x - 0.45 + width * i, diff_aux[t], width, label=t, color=violin_colors[t])
    #     positions = x - 0.35 + width * i
    #     violin_plot(ax, violin_colors[t], diff_aux[t], positions, width)
    #     # box_plot(ax, violin_colors[t], diff_aux[t], positions, width)
    #     ax.plot([], c=violin_colors[t], label=t)
    #
    # # ax.plot([x[0] - 0.45, x[-1] + 0.45], [0, 0], "--", color="grey")
    # ax.vlines((x-0.5)[1:], 0, 1, transform=ax.get_xaxis_transform(), ls=":", colors="grey", alpha=1, linewidth=1)
    #
    # fontP = FontProperties()
    # fontP.set_size('xx-small')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
    # ax.set_ylabel('Measure')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=30)
    # # plt.show()
    # plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    # plt.close()
    # plt.clf()

    labels = ["complexity reduction", "dynamics awareness", "orthogonality", "noninterference", "diversity", "sparsity", "transfer performance"]
    diff_prop = [diff_lip, diff_dist, diff_ortho, diff_interf, diff_diversity, diff_spars, diff_return]

    x = np.arange(len(targets))  # the reps
    width = 0.5  # the width of the bars

    col = 2
    row = 4
    fig, axs = plt.subplots(row, col, figsize=(5, 5))
    v_colors = [violin_colors[t] for t in targets]
    for i,l in enumerate(labels):
        ax = axs[i//col, i%col]
        for j in range(len(diff_prop[i])):
            violin_plot(ax, v_colors[j], [diff_prop[i][j]], [x[j]], width)
        ax.set_title(l, size=10)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xticks([])
        ax.set_yticks([diff_prop[i].min(), diff_prop[i].max()])
    for i,t in enumerate(targets):
        plt.plot([], c=violin_colors[t], label=t)
    if len(labels) <= (col * row):
        for i in range(len(labels), col*row):
            ax = axs[i//col, i%col]
            ax.axis('off')

    fig.tight_layout(pad=1.0)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
    plt.legend(bbox_to_anchor=(0.2, 0.5), loc='center', prop=fontP)
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

def simple_maze():
    targets_relu = [
        "ReLU",
        "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        ]
    targets_fta = [
        "FTA eta=0.2",# "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
        ]
    targets_eta = [
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8", "ReLU"
        ]
    simple_maze_check_aux(targets_relu, "maze_aux_relu", True)
    simple_maze_check_aux(targets_fta, "maze_aux_fta", True)
    violin_colors["ReLU"] = c_default[7]
    simple_maze_check_aux(targets_eta, "maze_eta", True)

if __name__ == '__main__':
    simple_maze()