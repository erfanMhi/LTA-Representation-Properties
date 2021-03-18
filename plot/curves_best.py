import os
import sys
import numpy as np
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

from plot.plot_utils import *
from plot.plot_paths import *

os.chdir("..")
print("Change dir to", os.getcwd())

# def load_return(paths):
#     all_rt = {}
#     for i in paths:
#         path = i["control"]
#         # print("Loading returns from", path)
#         returns = extract_from_setting(path, 0)
#         all_rt[i["label"]] = returns
#     return all_rt


def learning_curve(all_paths_dict, title, targets=None, xlim=None):
    if targets is not None:
        temp = []
        for item in all_paths_dict:
            if item["label"] in targets:
                temp.append(item)
        all_paths_dict = temp

    labels = [i["label"] for i in all_paths_dict]
    # control = load_return(all_paths_dict)
    control = load_info(all_paths_dict, 0, "return")
    print(control.keys())
    plt.figure()
    for label in labels:
        print(label)
        returns = arrange_order(control[label])
        draw_curve(returns, plt, label, violin_colors[label])
    # plt.title(title)
    # plt.legend()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()

def learning_curve_mean(all_paths_dict, title, targets=None, xlim=None):
    if targets is not None:
        temp = []
        for item in all_paths_dict:
            if item["label"] in targets:
                temp.append(item)
        all_paths_dict = temp

    labels = [i["label"] for i in all_paths_dict]
    # control = load_return(all_paths_dict)
    control = load_info(all_paths_dict, 0, "return")
    plt.figure()
    total = 0
    for label in labels:
        returns = arrange_order(control[label])
        draw_curve(returns, plt, label, violin_colors[label], alpha=0.3)
        total = returns if type(total) == int else total + returns
    draw_curve(total/len(labels), plt, "Avg", "black")

    # plt.title(title)
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()

def mountain_car():
    print("\nRep learning")
    learning_curve(mc_learn, "mountain car learning")

def simple_maze():
    print("\nRep learning")
    targets = ["LTA eta=0.2", "LTA+Control1g", "LTA+Control5g",
               "LTA+XY", "LTA+Decoder", "LTA+NAS", "LTA+Reward", "LTA+SF",
               "ReLU", "Random", "Input"]

    # learning_curve(gh_online, "maze online measure", targets, xlim=[0, 30])
    #
    # learning_curve(gh_same_early, "maze same early", targets, xlim=[0, 10])
    # learning_curve(gh_similar_early, "maze similar early", targets, xlim=[0, 10])
    # learning_curve(gh_diff_early, "maze different (fix) early", targets, xlim=[0, 10])
    # learning_curve(gh_diff_tune_early, "maze different (fine tune) early", targets, xlim=[0, 10])
    #
    # learning_curve(gh_same_last, "maze same last", targets, xlim=[0, 10])
    # learning_curve(gh_similar_last, "maze similar last", targets, xlim=[0, 10])
    # learning_curve(gh_diff_last, "maze different (fix) last", targets, xlim=[0, 10])
    # learning_curve(gh_diff_tune_last, "maze different (fine tune) last", targets, xlim=[0, 10])
    #
    # learning_curve_mean(gh_online, "maze online measure", targets, xlim=[0, 30])

def picky_eater():
    # learning_curve(crgb_online_dqn, "maze online dqn")

    # learning_curve(crgb_online_dqn_lta, "maze online dqn with lta")
    learning_curve(crgb_online_dt_fr, "maze transfer different task fix rep")
    learning_curve(crgb_online_st_fr, "maze transfer same task fixed rep")
    learning_curve(crgb_online_dt_ft, "maze transfer different task fine tune")


if __name__ == '__main__':
    # mountain_car()
    #simple_maze()
    picky_eater()
