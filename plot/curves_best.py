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


def learning_curve(all_paths_dict, title):
    labels = [i["label"] for i in all_paths_dict]
    # control = load_return(all_paths_dict)
    control = load_info(all_paths_dict, 0, "return")
    plt.figure()
    for label in labels:
        returns = arrange_order(control[label])
        draw_curve(returns, plt, label, violin_colors[label])
    # plt.title(title)
    plt.legend()
    # plt.xlim(0, 30)
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

def mountain_car():
    print("\nRep learning")
    learning_curve(mc_learn, "mountain car learning")

def simple_maze():
    print("\nRep learning")
    # learning_curve(gh_learn, "maze learning")
    # learning_curve(gh_online, "maze online measure")

    # # print("\nTransfer")
    # learning_curve(gh_etaStudy_diff_fix, "maze eta different (fix)")
    # learning_curve(gh_etaStudy_diff_tune, "maze eta different (fine tune)")

    # learning_curve(gh_same, "maze same")
    # learning_curve(gh_similar, "maze similar")
    # learning_curve(gh_diff, "maze different (fix)")
    learning_curve(gh_diff_tune, "maze different (fine tune)")

if __name__ == '__main__':
    # mountain_car()
    simple_maze()
    # picky_eater()
