import os
import sys
import numpy as np
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

from plot.plot_utils import *
from plot.plot_paths import *

os.chdir("..")
print("Change dir to", os.getcwd())

# def arrange_order(dict1):
#     lst = []
#     min_l = np.inf
#     for i in sorted(dict1):
#         v1 = dict1[i]
#         lst.append(v1)
#         l = len(v1)
#         min_l = l if l < min_l else min_l
#     for i in range(len(lst)):
#         lst[i] = lst[i][:min_l]
#     return np.array(lst)


def learning_curve(all_paths_dict, title, total_param=None, start_param=0):
    labels = [i["label"] for i in all_paths_dict]
    control = load_return(all_paths_dict, total_param)#, start_param)
    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(30, 4))
    for idx, label in enumerate(labels):
        print("------", label, "------")
        all_params = control[label]
        for param, returns in all_params.items():
            returns = arrange_order(returns)
            # draw_curve(returns, axs[idx], param, cmap(float(param)/len(list(all_params.keys()))))
            draw_curve(returns, axs[idx], param, cmap(param, len(list(all_params.keys()))))
        axs[idx].set_title(label)
        axs[idx].legend()

    fig.suptitle(title)
    # plt.xlim(0, 30)
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    # plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    # plt.clf()

def mountain_car():
    learning_curve(mc_learn_sweep, "mountain car learning sweep")

def simple_maze():
    # print("\nRep learning")
    # learning_curve(gh_learn_sweep, "maze learning sweep")
    learning_curve(gh_online_sweep, "maze online property sweep")

    # # print("\nControl")
    # learning_curve(gh_same_sweep, "maze same sweep")
    # learning_curve(gh_similar_sweep, "maze similar sweep")
    # learning_curve(gh_diff_sweep, "maze different (fix) sweep")
    # learning_curve(gh_diff_tune_sweep, "maze different (fine tune) sweep")

    # learning_curve(gh_etaStudy_diff_fix_sweep, "maze different (fix) eta study")
    # learning_curve(gh_etaStudy_diff_tune_sweep, "maze different (fine tune) eta study")



if __name__ == '__main__':
    # mountain_car()
    simple_maze()
    # picky_eater()
