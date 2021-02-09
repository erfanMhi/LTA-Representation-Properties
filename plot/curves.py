import os
import sys
import numpy as np
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

from plot.plot_utils import *
from plot.plot_dicts import *

os.chdir("..")
print("Change dir to", os.getcwd())

def arrange_order(dict1):
    lst = []
    min_l = np.inf
    for i in sorted(dict1):
        v1 = dict1[i]
        lst.append(v1)
        l = len(v1)
        min_l = l if l < min_l else min_l
    for i in range(len(lst)):
        lst[i] = lst[i][:min_l]
    return np.array(lst)

def load_return(paths):
    all_rt = {}
    for i in paths:
        path = i["control"]
        # print("Loading returns from", path)
        returns = extract_return_setting(path, 0)
        all_rt[i["label"]] = returns
    return all_rt

def learning_curve(all_paths_dict, title):
    labels = [i["label"] for i in all_paths_dict]
    control = load_return(all_paths_dict)
    plt.figure()
    for label in labels:
        returns = arrange_order(control[label])
        draw_curve(returns, plt, label, violin_colors[label])
    # plt.title(title)
    # plt.legend()
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()


def simple_maze():
    print("\nRep learning")
    learning_curve(maze_learn, "maze learning")

    print("\nSame task")
    learning_curve(maze_same, "maze same")

    print("\nSimilar task")
    learning_curve(maze_similar, "maze similar")

    print("\nDifferent task - fix")
    learning_curve(maze_different_fix, "maze different (fix)")

    print("\nDifferent task - tune")
    learning_curve(maze_different_tune, "maze different (fine tune)")

def picky_eater():
    print("\nRep learning")
    learning_curve(eater_learn, "picky eater learning")

    print("\nSame task")
    learning_curve(eater_same, "picky eater same")

    print("\nDifferent task - fix")
    learning_curve(eater_different_fix, "picky eater different (fix)")

    print("\nDifferent task - tune")
    learning_curve(eater_different_tune, "picky eater: different (fine tune)")

if __name__ == '__main__':

    simple_maze()
    picky_eater()