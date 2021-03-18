import os
import sys
import numpy as np
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

from plot.plot_utils import *
from plot.plot_paths import *

os.chdir("..")
print("Change dir to", os.getcwd())


def learning_curve(all_paths_dict, title, key):
    labels = [i["label"] for i in all_paths_dict]
    control = load_info(all_paths_dict, 0, key)
    plt.figure()
    for label in labels:
        returns = arrange_order(control[label])
        draw_curve(returns, plt, label, violin_colors[label])
    # plt.title(title)
    plt.legend()
    # plt.xlim(0, 30)
    plt.xlabel('step ($10^4$)')
    plt.ylabel(key)
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

def simple_maze():
    print("\nRep learning")
    learning_curve(gh_online, "maze return", key="return")
    learning_curve(gh_online, "maze online lipschitz", key="lipschitz")
    learning_curve(gh_online, "maze online distance", key="distance")
    learning_curve(gh_online, "maze online orthogonal", key="ortho")
    learning_curve(gh_online, "maze online noninterf", key="noninterf")
    learning_curve(gh_online, "maze online decorr", key="decorr")
    learning_curve(gh_online, "maze online sparsity", key="sparsity")

def picky_eater():
    print("\nRep learning")
    learning_curve(gh_online, "maze return", key="return")
    learning_curve(gh_online, "maze online lipschitz", key="lipschitz")
    learning_curve(gh_online, "maze online distance", key="distance")
    learning_curve(gh_online, "maze online orthogonal", key="ortho")
    learning_curve(gh_online, "maze online noninterf", key="noninterf")
    learning_curve(gh_online, "maze online decorr", key="decorr")
    learning_curve(gh_online, "maze online sparsity", key="sparsity")

if __name__ == '__main__':
    # mountain_car()
    # simple_maze()
    picky_eater()
