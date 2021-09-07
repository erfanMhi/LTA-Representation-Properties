import os
import sys
import numpy as np
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

from plot.plot_utils import *
from plot.plot_paths import *

os.chdir("..")
print("Change dir to", os.getcwd())

def learning_curve(all_paths_dict, title, targets=None, xlim=None, ylim=None, data_label=True):
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
        #print(control)
        returns = arrange_order(control[label])
        draw_curve(returns, plt, label, violin_colors[label], style=curve_styles[label])

    plt.title(title)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if data_label:
        plt.legend()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    # plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
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
    print(title)
    plt.title(title)
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(title), dpi=300)
    # plt.show()
    plt.close()
    plt.clf()

def mountain_car():
    print("\nRep learning")
    learning_curve(mc_learn, "mountain car learning")

def simple_maze():
    print("\nRep learning")
    targets = ["FTA eta=0.2", "FTA+Control1g", "FTA+Control5g",
               "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               "ReLU", "ReLU+Control1g", "ReLU+Control5g",
               "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               # "Random", "Input",
               "Scratch"]

    # learning_curve(gh_online, "maze online measure", targets, xlim=[0, 30])

    learning_curve(gh_same_early, "maze same early", targets, xlim=[0, 15])
    # learning_curve(gh_similar_early, "maze similar early", targets, xlim=[0, 30])
    # learning_curve(gh_diff_early, "maze different (fix) early", targets, xlim=[0, 10], data_label=False)
    draw_label(targets, "maze_label", ncol=2)

    # learning_curve(gh_diff_tune_early, "maze different (fine tune) early", targets, xlim=[0, 10])

    # learning_curve(gh_same_last, "maze same last", targets, xlim=[0, 10])
    # learning_curve(gh_similar_last, "maze similar last", targets, xlim=[0, 10])
    # learning_curve(gh_diff_last, "maze different (fix) last", targets, xlim=[0, 10])
    # learning_curve(gh_diff_tune_last, "maze different (fine tune) last", targets, xlim=[0, 10])
    #
    # learning_curve_mean(gh_online, "maze online measure", targets, xlim=[0, 30])

def picky_eater():
   # learning_curve(crgb_online_dqn, "maze online dqn")
    targets = ["FTA", "FTA+Control", "FTA+XY", "FTA+Color", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               "ReLU", "ReLU+Control", "ReLU+XY", "ReLU+Color", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", 
               "ReLU+SF", "Input", "Random", "Scratch"]

    #targets = ["ReLU+Decoder", "ReLU+Reward", "FTA+Decoder", "FTA+Reward"]

    learning_curve(pe_rep_best,  "Picky Eater Task", targets, xlim=[0,101])
    learning_curve(pe_transfer_best_dissimilar, "Dissimilar Task", targets, xlim=[0,201], legend=False)
    learning_curve(pe_transfer_best_similar, "Same Task", targets, xlim=[0,25])
    #learning_curve(crgb_online_dt_ft, "Dissimilar Task (fine tune)")
    draw_label(targets, "pe_best_chosen_label", ncol=4)

def pe_temp():
    learning_curve(pe_trans_best_temp, "pe diff fix v6 best")

def pe_linear():
    # targets_relu = ["ReLU", "ReLU+Control",
    #                 "ReLU+XY", "ReLU+Color", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF"]
    # targets_fta = ["FTA eta=2", "FTA+Control",
    #                "FTA+XY", "FTA+Color", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",]
    #
    # learning_curve(pe_linear_rep, "pelinear_relu_rep", targets_relu, data_label=False)
    # learning_curve(pe_linear_rep, "pelinear_fta_rep", targets_fta, data_label=False)
    #
    # learning_curve(pe_linear_trans_diff, "pelinear_relu_diff", targets_relu, data_label=False)
    # learning_curve(pe_linear_trans_diff, "pelinear_fta_diff", targets_fta, data_label=False)
    #
    # draw_label(targets_relu, "pelinear_relu_label", ncol=2)
    # draw_label(targets_fta, "pelinear_fta_label", ncol=2)

    targets = ["ReLU", "ReLU+Control",
               "ReLU+XY", "ReLU+Color", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               "FTA eta=2", "FTA+Control",
               "FTA+XY", "FTA+Color", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               "Random", "Input", "Scratch"]
    learning_curve(pe_linear_trans_diff, "pelinear_diff", targets, data_label=False)
    draw_label(targets, "pelinear_label", ncol=2)

def maze_multigoals():
    targets = ["ReLU", "ReLU+Control",
               "ReLU+XY", "ReLU+Color", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               "FTA eta=2", "FTA+Control",
               "FTA+XY", "FTA+Color", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               "Random", "Input", "Scratch"]
    targets = ["ReLU", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
               "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               "Random", "Input", "Scratch"]
 
    # learning_curve(maze_source_best_v12, "maze source")
#    learning_curve(maze_target_same_best_v12, "maze same")
#     learning_curve(maze_target_diff_best_v12, "maze diff same")
    # learning_curve(maze_target_diff_best, "maze dissimilar")

    learning_curve(maze_checkpoint50000_same_best_v12, "maze checkpoint50000 same", xlim=[0, 31], ylim=[0, 1.1])
    learning_curve(maze_checkpoint50000_dissimilar_best_v12, "maze checkpoint50000 dissimilar", xlim=[0, 31], ylim=[0, 1.1])
    learning_curve(maze_checkpoint150000_same_best_v12, "maze checkpoint150000 same", xlim=[0, 31], ylim=[0, 1.1])
    learning_curve(maze_checkpoint150000_dissimilar_best_v12, "maze checkpoint150000 dissimilar", xlim=[0, 31], ylim=[0, 1.1])


if __name__ == '__main__':
    # mountain_car()
    #simple_maze()
    #picky_eater()
    # pe_temp()
    # pe_linear()
    maze_multigoals()