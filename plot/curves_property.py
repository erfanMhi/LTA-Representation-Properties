import os
import sys
import numpy as np
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from plot.plot_utils import *
from plot.plot_paths import *

os.chdir("..")
print("Change dir to", os.getcwd())


# def learning_curve(all_paths_dict, title, key, ylim=None):
#     labels = [i["label"] for i in all_paths_dict]
#     control = load_info(all_paths_dict, 0, key)
#     model_saving = load_info(all_paths_dict, 0, "model")
#     fig, ax = plt.subplots()
#     arranged = {}
#     for label in labels:
#         returns = arrange_order(control[label])
#         arranged[label] = returns
#         draw_curve(returns, ax, label, violin_colors[label])
#     plt.legend()
#     if ylim is not None:
#         ax.set_ylim(ylim[0], ylim[1])
#     else:
#         ylim = ax.get_ylim()
#         ax.set_ylim(ylim[0], ylim[1])
#         for label in labels:
#             vline = arrange_order(model_saving[label], cut_length=False, scale=10000)
#             draw_cut(vline, arranged[label], ax, violin_colors[label], ylim[0])
#     plt.xlabel('step ($10^4$)')
#     plt.ylabel(key)
#     plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#     plt.clf()

def learning_curve_mean(all_paths_dict, title, key, targets=[], xlim=None, ylim=None, show_avg=False, show_model=True, data_label=None, save_path='unknown', legend=False):
    # if len(targets) > 0:
    #     temp = []
    #     for item in all_paths_dict:
    #         if item["label"] in targets:
    #             temp.append(item)
    #     all_paths_dict = temp

    labels = [i["label"] for i in all_paths_dict]
    control = load_info(all_paths_dict, 0, key, label=data_label)

    model_saving = load_info(all_paths_dict, 0, "model")

    arranged = {}
    total = 0
    alpha = 0.5 if show_avg else 1
    linewidth = 1 if show_avg else 1.5
    for label in labels:
        print(label)
#        print(control[label])
        returns = arrange_order(control[label])
        if xlim is not None:
            returns = returns[:, xlim[0]: xlim[1]]
        if key in ["lipschitz", "interf"]:
            returns = 1 - (returns - returns.min()) / (returns.max() - returns.min())
        if xlim is not None and xlim[0] == 1: # fill nan on x=0
            returns = np.concatenate([np.zeros((len(returns), 1))+np.nan, returns], axis=1)
        arranged[label] = returns

    fig, ax = plt.subplots()
    labels = targets
    for label in labels:
        print(label)
        returns = arranged[label]
        draw_curve(returns, ax, label, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
        total = returns if type(total) == int else total + returns
    if show_avg:
        draw_curve(total/len(labels), plt, "Avg", "black", linewidth=3)

    # plt.title(title)
    if legend:
        # fontP = FontProperties()
        # fontP.set_size('xx-small')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
        plt.legend()
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1])

    if show_model:
        for label in labels:
            vline = arrange_order(model_saving[label], cut_length=False, scale=10000)
            draw_cut(vline, arranged[label], ax, violin_colors[label], ylim[0])

    # plt.xlabel('step ($10^4$)')
    # plt.ylabel(key)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    save_path = save_path if save_path!='unknown' else title
    plt.savefig("plot/img/{}.pdf".format(save_path), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()

def simple_maze():
    # print("\nRep learning")
    targets = ["FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8", "ReLU"]
    learning_curve_mean(gh_online, "maze_eta_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False)
    learning_curve_mean(gh_online, "maze_eta_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False)
    learning_curve_mean(gh_online, "maze_eta_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.7], show_avg=False)
    learning_curve_mean(gh_online, "maze_eta_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False)
    learning_curve_mean(gh_online, "maze_eta_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False)
    learning_curve_mean(gh_online, "maze_eta_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.3, 1], show_avg=False)
    learning_curve_mean(gh_online, "maze eta return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False)
    learning_curve_mean(gh_same_early, "maze eta same return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze eta similar return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze eta diff (fix) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze eta diff (tune) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_same_early, "maze eta same complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.2, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze eta similar complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.2, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze eta diff (fix) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.2, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze eta diff (tune) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.2, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_same_early, "maze eta same interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze eta similar interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze eta diff (fix) interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze eta diff (tune) interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_same_early, "maze eta same diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze eta similar diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze eta diff (fix) diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze eta diff (tune) diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)

    targets = ["FTA eta=0.2", "FTA+Control1g", "FTA+Control5g",
               "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               # "ReLU",
               # "Random", "Input"
               ]
    learning_curve_mean(gh_online, "maze_fta_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False)
    learning_curve_mean(gh_online, "maze_fta_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False)
    learning_curve_mean(gh_online, "maze_fta_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.7], show_avg=False)
    learning_curve_mean(gh_online, "maze_fta_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False)
    learning_curve_mean(gh_online, "maze_fta_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False)
    learning_curve_mean(gh_online, "maze_fta_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.3, 1], show_avg=False)
    learning_curve_mean(gh_same_early, "maze fta same complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.1, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze fta similar complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.1, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze fta diff (fix) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.1, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze fta diff (tune) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.1, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_same_early, "maze fta same interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze fta similar interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze fta diff (fix) interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze fta diff (tune) interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_same_early, "maze fta same diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze fta similar diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze fta diff (fix) diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze fta diff (tune) diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    learning_curve_mean(gh_online, "maze fta return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=True)
    learning_curve_mean(gh_same_early, "maze fta same return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze fta similar return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze fta diff (fix) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze fta diff (tune) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)

    targets = ["ReLU", "ReLU+Control1g", "ReLU+Control5g",
               "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               # "Random", "Input"
               ]
    learning_curve_mean(gh_online, "maze_relu_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False)
    learning_curve_mean(gh_online, "maze_relu_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False)
    learning_curve_mean(gh_online, "maze_relu_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.7], show_avg=False)
    learning_curve_mean(gh_online, "maze_relu_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False)
    learning_curve_mean(gh_online, "maze_relu_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False)
    learning_curve_mean(gh_online, "maze_relu_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.3, 1], show_avg=False)

    learning_curve_mean(gh_online, "maze relu return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=True)
    learning_curve_mean(gh_same_early, "maze relu same return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze relu similar return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze relu diff (fix) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze relu diff (tune) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)

    targets = ["ReLU", "ReLU+Control5g", "ReLU+Reward",
               "FTA eta=0.2", "FTA+Control5g", "FTA+Reward",
               ]
    learning_curve_mean(gh_online, "maze_chosen_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], show_avg=False, show_model=True)
    learning_curve_mean(gh_online, "maze_chosen_online_distance", key="distance", targets=targets, xlim=[0, 31], show_avg=False, show_model=True)
    learning_curve_mean(gh_online, "maze_chosen_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], show_avg=False, show_model=True)
    learning_curve_mean(gh_online, "maze_chosen_online_interf", key="interf", targets=targets, xlim=[1, 31], show_avg=False, show_model=True)
    learning_curve_mean(gh_online, "maze_chosen_online_diversity", key="diversity", targets=targets, xlim=[0, 31], show_avg=False, show_model=True)
    learning_curve_mean(gh_online, "maze_chosen_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], show_avg=False, show_model=True, legend=True)

    learning_curve_mean(gh_online, "maze chosen return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=True)
    learning_curve_mean(gh_same_early, "maze chosen same return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze chosen similar return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze chosen diff (fix) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze chosen diff (tune) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)

    targets = ["ReLU",
               "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
               "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               # "Random", "Input"
               ]
    learning_curve_mean(gh_online, "maze online lipschitz", key="lipschitz", targets=targets, xlim=[0, 31])
    learning_curve_mean(gh_online, "maze online distance", key="distance", targets=targets, xlim=[0, 31])
    learning_curve_mean(gh_online, "maze online orthogonal", key="ortho", targets=targets, xlim=[0, 31])
    learning_curve_mean(gh_online, "maze online interf", key="interf", targets=targets, xlim=[1, 31])#, ylim=[0, 0.002])
    learning_curve_mean(gh_online, "maze online diversity", key="diversity", targets=targets, xlim=[0, 31])
    learning_curve_mean(gh_online, "maze online sparsity", key="sparsity", targets=targets, xlim=[0, 31])

    learning_curve_mean(gh_same_early, "maze same complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
    learning_curve_mean(gh_similar_early, "maze similar complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
    learning_curve_mean(gh_diff_early, "maze diff (fix) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze diff (tune) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)

    learning_curve_mean(gh_same_early, "maze all same return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_similar_early, "maze all similar return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_early, "maze all diff (fix) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    learning_curve_mean(gh_diff_tune_early, "maze all diff (tune) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)

def picky_eater():
    print("\nRep learning")
    targets_lta= [
        'LTA', 
        # 'LTA+XY',
        # 'LTA+Control'
        'LTA+Decoder', 
        'LTA+Decoder', 
        'LTA+NAS', 
        'LTA+SF'
            ]
    
    targets_relu= [
        'ReLU', 
        'ReLU+XY',
        'ReLU+Control'
        'ReLU+Decoder', 
        'ReLU+Decoder', 
        'ReLU+NAS', 
        'ReLU+SF'
            ]

    learning_curve_mean(crgb_online, "maze return", key="return", xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_return')
    learning_curve_mean(crgb_online, "Complexity Reduction", key="lipschitz", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_comp_reduc')
    learning_curve_mean(crgb_online, "Dynamic Awareness", key="distance", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_distance')
    learning_curve_mean(crgb_online, "Orthogonal", key="ortho", data_label='Random', xlim=[0, 101],ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_ortho') # learning_curve_mean(crgb_online, "maze online interf", key="interf", xlim=[0, 101], show_model=False, targets=targets_lta, save_path='lta_interf') learning_curve_mean(crgb_online, "Diversity", key="diversity", data_label='Random', xlim=[0, 101],ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_diversity')
    learning_curve_mean(crgb_online, "Sparsity", key="sparsity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_sparsity')
 
    learning_curve_mean(crgb_online, "maze return", key="return", xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_return')
    learning_curve_mean(crgb_online, "Complexity Reduction", key="lipschitz", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_comp_reduc')
    learning_curve_mean(crgb_online, "Dynamic Awareness", key="distance", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_distance')
    learning_curve_mean(crgb_online, "Orthogonal", key="ortho", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_ortho')
    # learning_curve_mean(crgb_online, "maze online interf", key="interf", xlim=[0, 101], show_model=False, targets=targets_lta, save_path='lta_interf')
    learning_curve_mean(crgb_online, "Diversity", key="diversity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_diversity')
    learning_curve_mean(crgb_online, "Sparsity", key="sparsity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_sparsity')
       
    # learning_curve_mean(gh_diff_tune_early, "maze diff (tune) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)

    # learning_curve_mean(gh_same_early, "maze same noninterf", key="interf", targets=targets, xlim=[0, 11], show_model=False)
    # learning_curve_mean(gh_similar_early, "maze similar noninterf", key="interf", targets=targets, xlim=[0, 11], show_model=False)
    # learning_curve_mean(gh_diff_early, "maze diff (fix) noninterf", key="interf", targets=targets, xlim=[0, 11], show_model=False)
    # learning_curve_mean(gh_diff_tune_early, "maze diff (tune) noninterf", key="interf", targets=targets, xlim=[0, 11], show_model=False)

    # learning_curve_mean(gh_online, "maze online return", key="return", targets=targets, xlim=[0, 31], show_avg=False)
    # learning_curve_mean(gh_same_early, "maze same return", key="return", targets=targets, xlim=[0, 11], show_avg=False, show_model=False)
    # learning_curve_mean(gh_similar_early, "maze similar return", key="return", targets=targets, xlim=[0, 11], show_avg=False, show_model=False)
    # learning_curve_mean(gh_diff_early, "maze diff (fix) return", key="return", targets=targets, xlim=[0, 11], show_avg=False, show_model=False)
    # learning_curve_mean(gh_diff_tune_early, "maze diff (tune) return", key="return", targets=targets, xlim=[0, 11], show_avg=False, show_model=False)

def picky_eater():
    print("\nRep learning")
    learning_curve_mean(crgb_online, "maze return", key="return")
    learning_curve_mean(crgb_online, "maze online lipschitz", key="lipschitz")
    learning_curve_mean(crgb_online, "maze online distance", key="distance")
    learning_curve_mean(crgb_online, "maze online orthogonal", key="ortho")
    learning_curve_mean(crgb_online, "maze online noninterf", key="noninterf")
    learning_curve_mean(crgb_online, "maze online decorr", key="decorr")
    learning_curve_mean(crgb_online, "maze online sparsity", key="sparsity")

if __name__ == '__main__':
    # mountain_car()
    simple_maze()
    picky_eater()
