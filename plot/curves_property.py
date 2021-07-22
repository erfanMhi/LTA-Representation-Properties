import os
import sys 
import numpy as np
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from plot.plot_utils import *
from plot.plot_paths import *

from itertools import chain, combinations
os.chdir("..")
print("Change dir to", os.getcwd())

def draw_label(targets, save_path, ncol):
    plt.figure(figsize=(0.1, 2))
    for label in targets:
        plt.plot([], color=violin_colors[label], linestyle=curve_styles[label], label=label)
    plt.axis('off')
    plt.legend(ncol=ncol)
    plt.savefig("plot/img/{}.pdf".format(save_path), dpi=300, bbox_inches='tight')
    # plt.savefig("plot/img/{}.png".format(save_path), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()

def learning_curve_mean(all_paths_dict, title, key, targets=[], xlim=None, ylim=None, show_avg=False, show_model=True, data_label=None, save_path='unknown', legend=False, independent_runs=False):

    labels = [i["label"] for i in all_paths_dict] if targets == [] else targets
    control = load_info(all_paths_dict, 0, key, label=data_label)
    if show_model:
        model_saving = load_info(all_paths_dict, 0, "model")
    arranged = {}
    total = 0
    alpha = 0.5 if show_avg else 1
    linewidth = 1 if show_avg else 3
    for label in labels:
        # print(label)
        # print(control[label])
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
    for k, label in enumerate(labels):
        print('----------------------draw_curve---------------------')
        returns = arranged[label]
        # print('min returns: ', returns[:,-1:].min())
        # print('max returns: ', returns[:,-1:].max())
        draw_curve(returns, ax, label, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
        if independent_runs:
            for i, r in enumerate(returns):
                # draw_curve(r.reshape((1, -1)), ax, None, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
                draw_curve(r.reshape((1, -1)), ax, None, c_default[k], curve_styles[label], alpha=alpha, linewidth=linewidth)
            plt.plot([], color=violin_colors[label], linestyle=curve_styles[label], label=label)
        else:
# <<<<<<< HEAD
            # pass
            # total = returns if type(total) == int else total + returns
# =======
            if show_avg:
                total = returns if type(total) == int else total + returns
    if show_avg:
        draw_curve(total/len(labels), plt, "Avg", "black", linewidth=3)

    plt.title(title)
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

    ax.set_xticks([xlim[0], xlim[1]-1])
    if ylim[1] > 1:
        ax.set_yticks([ylim[0], int(ylim[1])])
    else:
        ax.set_yticks([ylim[0], ylim[1]])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)

    if show_model:
        for label in labels:
            vline = arrange_order(model_saving[label], cut_length=False, scale=10000)
            draw_cut(vline, arranged[label], ax, violin_colors[label], ylim[0])

    plt.xlabel('step ($10^4$)')
    # plt.ylabel(key)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    save_path = save_path if save_path!='unknown' else title
    
    if data_label is None:
        plt.savefig("plot/img/{}.pdf".format(save_path), dpi=300, bbox_inches='tight')
    else:    
        plt.savefig("plot/img/{}_{}.pdf".format(data_label, save_path), dpi=300, bbox_inches='tight')
    
    # plt.savefig("plot/img/{}.png".format(save_path), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()

def learning_curve_mean_label(all_paths_dict, title, key, targets=[], xlim=None, ylim=None, show_avg=False, show_model=True, data_labels=['return'], save_path='unknown', legend=False, independent_runs=False):

    fig, ax = plt.subplots()
    for data_label in data_labels:
        labels = [i["label"] for i in all_paths_dict]
        control = load_info(all_paths_dict, 0, key, label=data_label)
        if show_model:
            model_saving = load_info(all_paths_dict, 0, "model")
        arranged = {}
        total = 0
        alpha = 0.5 if show_avg else 1
        linewidth = 1 if show_avg else 1.5
        label = labels[0]
        print(data_label)
#        print(control[label])
        returns = arrange_order(control[label])
        if xlim is not None:
            returns = returns[:, xlim[0]: xlim[1]]
        if key in ["lipschitz", "interf"]:
            returns = 1 - (returns - returns.min()) / (returns.max() - returns.min())
        if xlim is not None and xlim[0] == 1: # fill nan on x=0
            returns = np.concatenate([np.zeros((len(returns), 1))+np.nan, returns], axis=1)
        arranged[label] = returns

        #labels = targets
        # for k, label in enumerate(labels):
        print('----------------------draw_curve---------------------')
        returns = arranged[label]
        # print('min returns: ', returns[:,-1:].min())
        # print('max returns: ', returns[:,-1:].max())
#        draw_curve(returns, ax, label, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
        draw_curve(returns, ax, data_label, alpha=alpha, linewidth=linewidth)
#         if independent_runs:
            # for i, r in enumerate(returns):
                # # draw_curve(r.reshape((1, -1)), ax, None, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
                # draw_curve(r.reshape((1, -1)), ax, None, c_default[k], curve_styles[label], alpha=alpha, linewidth=linewidth)
            # plt.plot([], color=violin_colors[label], linestyle=curve_styles[label], label=label)
        # else:
            # pass
            # # total = returns if type(total) == int else total + returns
        # if show_avg:
            # draw_curve(total/len(labels), plt, "Avg", "black", linewidth=3)

    plt.title(title)
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

    plt.xlabel('step ($10^4$)')
    # plt.ylabel(key)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    save_path = save_path if save_path!='unknown' else title
    
    if data_label is None:
        plt.savefig("plot/img/{}.pdf".format(save_path), dpi=300, bbox_inches='tight')
    else:    
        plt.savefig("plot/img/{}_{}.pdf".format(data_label, save_path), dpi=300, bbox_inches='tight')
    
    # plt.savefig("plot/img/{}.png".format(save_path), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()


def simple_maze():
    # print("\nRep learning")
    # targets = ["FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8", "ReLU"]
    # learning_curve_mean(gh_online, "maze_eta_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False)
    # learning_curve_mean(gh_online, "maze_eta_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False)
    # learning_curve_mean(gh_online, "maze_eta_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.7], show_avg=False)
    # learning_curve_mean(gh_online, "maze_eta_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False)
    # learning_curve_mean(gh_online, "maze_eta_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False)
    # learning_curve_mean(gh_online, "maze_eta_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.3, 1], show_avg=False)
    # learning_curve_mean(gh_online, "maze_eta_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False)
    # learning_curve_mean(gh_same_early, "maze_eta_same_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_similar_early, "maze_eta_similar_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_diff_early, "maze_eta_diff_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_diff_tune_early, "maze_eta_diff_(tune)_return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_same_early, "maze_eta_same_complexity_reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.2, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_similar_early, "maze_eta_similar_complexity_reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.2, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_diff_early, "maze_eta_diff_(fix)_complexity_reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.2, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_diff_tune_early, "maze_eta_diff_(tune)_complexity_reduction", key="lipschitz", targets=targets, xlim=[0, 11], ylim=[0.2, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_same_early, "maze_eta_same_interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_similar_early, "maze_eta_similar_interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_diff_early, "maze_eta_diff_(fix)_interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_diff_tune_early, "maze_eta_diff_(tune)_interf", key="interf", targets=targets, xlim=[1, 11], ylim=[0.35, 1.1], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_same_early, "maze_eta_same_diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_similar_early, "maze_eta_similar_diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_diff_early, "maze_eta_diff_(fix)_diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    # # learning_curve_mean(gh_diff_tune_early, "maze_eta_diff_(tune)_diversity", key="diversity", targets=targets, xlim=[0, 11], ylim=[0.2, 0.7], show_avg=False, show_model=False)
    # draw_label(targets, "maze_eta_label", ncol=5)
    #
    # targets = ["ReLU",
    #            "ReLU+Control1g","ReLU+Control5g", "ReLU+Reward",
    #            "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+SF",
    #            # "Random", "Input"
    #            ]
    # learning_curve_mean(gh_online, "maze_relu_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False)
    # learning_curve_mean(gh_online, "maze_relu_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False)
    # learning_curve_mean(gh_online, "maze_relu_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.7], show_avg=False)
    # learning_curve_mean(gh_online, "maze_relu_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False)
    # learning_curve_mean(gh_online, "maze_relu_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False)
    # learning_curve_mean(gh_online, "maze_relu_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.3, 1], show_avg=False)
    # learning_curve_mean(gh_online, "maze_relu_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=True)
    # learning_curve_mean(gh_same_early, "maze_relu_same_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_similar_early, "maze_relu_similar_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_diff_early, "maze_relu_diff_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # draw_label(targets, "maze_relu_label", ncol=4)
    #
    # targets = ["FTA eta=0.2",
    #            "FTA+Control1g","FTA+Control5g", "FTA+Reward",
    #            "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+SF",
    #            ]
    # learning_curve_mean(gh_online, "maze_fta_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False, show_model=True, legend=False, independent_runs=False)
    # learning_curve_mean(gh_online, "maze_fta_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_fta_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.7], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_fta_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_fta_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_fta_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.3, 1], show_avg=False, show_model=True)
    # learning_curve_mean(gh_same_early, "maze_fta_same_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_similar_early, "maze_fta_similar_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_diff_early, "maze_fta_diff_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # draw_label(targets, "maze_fta_label", ncol=4)

    targets = ["ReLU+Control5g", "ReLU+Reward",
               "FTA+Control5g", "FTA+Reward"
               ]
    # learning_curve_mean(gh_online, "maze online lipschitz", key="lipschitz", targets=targets, xlim=[0, 31])
    # learning_curve_mean(gh_online, "maze online distance", key="distance", targets=targets, xlim=[0, 31])
    # learning_curve_mean(gh_online, "maze online orthogonal", key="ortho", targets=targets, xlim=[0, 31])
    # learning_curve_mean(gh_online, "maze online interf", key="interf", targets=targets, xlim=[1, 31])#, ylim=[0, 0.002])
    # learning_curve_mean(gh_online, "maze online diversity", key="diversity", targets=targets, xlim=[0, 31])
    # learning_curve_mean(gh_online, "maze online sparsity", key="sparsity", targets=targets, xlim=[0, 31])
    #
    # learning_curve_mean(gh_same_early, "maze same complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
    # learning_curve_mean(gh_similar_early, "maze similar complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
    # learning_curve_mean(gh_diff_early, "maze diff (fix) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
    # learning_curve_mean(gh_diff_tune_early, "maze diff (tune) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
    #
    # learning_curve_mean(gh_same_early, "maze all same return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_similar_early, "maze all similar return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_diff_early, "maze all diff (fix) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_diff_tune_early, "maze all diff (tune) return", key="return", targets=targets, xlim=[0, 11], ylim=[0, 1.1], show_avg=False, show_model=False)

def picky_eater(data_label='Random'):
    print("\nRep learning")
#     targets_lta= [
        # 'FTA eta=2', 
        # 'FTA+XY',
        # 'FTA+Control'
        # 'FTA+Reward', 
        # 'FTA+Decoder', 
        # 'FTA+NAS', 
        # 'FTA+SF'
            # ]
    
    # targets_relu= [
        # 'ReLU', 
        # 'ReLU+XY',
        # 'ReLU+Control'
        # 'ReLU+Decoder', 
        # 'ReLU+Reward', 
        # 'ReLU+NAS', 
        # 'ReLU+SF'
            # ]

    # learning_curve_mean(crgb_online, "maze return", key="return", xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_return')
    # learning_curve_mean(crgb_online, "Complexity Reduction", key="lipschitz", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_comp_reduc')
    # learning_curve_mean(crgb_online, "Dynamic Awareness", key="distance", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_distance')
    # learning_curve_mean(crgb_online, "Orthogonal", key="ortho", data_label='Random', xlim=[0, 101],ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_ortho') 
    # learning_curve_mean(crgb_online, "maze online interf", key="interf", xlim=[0, 101], show_model=False, targets=targets_lta, save_path='lta_interf') learning_curve_mean(crgb_online, "Diversity", key="diversity", data_label='Random', xlim=[0, 101],ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_diversity')
    # learning_curve_mean(crgb_online, "Sparsity", key="sparsity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_sparsity')
 
    # learning_curve_mean(crgb_online, "maze return", key="return", xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_return')
    # learning_curve_mean(crgb_online, "Complexity Reduction", key="lipschitz", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_comp_reduc')
    # learning_curve_mean(crgb_online, "Dynamic Awareness", key="distance", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_distance')
    # learning_curve_mean(crgb_online, "Orthogonal", key="ortho", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_ortho')
    # learning_curve_mean(crgb_online, "maze online interf", key="interf", xlim=[0, 101], show_model=False, targets=targets_lta, save_path='lta_interf')
    # learning_curve_mean(crgb_online, "Diversity", key="diversity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_diversity')
    # learning_curve_mean(crgb_online, "Sparsity", key="sparsity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_sparsity')
 
    targets_relu = [
        # 'ReLU', 
        # 'ReLU+XY',
        'ReLU+Control',
#         'ReLU+Decoder', 
        # 'ReLU+Reward', 
        # 'ReLU+NAS', 
        # 'ReLU+SF',
         # 'FTA', 
        # 'FTA+XY',
        'FTA+Control',
        # 'FTA+Reward', 
        # 'FTA+Decoder', 
        # 'FTA+NAS', 
        # 'FTA+SF'
             ]
#     learning_curve_mean(gh_online, "maze_chosen_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_chosen_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_chosen_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.7], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_chosen_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_chosen_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False, show_model=True)
    # learning_curve_mean(gh_online, "maze_chosen_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.3, 1], show_avg=False, show_model=True)
    # learning_curve_mean(gh_same_early, "maze_chosen_same_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_similar_early, "maze_chosen_similar_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # learning_curve_mean(gh_diff_early, "maze_chosen_diff_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1.1], show_avg=False, show_model=False)
    # draw_label(targets, "maze_chosen_label", ncol=4)


    # learning_curve_mean(crgb_online_best_3_f_control_test, "maze return", key="return", xlim=[0, 101], ylim=[-1, 4], independent_runs=True, show_model=False, targets=targets_relu, save_path='crgb_return', legend=True)
#     learning_curve_mean(crgb_online, "Complexity Reduction", key="lipschitz", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_comp_reduc')
    # learning_curve_mean(crgb_online, "Dynamic Awareness", key="distance", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_distance')
    # learning_curve_mean(crgb_online, "Orthogonal", key="ortho", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_ortho')
    # learning_curve_mean(crgb_online, "Non-Interference", key="interf", xlim=[1, 151], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_interf')
    # learning_curve_mean(crgb_online, "Diversity", key="diversity", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_diversity')
    # learning_curve_mean(crgb_online, "Sparsity", key="sparsity", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_sparsity', legend=True)
     
    learning_curve_mean(crgb_online_best_4_f_randc, "return", key="return", xlim=[0, 151], ylim=[-1, 3.1], independent_runs=False, show_model=False, targets=targets_relu, save_path='crgb_return')
    learning_curve_mean(crgb_online_best_4_f_randc, "Complexity Reduction", key="lipschitz", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_comp_reduc')
    learning_curve_mean(crgb_online_best_4_f_randc, "Dynamic Awareness", key="distance", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_distance')
    learning_curve_mean(crgb_online_best_4_f_randc, "Orthogonal", key="ortho", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_ortho')
    learning_curve_mean(crgb_online_best_4_f_randc, "Non-Interference", key="interf", xlim=[1, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_interf')
    learning_curve_mean(crgb_online_best_4_f_randc, "Diversity", key="diversity", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_diversity')
    learning_curve_mean(crgb_online_best_4_f_randc, "Sparsity", key="sparsity", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_sparsity', legend=True)
      
    learning_curve_mean(crgb_online_best_4_f_randc, "return", key="return", xlim=[0, 151], ylim=[-1, 3.1], independent_runs=False, show_model=False, targets=targets_relu, save_path='crgb_return')
    learning_curve_mean(crgb_online_best_4_f_randc, "Complexity Reduction", key="lipschitz", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_comp_reduc')
    learning_curve_mean(crgb_online_best_4_f_randc, "Dynamic Awareness", key="distance", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_distance')
    learning_curve_mean(crgb_online_best_4_f_randc, "Orthogonal", key="ortho", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_ortho')
    learning_curve_mean(crgb_online_best_4_f_randc, "Non-Interference", key="interf", xlim=[1, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_interf')
    learning_curve_mean(crgb_online_best_4_f_randc, "Diversity", key="diversity", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_diversity')
    learning_curve_mean(crgb_online_best_4_f_randc, "Sparsity", key="sparsity", data_label=data_label, xlim=[0, 151], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='crgb_sparsity', legend=True)
           
    # learning_curve_mean(gh_diff_tune_early, "maze diff (tune) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
# def picky_eater(data_label='Random'):
#     print("\nRep learning")
# #     targets_lta= [
#         # 'FTA eta=2',
#         # 'FTA+XY',
#         # 'FTA+Control'
#         # 'FTA+Reward',
#         # 'FTA+Decoder',
#         # 'FTA+NAS',
#         # 'FTA+SF'
#             # ]
#
#     # targets_relu= [
#         # 'ReLU',
#         # 'ReLU+XY',
#         # 'ReLU+Control'
#         # 'ReLU+Decoder',
#         # 'ReLU+Reward',
#         # 'ReLU+NAS',
#         # 'ReLU+SF'
#             # ]
#
#     # learning_curve_mean(crgb_online, "maze return", key="return", xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_return')
#     # learning_curve_mean(crgb_online, "Complexity Reduction", key="lipschitz", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_comp_reduc')
#     # learning_curve_mean(crgb_online, "Dynamic Awareness", key="distance", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_distance')
#     # learning_curve_mean(crgb_online, "Orthogonal", key="ortho", data_label='Random', xlim=[0, 101],ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_ortho')
#     # learning_curve_mean(crgb_online, "maze online interf", key="interf", xlim=[0, 101], show_model=False, targets=targets_lta, save_path='lta_interf') learning_curve_mean(crgb_online, "Diversity", key="diversity", data_label='Random', xlim=[0, 101],ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_diversity')
#     # learning_curve_mean(crgb_online, "Sparsity", key="sparsity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_lta, save_path='lta_sparsity')
#
#     # learning_curve_mean(crgb_online, "maze return", key="return", xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_return')
#     # learning_curve_mean(crgb_online, "Complexity Reduction", key="lipschitz", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_comp_reduc')
#     # learning_curve_mean(crgb_online, "Dynamic Awareness", key="distance", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_distance')
#     # learning_curve_mean(crgb_online, "Orthogonal", key="ortho", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_ortho')
#     # learning_curve_mean(crgb_online, "maze online interf", key="interf", xlim=[0, 101], show_model=False, targets=targets_lta, save_path='lta_interf')
#     # learning_curve_mean(crgb_online, "Diversity", key="diversity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_diversity')
#     # learning_curve_mean(crgb_online, "Sparsity", key="sparsity", data_label='Random', xlim=[0, 101], ylim=[0, 1.1], show_model=False, targets=targets_relu, save_path='relu_sparsity')
#
#     targets_relu = [
#         'ReLU',
#         'ReLU+XY',
#         'ReLU+Control',
#         'ReLU+Decoder',
#         'ReLU+Reward',
#         'ReLU+NAS',
#         'ReLU+SF',
#          'FTA eta=2',
#         'FTA+XY',
#         'FTA+Control',
#         'FTA+Reward',
#         'FTA+Decoder',
#         'FTA+NAS',
#         'FTA+SF'
#             ]
#
#     # learning_curve_mean(crgb_diff_tune_early, "maze return", key="return", xlim=[0, 101], ylim=[0, 6], show_model=False, targets=targets_relu, save_path='crgb_return', legend=True)
#
#
#   #   targets_relu = [
#         # 'ReLU',
#         # 'ReLU+Control',
#         # 'ReLU+Decoder',
#         # 'FTA eta=2',
#         # 'FTA+Control',
#         # 'FTA+Decoder',
#   #           ]
#
#     learning_curve_mean(crgb_same_early, "maze return", key="return", xlim=[0, 101], ylim=[0, 6], show_model=False, targets=targets_relu, save_path='crgb_return', legend=True)
#     # learning_curve_mean(crgb_online, "Complexity Reduction", key="lipschitz", data_label=data_label, xlim=[0, 101], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_comp_reduc')
#     # learning_curve_mean(crgb_online, "Dynamic Awareness", key="distance", data_label=data_label, xlim=[0, 101], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_distance')
#     # learning_curve_mean(crgb_online, "Orthogonal", key="ortho", data_label=data_label, xlim=[0, 101], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_ortho')
#     # learning_curve_mean(crgb_online, "Non-Interference", key="interf", xlim=[1, 101], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_interf')
#     # learning_curve_mean(crgb_online, "Diversity", key="diversity", data_label=data_label, xlim=[0, 101], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_diversity', legend=True)
#     # learning_curve_mean(crgb_online, "Sparsity", key="sparsity", data_label=data_label, xlim=[0, 101], ylim=[0, 1.1], show_model=True, targets=targets_relu, save_path='crgb_sparsity')
#
#     # learning_curve_mean(gh_diff_tune_early, "maze diff (tune) complexity reduction", key="lipschitz", targets=targets, xlim=[0, 11], show_model=False)
#
#     # learning_curve_mean(gh_same_early, "maze same noninterf", key="interf", targets=targets, xlim=[0, 11], show_model=False)
#     # learning_curve_mean(gh_similar_early, "maze similar noninterf", key="interf", targets=targets, xlim=[0, 11], show_model=False)
#     # learning_curve_mean(gh_diff_early, "maze diff (fix) noninterf", key="interf", targets=targets, xlim=[0, 11], show_model=False)
#     # learning_curve_mean(gh_diff_tune_early, "maze diff (tune) noninterf", key="interf", targets=targets, xlim=[0, 11], show_model=False)
#
#     # learning_curve_mean(gh_online, "maze online return", key="return", targets=targets, xlim=[0, 31], show_avg=False)
#     # learning_curve_mean(gh_same_early, "maze same return", key="return", targets=targets, xlim=[0, 11], show_avg=False, show_model=False)
#     # learning_curve_mean(gh_similar_early, "maze similar return", key="return", targets=targets, xlim=[0, 11], show_avg=False, show_model=False)
#     # learning_curve_mean(gh_diff_early, "maze diff (fix) return", key="return", targets=targets, xlim=[0, 11], show_avg=False, show_model=False)
#     # learning_curve_mean(gh_diff_tune_early, "maze diff (tune) return", key="return", targets=targets, xlim=[0, 11], show_avg=False, show_model=False)

def picky_eater():
    print("\nRep learning")
    data_label = "Random"
    targets = [
        "ReLU+Control", "ReLU+Color", "FTA+Control", "FTA+Color",
        # "ReLU+Control+XY+Color", "FTA+Control+XY+Color"
    ]

    rep_data_addr = pe_rep_best
    learning_curve_mean(rep_data_addr, "pe return", key="return", data_label=data_label, xlim=[0, 101], ylim=[0, 3], show_model=False, targets=targets, legend=True)
    learning_curve_mean(rep_data_addr, "pe complexity reduction", key="lipschitz", data_label=data_label, xlim=[0, 101], ylim=[0, 1], show_model=False, targets=targets, legend=True)
    learning_curve_mean(rep_data_addr, "pe dynamic awareness", key="distance", data_label=data_label, xlim=[0, 101], ylim=[0, 1], show_model=False, targets=targets, legend=True)
    learning_curve_mean(rep_data_addr, "pe orthogonal", key="ortho", data_label=data_label, xlim=[0, 101], ylim=[0, 1], show_model=False, targets=targets, legend=True)
    learning_curve_mean(rep_data_addr, "pe noninterference", key="interf", data_label=data_label, xlim=[1, 101], ylim=[0, 1], show_model=False, targets=targets, legend=True)
    learning_curve_mean(rep_data_addr, "pe diversity", key="diversity", data_label=data_label, xlim=[0, 101], ylim=[0, 1], show_model=False, targets=targets, legend=True)
    learning_curve_mean(rep_data_addr, "pe sparsity", key="sparsity", data_label=data_label, xlim=[0, 101], ylim=[0, 1], show_model=False, targets=targets, legend=True)
    #learning_curve_mean(rep_data_addr, "pe multual info", key="mi", data_label=data_label, xlim=[0, 101], ylim=[0, 0.05], show_model=False, targets=targets, legend=True)
    learning_curve_mean(pe_transfer_best_dissimilar, "pe transfer", key="return", data_label=data_label, xlim=[0, 101], ylim=[0, 3], show_model=False, targets=targets, legend=True)

def pe_test():
    targets = [
         "ReLU", 
    #"ReLU+Control", #"ReLU+Reward",
    #     "FTA",
        # "FTA+Reward"
    ]
    # learning_curve_mean(pe_best_temp, "pe return", key="return", show_model=False, independent_runs=True, targets=["ReLU"], legend=False)
    # learning_curve_mean(pe_trans_best_temp, "pe trans diff", key="return", show_model=False, independent_runs=False, targets=targets, legend=True)

    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="state-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(green, 0)')

    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="state-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(green, 1)')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="state-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(green, 2)')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="state-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(red, 0)')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="state-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(red, 1)')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="state-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(red, 2)')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="return", show_model=False, independent_runs=False, targets=targets, legend=True)
    
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(red, 2) directed')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(red, 1) directed')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(red, 0) directed')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(green, 0) directed')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(green, 0) directed')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(green, 0) directed')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(green, 0) directed')
    # learning_curve_mean(switch_color_representation_best, "2 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_label='(green, 0) directed')

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    learning_curve_mean_label(switch_color_control_best, "3 Fruits", key="return", show_model=False, independent_runs=False, targets=targets, legend=True)
    for rm_fruits in powerset(list(range(3))):
        data_labels = ['(green, 0) ', '(green, 1) ', '(green, 2) ', '(red, 0) ', '(red, 1) ', '(red, 2) ']
        rewarding_fruit = 'red'

        for rm_fruit in rm_fruits:
            print('({}, {}) '.format(rewarding_fruit, str(rm_fruit)))
            data_labels.remove('({}, {}) '.format(rewarding_fruit, str(rm_fruit)))
        print([l+str(rm_fruits)+ ' directed' for l in data_labels])
        learning_curve_mean_label(switch_color_control_best, "3 Fruits", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_labels=[l+str(rm_fruits)+ ' directed' for l in data_labels])

        learning_curve_mean_label(switch_color_control_best, "3 Fruits Aux Control", key="action-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_labels=[l+str(rm_fruits)+ ' undirected' for l in data_labels])
        
        learning_curve_mean_label(switch_color_control_best, "3 Fruits", key="state-values", show_model=False, independent_runs=False, targets=targets, legend=True, data_labels=[l+str(rm_fruits) for l in data_labels])

if __name__ == '__main__':
    # mountain_car()
    # simple_maze()
    # picky_eater()
#     picky_eater('Random')
    # picky_eater('Red')
    # picky_eater('Green')
#     simple_maze()
    #picky_eater()
    #pe_test()
    #simple_maze()
    picky_eater()
    # picky_eater('Random')
    # picky_eater('Red')
    # picky_eater('Green')
    # pe_test()