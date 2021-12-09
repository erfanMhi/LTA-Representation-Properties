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


def learning_curve_mean(all_paths_dict, title, key, targets=[], xlim=None, ylim=None, show_avg=False, show_model=True, data_label=None, save_path='unknown', legend=False, independent_runs=False):

    labels = [i["label"] for i in all_paths_dict] if targets == [] else targets
    control = load_info(all_paths_dict, None, key, label=data_label)
    if show_model:
        model_saving = load_info(all_paths_dict, None, "model", label=data_label)
    arranged = {}
    total = 0
    alpha = 0.5 if show_avg else 1
    linewidth = 1 if show_avg else 1
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
        # draw_curve(returns, ax, label, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
        draw_curve(returns, ax, label, "black", "-", alpha=alpha, linewidth=linewidth, draw_ste=False)
        if independent_runs:
            for i, r in enumerate(returns):
                # draw_curve(r.reshape((1, -1)), ax, None, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
                draw_curve(r.reshape((1, -1)), ax, None, "black", "-", alpha=alpha, linewidth=linewidth)
            plt.plot([], color=violin_colors[label], linestyle=curve_styles[label], label=label)
        else:
            if show_avg:
                total = returns if type(total) == int else total + returns
    if show_avg:
        draw_curve(total/len(labels), plt, "Avg", "black", linewidth=3)

    plt.title(property_keys[key], fontsize=30)
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

    # ax.set_xticks([xlim[0], xlim[1]-1])
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
            if independent_runs:
                for i in range(len(vline)):
                    # draw_cut(vline[i].reshape((1, -1)), arranged[label][i].reshape((1, -1)), ax, violin_colors[label], ylim[0])
                    draw_cut(vline[i].reshape((1, -1)), arranged[label][i].reshape((1, -1)), ax, "black", ylim[0])
            else:
                # draw_cut(vline, arranged[label], ax, violin_colors[label], ylim[0])
                draw_cut(vline, arranged[label], ax, "black", ylim[0])

    # plt.xlabel('step ($10^4$)')
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

def preprocess_path(paths):
    # gh_nonlinear_transfer_sweep_v13 = [
    #     {"label": "ReLU",
    #      "control": "data/output/test_v13/gridhard/nonlinear_vf/original_0909/transfer/goal_id_{}/dqn/sweep/",
    #      "online_measure": ["data/output/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/sweep/", 3]
    #      },
    new = []
    for obj in paths:
        new.append({"label": obj["label"], "control": obj["online_measure"]})
    return new

def simple_maze():
    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF", "ReLU+ATC",
        "ReLU(L)",
        "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
    ]
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False, show_model=True, data_label=None)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.8], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.4, 1], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1], show_avg=False, show_model=True)

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF", "ReLU+ATC",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
    ]
    learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False, show_model=True, data_label=None)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False, show_model=True)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.8], show_avg=False, show_model=True)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False, show_model=True)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False, show_model=True)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.4, 1], show_avg=False, show_model=True)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1], show_avg=False, show_model=True)

if __name__ == '__main__':
    simple_maze()
