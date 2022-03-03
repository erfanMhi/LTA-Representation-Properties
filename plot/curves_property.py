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


def learning_curve_mean(all_paths_dict, title, key,
                        targets=[], xlim=None, ylim=None, show_avg=False, show_model=True, data_label=None, save_path='unknown', legend=False, independent_runs=False,
                        xscale="linear", xticks=None,
                        given_ax=None, given_color="black"):

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

    if given_ax is None:
        fig, ax = plt.subplots()
    else:
        ax = given_ax
    labels = targets
    conv_counter = 0

    conv_counter = 0
    max_intensity = 0
    min_intensity = float('+inf')
    for k, label in enumerate(labels):
        vline = arrange_order(model_saving[label], cut_length=False, scale=10000)
        intensity = convergence_intensity(vline, arranged[label])
        if max_intensity <= intensity:
            max_intensity = intensity
        if min_intensity >= min_intensity:
            min_intensity = intensity

    for k, label in enumerate(labels):
        print('----------------------draw_curve---------------------')
        returns = arranged[label]
        # print('min returns: ', returns[:,-1:].min())
        # print('max returns: ', returns[:,-1:].max())
        # draw_curve(returns, ax, label, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
        vline = arrange_order(model_saving[label], cut_length=False, scale=10000)
        intensity = convergence_intensity(vline, arranged[label])

        
        # curve_alpha = 0.2 + (1 - (intensity - min_intensity)/(max_intensity - min_intensity) * 0.8)

        if is_converged(vline, arranged[label]):
            curve_alpha = 1.0
            conv_counter += 1
        else:
            curve_alpha = 0.3
        
        draw_curve(returns, ax, label, given_color, "-", alpha=curve_alpha, linewidth=linewidth, draw_ste=False, break_point=int(vline.max()))

        if independent_runs:
            for i, r in enumerate(returns):
                # draw_curve(r.reshape((1, -1)), ax, None, violin_colors[label], curve_styles[label], alpha=alpha, linewidth=linewidth)
                draw_curve(r.reshape((1, -1)), ax, None, given_color, "-", alpha=alpha, linewidth=linewidth)
            ax.plot([], color=violin_colors[label], linestyle=curve_styles[label], label=label)
        else:
            if show_avg:
                total = returns if type(total) == int else total + returns
        
    print('{}_key: '.format(key), conv_counter, '/', len(labels))
    if show_avg:
        draw_curve(total/len(labels), plt, "Avg", "black", linewidth=3)

    # plt.title(property_keys[key], fontsize=30)
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
    ax.set_xscale(xscale)
    if xticks is not None:
        ax.set_xticks(xticks, xticks)
    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)

    # if show_model:
    #     for label in labels:
    #         vline = arrange_order(model_saving[label], cut_length=False, scale=10000)
    #         if independent_runs:
    #             for i in range(len(vline)):
    #                 # draw_cut(vline[i].reshape((1, -1)), arranged[label][i].reshape((1, -1)), ax, violin_colors[label], ylim[0])
    #                 draw_cut(vline[i].reshape((1, -1)), arranged[label][i].reshape((1, -1)), ax, given_color, ylim[0])
    #         else:
    #             # draw_cut(vline, arranged[label], ax, violin_colors[label], ylim[0])
    #             draw_cut(vline, arranged[label], ax, given_color, ylim[0])

    # plt.xlabel('step ($10^4$)')
    # plt.ylabel(key)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    save_path = save_path if save_path!='unknown' else title
    
    if given_ax is None:
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
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
    ]
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31], ylim=[0.1, 1], show_avg=False, show_model=True, data_label=None)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_distance", key="distance", targets=targets, xlim=[0, 31], ylim=[0.2, 0.9], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31], ylim=[0, 0.8], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_interf", key="interf", targets=targets, xlim=[1, 31], ylim=[0.3, 1], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_diversity", key="diversity", targets=targets, xlim=[0, 31], ylim=[0, 0.9], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31], ylim=[0.4, 1], show_avg=False, show_model=True)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_transfer_sweep_v13)), "linear/properties/maze_all_online_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1], show_avg=False, show_model=True)

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF", "ReLU+ATC",
        "ReLU(L)",
        "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF", "ReLU(L)+ATC",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
    ]
    fig, axs = plt.subplots(1, 6, figsize=(32, 4))
    learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_lipschitz", key="lipschitz", targets=targets, xlim=[0, 31],
                        ylim=[0.1, 1], show_avg=False, show_model=True, data_label=None, xscale="log", xticks=[10], given_ax=axs[0], given_color="#3498db")
    axs[0].set_title("Complexity\nReduction", fontsize=30)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_distance", key="distance", targets=targets, xlim=[0, 31],
                        ylim=[0.2, 0.9], show_avg=False, show_model=True, xscale="log", xticks=[10], given_ax=axs[3], given_color="#c0392b")
    axs[3].set_title("Dynamic\nAwareness", fontsize=30)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_orthogonal", key="ortho", targets=targets, xlim=[0, 31],
                        ylim=[0, 0.8], show_avg=False, show_model=True, xscale="log", xticks=[10], given_ax=axs[2], given_color="#9b59b6")
    axs[2].set_title("Orthogonality", fontsize=30)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_interf", key="interf", targets=targets, xlim=[1, 31],
                        ylim=[0.3, 1], show_avg=False, show_model=True, xscale="log", xticks=[10], given_ax=axs[4], given_color="#1abc9c")
    axs[4].set_title("Non-\ninterference", fontsize=30)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_diversity", key="diversity", targets=targets, xlim=[0, 31],
                        ylim=[0, 0.9], show_avg=False, show_model=True, xscale="log", xticks=[10], given_ax=axs[1], given_color="#e67e22")
    axs[1].set_title("Diversity", fontsize=30)
    learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_sparsity", key="sparsity", targets=targets, xlim=[0, 31],
                        ylim=[0.4, 1], show_avg=False, show_model=True, xscale="log", xticks=[10], given_ax=axs[5], given_color='#f1c40f') #given_color="#34495e")
    axs[5].set_title("Sparsity", fontsize=30)
    # learning_curve_mean(preprocess_path(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)), "nonlinear/properties/maze_all_online_return", key="return", targets=targets, xlim=[0, 31], ylim=[0, 1], show_avg=False, show_model=True, xscale="log")
    plt.savefig("plot/img/nonlinear/properties.pdf", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    simple_maze()
