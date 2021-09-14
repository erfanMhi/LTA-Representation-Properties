import os
import sys
import numpy as np

sys.path.insert(0, '..')

import matplotlib.pyplot as plt

from plot.plot_utils import *
from plot.plot_paths import *
from experiment.sweeper import Sweeper

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

def compare_learning_curve(all_paths_dict, title, total_param=None,
                           start_param=0, label_keys=None, key='return'):
    labels = [i["label"] for i in all_paths_dict]
    control = load_return(all_paths_dict, total_param)  # , start_param)
    # control = load_info(all_paths_dict, total_param, key)#, start_param)
    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(8, 8))

    if len(labels) == 1:
        axs = [axs]

    for idx, label in enumerate(labels):
        print("------", label, "------")
        all_params = control[label]
        aucs = []
        for i, (param, returns) in enumerate(all_params.items()):
            l = i
            # if l not in [3, 4, 5, 6, 7]:
            # continue
            if label_keys is not None:
                print(all_paths_dict[idx]['control'])
                root_idx = len('data/output/') + all_paths_dict[idx]['control'].rindex('data/output/')
                print(all_paths_dict[idx]['control'][root_idx:])
                config_path = 'experiment/config/' + all_paths_dict[idx]['control'][root_idx:]
                config_path = config_path[:-1] + '.json'
                project_root = os.path.abspath(os.path.dirname(__file__))
                cfg = Sweeper(project_root, config_path).parse(param)
                l = ''
                for label_key in label_keys:
                    l += str(getattr(cfg, label_key)) + ' '
                print(l)

            returns = arrange_order(returns)
            aucs.append(returns.mean(axis=0).sum())
            # print('max: ', np.max(returns))
            print('dimensions: ', returns.shape)
            draw_curve(returns, axs[idx], l, cmap(list(all_params.keys()).index(param), len(list(all_params.keys()))))

        axs[idx].set_title(label)
        axs[idx].legend()

    #     print('-------------------------------------------------')
    # for idx, label in enumerate(labels):
    # param = np.argmax(aucs)
    # print('argmax: ', param)
    # for arg_idx in np.argsort(aucs)[-10:][::-1]:
    # root_idx = len('data/output/') + all_paths_dict[idx]['control'].rindex('data/output/')
    # config_path = 'experiment/config/' + all_paths_dict[idx]['control'][root_idx:]
    # config_path = config_path[:-1]  + '.json'
    # project_root = os.path.abspath(os.path.dirname(__file__))
    # print('id: ', arg_idx)
    # cfg = Sweeper(project_root, config_path).parse(arg_idx)
    # print('learning-rate: ', str(getattr(cfg, 'learning_rate')))
    # print('rep_config: ', str(getattr(cfg, 'rep_config')))
    # print('activation_config: ', str(getattr(cfg, 'activation_config')))
    # print('-------------------------------------------------')

    fig.suptitle(title)
    # plt.xlim(0, 30)
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    # plt.clf()


def learning_curve(all_paths_dict, title, total_param=None,
                   start_param=0, labels_map=None):
    labels = [i["label"] for i in all_paths_dict]
    # control = load_return(all_paths_dict, total_param, start_param)
    control = load_return(all_paths_dict, total_param)  # , start_param)
    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(6 * len(labels), 4))

    if len(labels) == 1:
        axs = [axs]

    for label in labels:
        print(label)
        returns = arrange_order(control[label])
        draw_curve(returns, plt, label, violin_colors[label], style=curve_styles[label])

    fig.suptitle(title)
    # plt.xlim(0, 30)
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}.png".format(title))
    # plt.show()
    plt.close()
    # plt.clf()


def maze_multigoals():
    # learning_curve(maze_source_sweep, "maze source")
    learning_curve(maze_target_same_sweep, "maze compare same")
    # learning_curve(maze_target_diff_sweep, "maze dissimilar")


if __name__ == '__main__':
    maze_multigoals()
