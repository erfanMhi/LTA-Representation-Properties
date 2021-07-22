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
        start_param=0, label_keys = None, key='return'):

    labels = [i["label"] for i in all_paths_dict]
    control = load_return(all_paths_dict, total_param)#, start_param)
    #control = load_info(all_paths_dict, total_param, key)#, start_param)
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
                config_path = config_path[:-1]  + '.json'
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
            draw_curve(returns, axs[idx], l, cmap(param, len(list(all_params.keys()))))
        
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
    control = load_return(all_paths_dict, total_param)#, start_param)
    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(6*len(labels), 4))

    if len(labels) == 1:
        axs = [axs]
    
    for idx, label in enumerate(labels):
        print("\n", idx, label)
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
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}.png".format(title))
    # plt.show()
    plt.close()
    # plt.clf()

def mountain_car():
    learning_curve(mc_learn_sweep, "mountain car learning sweep")

def simple_maze():
    # print("\nRep learning")
    # learning_curve(gh_online_sweep, "maze rep sweep result ")

    # # print("\nControl")
    # learning_curve(gh_same_early_sweep, "maze same sweep")
    # learning_curve(gh_similar_early_sweep, "maze similar sweep")
    learning_curve(gh_diff_early_sweep, "maze different (fix) sweep(temp)")
    # learning_curve(gh_diff_tune_early_sweep, "maze different (fine tune) sweep")

def picky_eater():
#    titles = ["ReLU", "ReLU+Control", "ReLU+XY", "ReLU+Color", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF" ,"FTA", "FTA+Control", "FTA+Decoder", "FTA+XY", "FTA+Color", "FTA+NAS", "FTA+Reward", "FTA+SF"]
    titles = ["FTA+Decoder", "ReLU+Control", "FTA", "FTA+Control"]
    print('here')
    for i, crgb_sweep in enumerate(pe_t_sweep_v2):
        compare_learning_curve([crgb_sweep], titles[i], label_keys=None)
        # learning_curve([crgb_sweep], "maze online property")
    # for crgb_sweep in crgb_online_sweep_1_f:
    #     compare_learning_curve([crgb_sweep], "maze online property", label_keys=['learning_rate', ])
    #     # learning_curve([crgb_sweep], "maze online property")

    # learning_curve(crgb_online_best, "maze online property")
    #compare_learning_curve(pe_learn_sweep, "picky eater learning curve",
    #        label_keys = [['target_network_update_freq', 'learning_rate']], config_paths = ['experiment/config/test/picky_eater/online_property/dqn_lta/sweep.json'])
  
#    compare_learning_curve(dqn_learn_sweep, "picky eater learning curve",
    #        label_keys = [['target_network_update_freq', 'learning_rate']], config_paths = ['experiment/config/test/picky_eater/online_property/dqn/sweep.json'])
    # compare_learning_curve(dqn_lta_1_learn_sweep, "picky eater learning curve",
            # label_keys = [['target_network_update_freq', 'learning_rate']], config_paths = ['experiment/config/test/picky_eater/online_property/dqn_lta/sweep.json'])
    #compare_learning_curve(dqn_lta_learn_sweep, "picky eater learning curve",
    #        label_keys = [['target_network_update_freq', 'learning_rate']], config_paths = ['experiment/config/test/picky_eater/online_property/dqn_lta/sweep.json'])
    # compare_learning_curve(dqn_lta_1_learn_sweep, "picky eater learning curve (DQN+LTA+Without target)",
            # label_keys = [['target_network_update_freq', 'learning_rate']], config_paths = ['experiment/config/test/picky_eater/online_property/dqn_lta/sweep.json'])
    return
def pe_temp():
    # learning_curve(pe_sweep_temp, "pe rep result temp")
    # learning_curve(perand_sweep_temp, "perandc rep v6")
    learning_curve(pe_trans_sweep_temp, "pe diff fix result")
    # learning_curve(perand_trans_sweep_temp, "perandc diff fix avg v6")

def maze_multigoals():
    # learning_curve(maze_source_sweep, "maze source")
    learning_curve(maze_target_sweep, "maze dissimilar")

if __name__ == '__main__':
    # mountain_car()
    # simple_maze()
    # picky_eater()
    # pe_temp()
    maze_multigoals()
