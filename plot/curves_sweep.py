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
        start_param=0, label_keys = None):

    labels = [i["label"] for i in all_paths_dict]
    control = load_return(all_paths_dict, total_param)#, start_param)
    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(30, 4))
    
    if len(labels) == 1:
        axs = [axs]
 
    for idx, label in enumerate(labels):
        print("------", label, "------")
        all_params = control[label]
        
        for param, returns in all_params.items():

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
            draw_curve(returns, axs[idx], l, cmap(param, len(list(all_params.keys()))))
        
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



def learning_curve(all_paths_dict, title, total_param=None,
        start_param=0, labels_map=None):

    labels = [i["label"] for i in all_paths_dict]
    # control = load_return(all_paths_dict, total_param, start_param)
    control = load_return(all_paths_dict, total_param)#, start_param)
    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(6*len(labels), 4))

    if len(labels) == 1:
        axs = [axs]
    
    for idx, label in enumerate(labels):
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
    learning_curve(gh_online_sweep, "maze online property sweep")

    # # print("\nControl")
    learning_curve(gh_same_early_sweep, "maze same sweep")
    learning_curve(gh_similar_early_sweep, "maze similar sweep")
    learning_curve(gh_diff_early_sweep, "maze different (fix) sweep")
    learning_curve(gh_diff_tune_early_sweep, "maze different (fine tune) sweep")

def picky_eater():
    for crgb_sweep in crgb_online_sweep_2:
        compare_learning_curve([crgb_sweep], "maze online property", label_keys=['learning_rate', ])
        # learning_curve([crgb_sweep], "maze online property")
    
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
    
   

if __name__ == '__main__':
    # mountain_car()
    picky_eater()
