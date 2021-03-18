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
        start_param=0, label_keys = None, config_paths=None):

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
                assert config_paths is not None
                project_root = os.path.abspath(os.path.dirname(__file__))
                print(project_root)
                cfg = Sweeper(project_root, config_paths[idx]).parse(param)
                l = ''
                for label_key in label_keys[idx]:
                    l += str(getattr(cfg, label_key)) + ' '
            
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
    control = load_return(all_paths_dict, total_param)
    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(30, 4))

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
    # learning_curve(gh_learn_sweep, "maze learning sweep")
    learning_curve(gh_online_sweep, "maze online property sweep")

    # # print("\nControl")
    # learning_curve(gh_same_sweep, "maze same sweep")
    # learning_curve(gh_similar_sweep, "maze similar sweep")
    # learning_curve(gh_diff_sweep, "maze different (fix) sweep")
    # learning_curve(gh_diff_tune_sweep, "maze different (fine tune) sweep")

    # learning_curve(gh_etaStudy_diff_fix_sweep, "maze different (fix) eta study")
    # learning_curve(gh_etaStudy_diff_tune_sweep, "maze different (fine tune) eta study")

def picky_eater():

    compare_learning_curve(crgb_online, "maze online property")
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
    # simple_maze()
    picky_eater()
