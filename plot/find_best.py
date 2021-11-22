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


def get_best_run_and_params(all_paths_dict, title, total_param=None,
        start_param=0, label_keys = None, config_paths=None, last_evals_num=1):

    labels = [i["label"] for i in all_paths_dict]
    print('loading')
    control = load_return(all_paths_dict, total_param, start_param)

    best_param_label = 'None'
    best_param = -1
    best_param_val = float('-inf')
    best_label = -1
    best_run = -1
    print('here')
    for idx, label in enumerate(labels):
        all_params = control[label]
        print(idx, '/', len(labels))
        for param, returns in all_params.items():

            if label_keys is not None:
                assert config_paths is not None
                project_root = os.path.abspath(os.path.dirname(__file__))
                cfg = Sweeper(project_root, config_paths[idx]).parse(param)
                l = ''
                for label_key in label_keys[idx]:
                    l += str(getattr(cfg, label_key)) + ' '
            
            returns = arrange_order(returns)
            performance = returns[:, -last_evals_num:].sum(1).mean()
            if performance >= best_param_val:
                best_param_val = performance
                best_param_label = l
                best_param = param
                best_label = label
                best_run = returns[:, -last_evals_num:].sum(1).argmax()
    print('Best Param: ', best_param)
    print('Best Param Label: ', best_param_label)
    print('Best Label: ', best_label)
    print('Best Run: ', best_run)
    print('Best Performance: ', best_param_val)

def picky_eater():

    get_best_run_and_params(dqn_lta_learn_sweep, "picky eater learning curve",
            label_keys = [['target_network_update_freq', 'learning_rate']], config_paths = ['experiment/config/test/picky_eater/online_property/dqn_lta/sweep.json'])
 
    get_best_run_and_params(dqn_learn_sweep, "picky eater learning curve",
            label_keys = [['target_network_update_freq', 'learning_rate']], config_paths = ['experiment/config/test/picky_eater/online_property/dqn/sweep.json'])
    
def simple_maze():
#    get_best_run_and_params(dqn_cl_maze_sweep, "picky eater learning curve",
#            label_keys = [['ul_learning_rate', 'learning_rate', 'ul_batch_size', 'ul_anchor_hidden_sizes', 'ul_delta_T']], config_paths = ['experiment/config//test_cl/maze/nonlinear_vf/dqn_cl/sweep.json'])
    get_best_run_and_params(dqn_cl_maze_sweep, "picky eater learning curve",
            label_keys = [['ul_learning_rate', 'learning_rate']], config_paths = ['experiment/config//test_cl/gridhard/nonlinear_vf/online_property/dqn_fta_cl/sweep.json'])
 
if __name__ == '__main__':
    # mountain_car()
    simple_maze()
#    picky_eater()
