import os
import sys
import copy
import numpy as np

import matplotlib.pyplot as plt

sys.path.insert(0, '..')
print(sys.path)
from plot.plot_paths import *
from plot.plot_utils import *
from experiment.sweeper import Sweeper

print("Change dir to", os.getcwd())

os.chdir("..")
print(os.getcwd())
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
    
    fig.suptitle(label)
    # plt.xlim(0, 30)
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(label), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    # plt.clf()


def learning_curve(all_paths_dict, title, total_param=None,
        start_param=0, labels_map=None, xlim=[]):

    labels = [i["label"] for i in all_paths_dict]
    # control = load_return(all_paths_dict, total_param, start_param)
    control = load_return(all_paths_dict, total_param, search_lr=True)#, start_param)
    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(6*len(labels), 4))

    if len(labels) == 1:
        axs = [axs]

    for idx, label in enumerate(labels):
        print("\n", idx, label)
        all_params = control[label]
        auc_rec = []
        param_rec = []
        for param, returns in all_params.items():
            returns = arrange_order(returns)
            mu = draw_curve(returns, axs[idx], param.split("_")[1], cmap(list(all_params.keys()).index(param), len(list(all_params.keys()))))
            auc_rec.append(np.sum(mu))
            param_rec.append(param)
        print("best index {} (param {})".format(param_rec[np.argmax(auc_rec)].split("_")[0], param_rec[np.argmax(auc_rec)].split("_")[1]))
        if xlim != []:
            axs[idx].set_xlim(xlim[0], xlim[1])
        axs[idx].set_title(label)
        axs[idx].legend()

    fig.suptitle(title)
    plt.xlabel('step ($10^4$)')
    plt.ylabel('return')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}.png".format(title))
    # plt.show()
    plt.close()
    # plt.clf()



def performance_change(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[]):
    labels = [i["label"] for i in all_paths_dict]
    # control = load_return(all_paths_dict, total_param, start_param)

    all_goals_auc = {}
    for goal in goal_ids:
        print("Loading auc from goal id {}".format(goal))
        single_goal_paths_dict = copy.deepcopy(all_paths_dict)
        for i in range(len(single_goal_paths_dict)):
            single_goal_paths_dict[i]["control"] = single_goal_paths_dict[i]["control"].format(goal)
        control = load_return(single_goal_paths_dict, total_param, search_lr=True)  # , start_param)

        rep_auc = {}
        for idx, label in enumerate(labels):
            print("\n", idx, label)
            all_params = control[label]
            auc_rec = []
            param_rec = []
            curve_rec = []
            for param, returns in all_params.items():
                returns = arrange_order(returns)
                mu, ste = get_avg(returns)
                if xlim != []:
                    mu, ste = mu[xlim[0]: xlim[1]], ste[xlim[0]: xlim[1]]
                auc_rec.append(np.sum(mu))
                param_rec.append(param)
                curve_rec.append([mu, ste])
            best_idx = np.argmax(auc_rec)
            best_param_folder = param_rec[best_idx].split("_")[0]
            best_param = param_rec[best_idx].split("_")[1]
            best_auc = auc_rec[best_idx]
            rep_auc[label] = [best_auc, best_param_folder, best_param]
        all_goals_auc[goal] = rep_auc

    curves = {}
    for label in labels:
        ranked_auc = np.zeros(len(goal_ids))
        ranked_ste = np.zeros(len(goal_ids))
        for goal in goal_ids:
            rank = ranks[goal]
            print(rank, goal, label, all_goals_auc[goal][label][0])
            ranked_auc[rank] = all_goals_auc[goal][label][0]
            ranked_ste[rank] = all_goals_auc[goal][label][0]
        curves[label] = ranked_auc

    plt.figure()
    for label in labels:
        plt.plot(curves[label], color=violin_colors[label], linestyle=curve_styles[label], label=label)
    plt.legend()
    plt.show()

def mountain_car():
    learning_curve(mc_learn_sweep, "mountain car learning sweep")

def simple_maze():
    # print("\nRep learning")
    # learning_curve(gh_original_sweep_v13, "maze rep sweep result ")

    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1
    goal_ids = [106, 107, 108, 109, 110, 111, 118, 119, 120, 121, 122, 123, 128, 129, 130,
                138, 139, 140, 141, 142, 143, 144, 152, 153, 154, 155, 156, 157, 158, 166, 167, 168, 169, 170, 171, 172]
    performance_change(gh_transfer_samelr_v13, goal_ids, ranks, "maze transfer change", xlim=[0, 11])

    # # print("\nControl")
    # learning_curve(gh_same_early_sweep, "maze same sweep")
    # learning_curve(gh_similar_early_sweep, "maze similar sweep")
    # learning_curve(gh_diff_early_sweep, "maze different (fix) sweep(temp)")
    # learning_curve(gh_diff_tune_early_sweep, "maze different (fine tune) sweep")

def picky_eater():
    #titles = ["ReLU", "ReLU+Control", "ReLU+XY", "ReLU+Color", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF" ,"FTA", "FTA+Control", "FTA+Decoder", "FTA+XY", "FTA+Color", "FTA+NAS", "FTA+Reward", "FTA+SF"]
    #titles = ["ReLU", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF" ,"FTA", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF"]
#    titles = ["FTA+Decoder", "ReLU+Control", "FTA", "FTA+Control"]

    for i, crgb_sweep in enumerate(maze_target_diff_sweep_v12):
        compare_learning_curve([crgb_sweep], None, label_keys=None)
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
    learning_curve(pe_transfer_best_dissimilar, "pe diff fix result")
    # learning_curve(perand_trans_sweep_temp, "perandc diff fix avg v6")

def maze_multigoals():
    # learning_curve(maze_source_sweep, "maze source")
    # learning_curve(maze_checkpoint50000_same_sweep_v12, "maze checkpoint50000 same")
    # learning_curve(maze_checkpoint50000_dissimilar_sweep_v12, "maze checkpoint50000 dissimilar")
    # learning_curve(maze_checkpoint100000_same_sweep_v12, "maze checkpoint100000 same")
    # learning_curve(maze_checkpoint100000_dissimilar_sweep_v12, "maze checkpoint100000 dissimilar")
    # learning_curve(maze_checkpoint150000_same_sweep_v12, "maze checkpoint150000 same")
    # learning_curve(maze_checkpoint150000_dissimilar_sweep_v12, "maze checkpoint150000 dissimilar")
    # learning_curve(maze_checkpoint200000_same_sweep_v12, "maze checkpoint200000 same")
    # learning_curve(maze_checkpoint200000_dissimilar_sweep_v12, "maze checkpoint200000 dissimilar")
    # learning_curve(maze_checkpoint250000_same_sweep_v12, "maze checkpoint250000 same")
    # learning_curve(maze_checkpoint250000_dissimilar_sweep_v12, "maze checkpoint250000 dissimilar")
    # learning_curve(maze_checkpoint300000_same_sweep_v12, "maze checkpoint300000 same")
    # learning_curve(maze_checkpoint300000_dissimilar_sweep_v12, "maze checkpoint300000 dissimilar")

    # learning_curve(mazesimple_notarget_same_sweep_v12, "mazesimple dqn notarget same")
    # learning_curve(mazesimple_qlearning_same_sweep_v12, "mazesimple qlearning same")
    # learning_curve(maze_multigoal_notarget_same_sweep_v12, "maze_multigoal dqn notarget same")
    learning_curve(maze_multigoal_notarget_diff_sweep_v12, "maze_multigoal dqn notarget dissimilar", xlim=[0, 30])
    # learning_curve(maze_multigoal_qlearning_same_sweep_v12, "maze_multigoal qlearning same")

if __name__ == '__main__':
    # mountain_car()
    simple_maze()
    # picky_eater()
    # pe_temp()
    # maze_multigoals()
