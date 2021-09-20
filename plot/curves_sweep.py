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


def pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels):
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
            print("{}, best param {}".format(label, best_param))
        all_goals_auc[goal] = rep_auc


    return all_goals_auc


def performance_change(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], smooth=1.0):
    labels = [i["label"] for i in all_paths_dict]
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)

    all_goals_auc = pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels)

    curves = {}
    for label in labels:
        ranked_auc = np.zeros(all_ranks.max() + 1) * np.inf
        ranked_goal = np.zeros(all_ranks.max() + 1) * np.inf
        for goal in goal_ids:
            rank = ranks[goal]
            # print(rank, goal, label, all_goals_auc[goal][label][0])
            ranked_auc[rank] = all_goals_auc[goal][label][0]
            ranked_goal[rank] = goal
        ranked_auc = exp_smooth(ranked_auc, smooth)
        curves[label] = ranked_auc

    plt.figure(figsize=(12, 9))
    for label in labels:
        plt.plot(curves[label], color=violin_colors[label], linestyle=curve_styles[label], label=label)

    xticks_pos = list(range(0, all_ranks.max()+1, 25))
    xticks_labels = list(range(0, all_ranks.max()+1, 25))
    plt.xticks(xticks_pos, xticks_labels, rotation=60)
    # plt.xticks(list(range(all_ranks.max()+1)), all_ranks, rotation=90)

    plt.legend()
    # plt.show()
    # plt.xlabel('goal index\nOrdered by the similarity')
    plt.xlabel('Goal Ranks')
    plt.ylabel('AUC')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}.png".format(title))
    # plt.show()
    plt.close()


def correlation_change(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], smooth=1.0):
    labels = [i["label"] for i in all_paths_dict]
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)

    all_goals_auc = pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels)

    formated_path = {}
    for goal in goal_ids:
        g_path = copy.deepcopy(all_paths_dict)
        for i in range(len(all_paths_dict)):
            label = g_path[i]["label"]
            best_param_folder = all_goals_auc[goal][label][1]
            best = int(best_param_folder.split("_")[0])
            g_path[i]["control"] = [g_path[i]["control"].format(goal), best]
        formated_path[goal] = g_path

    property_keys = {"lipschitz": "complexity reduction", "distance": "dynamics awareness", "ortho": "orthogonality", "interf":"noninterference", "diversity":"diversity", "sparsity":"sparsity"}

    all_goals_cor = {}
    for pk in property_keys.keys():
        properties, _ = load_property([formated_path[goal]], property_key=pk, early_stopped=True)
        all_goals_cor[pk] = {}
        for goal in goal_ids:
            transf_perf, temp_labels = load_property([formated_path[goal]], property_key="return", early_stopped=True)
            cor = correlation_calc(temp_labels, transf_perf, properties, pk, perc=None, relationship=None)
            all_goals_cor[pk][goal] = cor

    curves = {}
    for pk in property_keys.keys():
        ranked_cor = np.zeros(all_ranks.max() + 1) * np.inf
        ranked_goal = np.zeros(all_ranks.max() + 1) * np.inf
        for goal in goal_ids:
            rank = ranks[goal]
            ranked_cor[rank] = all_goals_cor[pk][goal]
            ranked_goal[rank] = goal
        ranked_cor = exp_smooth(ranked_cor, smooth)
        curves[pk] = ranked_cor

    plt.figure(figsize=(12, 9))
    for pk in property_keys.keys():
        plt.plot(curves[pk], label=property_keys[pk])

    xticks_pos = list(range(0, all_ranks.max()+1, 25))
    xticks_labels = list(range(0, all_ranks.max()+1, 25))
    plt.xticks(xticks_pos, xticks_labels, rotation=60)
    # plt.xticks(list(range(all_ranks.max()+1))[1:], all_ranks, rotation=90)

    plt.legend()
    # plt.show()
    # plt.xlabel('goal index\nOrdered by the similarity')
    plt.xlabel('Goal Ranks')
    plt.ylabel('Correlation')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}.png".format(title))
    # plt.show()
    plt.close()

def transfer_curve_choose(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[]):
    labels = [i["label"] for i in all_paths_dict]

    all_goals_res = {}
    for goal in goal_ids:
        print("Loading auc from goal id {}".format(goal))
        single_goal_paths_dict = copy.deepcopy(all_paths_dict)
        for i in range(len(single_goal_paths_dict)):
            single_goal_paths_dict[i]["control"] = single_goal_paths_dict[i]["control"].format(goal)
        control = load_return(single_goal_paths_dict, total_param, search_lr=True)  # , start_param)

        rep_return = {}
        for idx, label in enumerate(labels):
            print("\n", idx, label)
            all_params = control[label]
            auc_rec = []
            param_rec = []
            returns_rec = []
            for param, returns in all_params.items():
                returns = arrange_order(returns)
                mu, ste = get_avg(returns)
                if xlim != []:
                    mu, ste = mu[xlim[0]: xlim[1]], ste[xlim[0]: xlim[1]]
                    returns = returns[:, xlim[0]: xlim[1]]
                auc_rec.append(np.sum(mu))
                param_rec.append(param)
                returns_rec.append(returns)
            best_idx = np.argmax(auc_rec)
            best_param_folder = param_rec[best_idx].split("_")[0]
            best_param = param_rec[best_idx].split("_")[1]
            best_return = returns_rec[best_idx]
            rep_return[label] = [best_return, best_param_folder, best_param]
            print("{}, best param {}".format(label, best_param))
        all_goals_res[goal] = rep_return

    for goal in goal_ids:
        plt.figure()
        for label in labels:
            # print(rank, goal, label, all_goals_auc[goal][label][0])
            returns = all_goals_res[goal][label][0]
            draw_curve(returns, plt, label, violin_colors[label], style=curve_styles[label])
        plt.legend()
        # plt.show()
        plt.xlabel('Step (10^4)')
        plt.ylabel('Return per Ep')
        plt.savefig("plot/img/goal{}_rank{}_{}.png".format(goal, ranks[goal], title), dpi=300, bbox_inches='tight')
        print("Save in plot/img/{}.png".format(title))
        plt.close()

    # if targets is not None:
    #     temp = []
    #     for item in all_paths_dict:
    #         if item["label"] in targets:
    #             temp.append(item)
    #     all_paths_dict = temp
    #
    # labels = [i["label"] for i in all_paths_dict]
    # # control = load_return(all_paths_dict)
    # control = load_info(all_paths_dict, 0, "return")
    # print(control.keys())
    # plt.figure()
    # for label in labels:
    #     print(label)
    #     #print(control)
    #     returns = arrange_order(control[label])
    #     draw_curve(returns, plt, label, violin_colors[label], style=curve_styles[label])
    #
    # plt.title(title)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # if data_label:
    #     plt.legend()
    # if xlim is not None:
    #     plt.xlim(xlim[0], xlim[1])
    # if ylim is not None:
    #     plt.ylim(ylim[0], ylim[1])
    #
    # # plt.xlabel('step ($10^4$)')
    # plt.ylabel('return')
    # plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # plt.clf()

def mountain_car():
    learning_curve(mc_learn_sweep, "mountain car learning sweep")

def simple_maze():
    # print("\nRep learning")
    # learning_curve(gh_original_sweep_v13, "maze rep sweep result ")

    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1
    goal_ids = [106,
                107, 108, 109, 110, 111, 118, 119, 120, 121, 122, 123, 128, 129, 130,
                138, 139, 140, 141, 142, 143, 144, 152, 153, 154, 155, 156, 157, 158, 166, 167, 168, 169, 170, 171, 172,
                137, 151, 165, 127, 117, 105, 99, 98, 104, 116, 126, 136, 150, 164, 86, 85, 84, 87, 83, 88, 89, 97, 90, 103, 115,
                91, 92, 93, 82, 96, 102, 114, 81, 80, 71, 95, 70, 69, 68, 101, 67, 66, 113, 62, 65, 125, 61, 132, 47, 133,
                79, 48, 72, 94, 52, 146, 73, 49, 33, 100, 134, 34, 74, 50, 147, 35, 112, 75, 135, 160, 36, 148, 76, 161, 63, 77, 124, 149, 78, 162, 163, 131, 53, 145, 159, 54, 55, 56, 57, 58, 59, 64, 51, 38,
                60, 39, 40, 46, 41, 42, 43, 44, 45, 32, 31, 37, 30, 22, 21, 23, 20, 24, 19, 25, 18, 26, 17, 16, 27, 15, 28, 29, 7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0,
    ]
    # performance_change(gh_transfer_samelr_v13, goal_ids, ranks, "maze transfer change (same lr)", xlim=[0, 11])
    performance_change(gh_transfer_sweep_v13, goal_ids, ranks, "maze transfer change (smooth 0.2)", xlim=[0, 11], smooth=0.2)
    # correlation_change(gh_transfer_sweep_v13, goal_ids, ranks, "maze correlation change (smooth 0.2)", xlim=[0, 11], smooth=0.2)
    # correlation_change(gh_transfer_sweep_v13, goal_ids, ranks, "maze correlation change", xlim=[0, 11], smooth=1)
    # goal_ids = [107, 108,
    #             90, 103,
    # ]
    # transfer_curve_choose(gh_transfer_sweep_v13, goal_ids, ranks, "transfer", total_param=None, xlim=[0, 11])
    # print(len(goal_ids))

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
