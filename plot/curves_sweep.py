import os
import sys
import copy
import numpy as np
import itertools
import pickle

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
    # control = load_return(all_paths_dict, total_param, search_lr=True, key="diversity")#, start_param)

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
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}.pdf".format(title))
    # plt.show()
    plt.close()
    # plt.clf()

def performance_change(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], smooth=1.0, top_runs=[0, 1.0],
                       xy_label=True, data_label=True, linewidth=1, figsize=(8, 6), save_data_name=None):
    labels = [i["label"] for i in all_paths_dict]
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)
    inv_ranks = {v: k for k, v in ranks.items()}

    all_ranks.sort()
    #print(all_ranks)
    all_goals_auc = pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels, top_runs=top_runs)
    curves = {}
    for label in labels:
        #ranked_auc = np.zeros(all_ranks.max() + 1) * np.inf
        #ranked_goal = np.zeros(all_ranks.max() + 1) * np.inf
        ranked_auc = np.zeros_like(all_ranks).astype(np.float)
        ranked_goal = np.zeros_like(all_ranks).astype(np.float)
        for idx, rank in enumerate(all_ranks):
            #rank = ranks[goal]
            goal = inv_ranks[rank]
            # print(rank, goal, label, all_goals_auc[goal][label][0])
            ranked_auc[idx] = all_goals_auc[goal][label][0]
            ranked_goal[idx] = rank
            #ranked_auc[rank] = all_goals_auc[goal][label][0]
            #ranked_goal[rank] = goal
        #print(ranked_goal, ranked_auc)
        print(label, ': ', np.sum(ranked_auc))
        ranked_auc = exp_smooth(ranked_auc, smooth)
        curves[label] = ranked_auc
    import pickle
    if save_data_name is not None:
        with open(save_data_name, 'wb') as f:
            pickle.dump(curves, f)
    print(curves)
    plt.figure(figsize=figsize)
    print(all_ranks)
    i=0
    for label in labels:
        # draw_curve(curves[label], plt, label, violin_colors[label], style=curve_styles[label], linewidth=linewidth)
        plt.plot(all_ranks, curves[label], color=c_contrast[i], label=label, linewidth=linewidth)
        i +=1

    xticks_pos = list(range(0, all_ranks.max()+1, 25))
    xticks_labels = list(range(0, all_ranks.max()+1, 25))
    plt.xticks(xticks_pos, xticks_labels, rotation=60, fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if data_label:
        plt.legend()
    # plt.show()
    # plt.xlabel('goal index\nOrdered by the similarity')
    if xy_label:
        plt.xlabel('Goal Ranks')
        plt.ylabel('AUC')
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}".format(title))
    # plt.show()
    plt.close()


def correlation_change(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], ylim=[], smooth=1.0, classify=[None, 50,50], label=True,
                       property_keys = {"lipschitz": "Complexity Reduction", "distance": "Dynamics Awareness", "ortho": "Orthogonality", "interf":"Noninterference", "diversity":"Diversity", "sparsity":"Sparsity"},
                       property_perc=None, plot_origin=True):
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)

    all_goals_cor, property_keys = correlation_load(all_paths_dict, goal_ids,
                                                    total_param=total_param, xlim=xlim, property_keys=property_keys, property_perc=property_perc)

    unsmooth_curves = {}
    curves = {}
    for pk in property_keys.keys():
        ranked_cor = np.zeros(all_ranks.max() + 1) * np.inf
        ranked_goal = np.zeros(all_ranks.max() + 1) * np.inf
        for goal in goal_ids:
            rank = ranks[goal]
            ranked_cor[rank] = all_goals_cor[pk][goal]
            ranked_goal[rank] = goal
        unsmooth_curves[pk] = copy.deepcopy(ranked_cor)
        ranked_cor = exp_smooth(ranked_cor, smooth)
        curves[pk] = ranked_cor

    plt.figure(figsize=(8, 6))
    pk_all = list(property_keys.keys())
    plt.plot([0]*len(curves[pk_all[0]]), "--", color="black")
    # tpk = len(pk_all)
    for pk in pk_all:
        plt.plot(curves[pk], label=property_keys[pk], linewidth=2, color=violin_colors[property_keys[pk]])
        if plot_origin:
            plt.plot(unsmooth_curves[pk], alpha=0.4, color=violin_colors[property_keys[pk]])

    xticks_pos = list(range(0, all_ranks.max()+1, 25))
    xticks_labels = list(range(0, all_ranks.max()+1, 25))
    plt.xticks(xticks_pos, xticks_labels, rotation=60)
    # plt.xticks(list(range(all_ranks.max()+1))[1:], all_ranks, rotation=90)

    if label:
        plt.legend()
        plt.xlabel('Goal Ranks')
        plt.ylabel('Correlation')
    if ylim != []:
        plt.ylim(ylim[0], ylim[1])
    xticks_pos = list(range(0, all_ranks.max() + 1, 25))
    xticks_labels = list(range(0, all_ranks.max() + 1, 25))
    plt.xticks(xticks_pos, xticks_labels, rotation=60, fontsize=14)
    yticks_pos = [-0.6, -0.4, 0.0, 0.4, 0.6]
    yticks_labels = yticks_pos
    plt.yticks(yticks_pos, yticks_labels, fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}".format(title))
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


    targets = [
       # "ReLU",
        #"ReLU+Control5g", "ReLU+SF",
  #      "ReLU+SF",  #"ReLU+ATC", "ReLU(L)+ATC", "FTA+ATC", 
        #"ReLU+Control5g",
 #       "FTA eta=0.8",
        #"ReLU(L)", "ReLU+ATC", 
 #       "FTA+SF", 
       # "FTA+SF+Aug",
   #   "ReLU+ATC+NoAug",
   #     "ReLU+Aug",
     #   "ReLU+Ortho",
     #   "ReLU(L)+Ortho",
        #"FTA+Control5g", 
   #     "FTA+ATC", 
    #    "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",

    #    "Scratch", "Random", "Input",
        "FTA+SF+Aug",
        "FTA+SF+AuxAug",
        "FTA+SF+VFAug",
    
    ]

    learning_curve(label_filter(targets, nonlinear_maze_online_dynamic), "mountain car learning sweep")

def simple_maze():
    # print("\nRep learning")
    # targets = ["ReLU+divConstr w0.01", "ReLU+divConstr w0.001", "ReLU+divConstr w0.0001", "ReLU+divConstr w0.00001"]
    # learning_curve(label_filter(targets, gh_original_sweep_v13), "linear/maze rep sweep result")
    # learning_curve(gh_original_sweep_v13, "linear/maze rep sweep")
    # learning_curve(gh_nonlinear_original_sweep_v13, "nonlinear/maze rep sweep")
    # learning_curve(gh_nonlinear_original_sweep_v13_largeReLU, "nonlinear/maze rep sweep largerelu")

    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1

    targets = [
        "ReLU",
        "ReLU+Control1g", "ReLU+Control5g", "ReLU+ATC",  "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
   #     "ReLU(L)",
   #     "ReLU(L)+Control1g", "ReLU(L)+Control5g", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF",
   #     "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
   #     "FTA+Control1g", "FTA+Control5g",  "FTA+ATC", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
   #     "Scratch", "Random", 
   #     "Scratch(L)", "Random(L)", 
   #     "Input",
    ]
    goal_ids = [106,
                107, 108, 118, 119, 109, 120, 121, 128, 110, 111, 122, 123, 129, 130, 142, 143, 144, 141, 140, 139, 138, 156, 157, 158, 155, 170, 171, 172, 169, 154, 168, 153, 167, 152, 166, 137, 151,
                165, 127, 117, 105, 99, 136, 126, 150, 164, 116, 104, 86, 98, 85, 84, 87, 83, 88, 89, 97, 90, 103, 115, 91, 92, 93, 82, 96, 102, 114, 81, 80, 71, 95, 70, 69, 68, 101, 67, 66, 113, 62,
                65, 125, 61, 132, 47, 133, 79, 48, 72, 94, 52, 146, 73, 49, 33, 100, 134, 34, 74, 50, 147, 35, 112, 75, 135, 160, 36, 148, 76, 161, 63, 77, 124, 149, 78, 162, 163, 131, 53, 145, 159,
                54, 55, 56, 57, 58, 59, 64, 51, 38, 60, 39, 40, 46, 41, 42, 43, 44, 45, 32, 31, 37, 30, 22, 21, 23, 20, 24, 19, 25, 18, 26, 17, 16, 27, 15, 28, 29, 7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2,
                13, 1, 14, 0,
                ]


    # return
    # performance_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "linear/maze transfer auc (no smooth)", xlim=[0, 11], smooth=1,
    #                    xy_label=False, data_label=False, linewidth=1, figsize=(8, 6))
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer auc (no smooth)",
    #                    xlim=[0, 11], smooth=1,
    #                    xy_label=False, data_label=False, linewidth=1, figsize=(8, 6))
    # performance_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "linear/maze transfer auc (smooth 0.1)", xlim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=1, figsize=(8, 6))
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer auc (smooth 0.1)",
    #                    xlim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=1, figsize=(8, 6))
    # draw_label(targets, "auc all label", ncol=5)

    targets = [
    #    "ReLU",  "FTA+SF",
     #  "ReLU+ATC", "FTA+ATC",
        #"ReLU+Control5g", "ReLU+SF",
  #      "ReLU+SF",  #"ReLU+ATC", "ReLU(L)+ATC", "FTA+ATC", 
        #"ReLU+Control5g",
       # "FTA eta=0.8",
        # "ReLU(L)", "ReLU+ATC", 
        #"FTA+SF", 

        # "FTA+SF+Aug",
        # "FTA+SF+AuxAug",
        # "FTA+SF+VFAug",
        #  "ReLU+Laplacian",
        # "ReLU+Diversity",
    #     "ReLU+Laplacian",
    #    "ReLU+Laplacian (rop)",
        #  "ReLU+Ortho",
    #     "ReLU+Ortho (prop)"
   #   "ReLU+ATC+NoAug",
    #    "ReLU+Aug",
    #    "ReLU+Ortho",
   #     "ReLU(L)+Ortho",
        # "FTA+Control5g", 
        # "FTA+ATC", 
#   "ReLU+Aug", 
#   "FTA eta=0.2", 
#   "ReLU(L)",
  "VF5",
  "No Aux"
  #"FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",

#   "ReLU+Laplacian (standard)",
 #  "ReLU(L)+Laplacian (standard)",
 #  "ReLU+Laplacian (non-standard)",
    #    "Scratch", "Random", "Input",
        # "ReLU+Laplacian", "ReLU+DA", 
    #    "ReLU+CR",
    #    "ReLU(L)+CR",
    #    "FTA+CR",
    # "ReLU+VirtualVF5",
    # "FTA+SF",
    #     "ReLU+DA+O",
    #     "ReLU(L)+DA+O",
    #     "FTA+DA+O",


       #"ReLU+CompOrtho",  
    #    "ReLU+Laplacian", 
    #    "ReLU+Laplacian (0)",
    #    "ReLU+Laplacian (0.1)", 
    #    "ReLU+Laplacian (0.25)", 
    #    "ReLU+Laplacian (0.5)", 
    #    "ReLU+Laplacian (0.75)", 

        # "ReLU+DO", 
        # "ReLU(L)+DO", 
        # "FTA+DO", 

        # "ReLU+Laplacian",
        # "ReLU(L)+Laplacian",
        # "FTA+Laplacian", 

    #     "ReLU+DO": m_default[1],

    # "ReLU+CompOrtho": m_default[1],
    
    # "ReLU+CR",
    # "ReLU(L)+CR",
    # "FTA+CR",


    # "ReLU+DynaOrtho": c_default[8],
    # "ReLU(L)+DO":m_default[1],
    # "FTA+DO":m_default[0]
        # "Scratch",
        # "Scratch(L)"

    ]

    
    # performance_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "linear/maze transfer chosen (smooth 0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), emphasize=emphasize)
    emphasize = [
        "ReLU",
        #"ReLU+Control5g", "ReLU+SF",
        "ReLU+SF",  #"ReLU+ATC", "ReLU(L)+ATC", "FTA+ATC", 
        #"ReLU(L)+Control5g", "ReLU(L)", "ReLU+ATC", 
        "FTA+SF", 
   #     "FTA+SF+Aug",
        "ReLU+ATC+NoAug",
        "ReLU+ATC",
        "ReLU+Aug",
        "ReLU+Control5g", 
        "ReLU+Laplacian"

    #   "FTA+Control5g", 
        #"FTA+Control5g", "FTA+ATC", 
    #    "Scratch", "Random", "Input",
    ]

    emphasize = [
        "ReLU",
        #"ReLU+Control5g", "ReLU+SF",
        #"ReLU+SF",  #"ReLU+ATC", "ReLU(L)+ATC", "FTA+ATC", 
        #"ReLU(L)+Control5g", "ReLU(L)", "ReLU+ATC", 
        "FTA+SF", 
   #     "FTA+SF+Aug",
        # "ReLU+ATC+NoAug",
        # "ReLU+ATC",
        "ReLU+Laplacian",
        "ReLU+Ortho"
        #"ReLU+Aug",
        #"ReLU+Control5g", 

    #   "FTA+Control5g", 
        #"FTA+Control5g", "FTA+ATC", 
    #    "Scratch", "Random", "Input",
    ]

    emphasize = targets
    performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer chosen (smooth 0.1)", smooth=0.1,
                       xy_label=False, data_label=True, linewidth=3, figsize=(8, 6))
    # draw_label(targets, "auc chosen label", ncol=4,
    #            emphasize=["ReLU+Control5g", "ReLU+SF", "ReLU+ATC",  "ReLU(L)+Control5g", "ReLU(L)", "FTA+SF", "FTA+Control5g", "FTA+NAS", "FTA+ATC", "Scratch", "Random", "Input"])


    targets = [
        "ReLU",
        "ReLU+Control1g", "ReLU+Control5g", "ReLU+ATC", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
   #     "ReLU(L)", "FTA+Control5g"
    #    "ReLU(L)+Control1g", "ReLU(L)+Control5g", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF",
    #    "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
    #    "FTA+Control1g", "FTA+Control5g", "FTA+ATC", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", 
    ]
    # correlation_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks,
    #                    "linear/maze correlation change (no smooth)", xlim=[0, 11], smooth=1, label=False)
    #correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                   "nonlinear/maze correlation change (no smooth)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=1, label=False, plot_origin=False)
    #correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                   "nonlinear/maze correlation change (smooth)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=.1, label=False, plot_origin=False)    
    # correlation_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks,
    #                    "linear/maze correlation change (smooth 0.1)", xlim=[0, 11], smooth=0.1, label=False)
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (smooth 0.1)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False)

    #correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low otho)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_perc={"ortho": [0, 70]}
    #                    )
    #correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                   "nonlinear/maze correlation change (high otho)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                   property_perc={"ortho": [70, 100]}
    #                   )
    #correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change 0-55 (low otho)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_perc={"ortho": [0, 55]}
    #                    )
    #correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                   "nonlinear/maze correlation change 55-10 (high otho)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                   property_perc={"ortho": [55, 100]}
    #                   )
    #draw_label(["Complexity Reduction", "Dynamics Awareness", "Noninterference", "Diversity", "Sparsity"], "ortho slice label", ncol=3)

    #performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer high ortho (smooth 0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), property_perc={"ortho": [55, 100]})
    #performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer low ortho (smooth 0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                   xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), property_perc={"ortho": [0, 55]})
    # draw_label(targets, "auc all label", ncol=4)

    targets = [
        "ReLU",
        "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
    ]
    # correlation_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks,
    #                    "linear/maze correlation change (smooth 0.1, relu)", xlim=[0, 11], smooth=0.1, label=False)
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (smooth 0.1, relu)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False)
    # targets = [
    #     "ReLU(L)",
    #     "ReLU(L)+Control1g", "ReLU(L)+Control5g", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF",
    # ]
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (smooth 0.1, relu(l))", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False)
    targets = [
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
    ]
    # correlation_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "linear/maze correlation change (smooth 0.1, fta)", xlim=[0, 11], smooth=0.1)
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13), goal_ids, ranks, "nonlinear/maze correlation change (smooth 0.1, fta)", xlim=[0, 11], smooth=0.1, label=False, plot_origin=False)

    # # goal_ids = [106, 107, 109, 155, 98, 147]
    # goal_ids = [106, 155, 98, 58, 18]  # 0 25 50 100 125 150
    # targets = [
    #     "ReLU",
    #     "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
    #     "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
    #     "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
    # ]
    # # transfer_curve_choose(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "linear/transfer", total_param=None, xlim=[0, 11])
    # # transfer_curve_choose(label_filter(targets, gh_nonlinear_transfer_sweep_v13), goal_ids, ranks, "nonlinear/transfer", total_param=None, xlim=[0, 11])


def maze_multigoals():
    # learning_curve(ghmg_original_sweep_v13, "multigoal rep sweep result")

    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1
    goal_ids = [106,
                107, 108, 118, 119, 109, 120, 121, 128, 110, 111, 122, 123, 129, 130, 142, 143, 144, 141, 140, 139, 138, 156, 157, 158, 155, 170, 171, 172, 169, 154, 168, 153, 167, 152, 166, 137, 151,
                165, 127, 117, 105, 99, 136, 126, 150, 164, 116, 104, 86, 98, 85, 84, 87, 83, 88, 89, 97, 90, 103, 115, 91, 92, 93, 82, 96, 102, 114, 81, 80, 71, 95, 70, 69, 68, 101, 67, 66, 113, 62,
                65, 125, 61, 132, 47, 133, 79, 48, 72, 94, 52, 146, 73, 49, 33, 100, 134, 34, 74, 50, 147, 35, 112, 75, 135, 160, 36, 148, 76, 161, 63, 77, 124, 149, 78, 162, 163, 131, 53, 145, 159,
                54, 55, 56, 57, 58, 59, 64, 51, 38, 60, 39, 40, 46, 41, 42, 43, 44, 45, 32, 31, 37, 30, 22, 21, 23, 20, 24, 19, 25, 18, 26, 17, 16, 27, 15, 28, 29, 7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2,
                13, 1, 14, 0,
                ]
    # performance_change(ghmg_transfer_sweep_v13, goal_ids, ranks, "multigoal transfer change (smooth 0.2)", xlim=[0, 11], smooth=0.2)
    # performance_change(ghmg_transfer_last_sweep_v13, goal_ids, ranks, "multigoal lastrep transfer change (smooth 0.2)", xlim=[0, 11], smooth=0.2)

    performance_change(ghmg_transfer_sweep_v13, goal_ids, ranks, "multigoal transfer change", xlim=[0, 11], smooth=1)
    performance_change(ghmg_transfer_last_sweep_v13, goal_ids, ranks, "multigoal lastrep transfer change", xlim=[0, 11], smooth=1)

if __name__ == '__main__':
    # mountain_car()
    #mountain_car()
    simple_maze()
    # maze_multigoals()

