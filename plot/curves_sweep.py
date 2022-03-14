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
    control = load_return(all_paths_dict, total_param, search_lr=True, path_key="online_measure")  # , start_param)
    # control = load_return(all_paths_dict, total_param, search_lr=True, key="diversity")#, start_param)

    fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(6 * len(labels), 4))

    if len(labels) == 1:
        axs = [axs]

    for idx, label in enumerate(labels):
        print("\n", idx, label)
        all_params = control[label]
        auc_rec = []
        param_rec = []
        for param, returns in all_params.items():
            returns = arrange_order(returns)
            mu = draw_curve(returns, axs[idx], param.split("_")[1],
                            cmap(list(all_params.keys()).index(param), len(list(all_params.keys()))))
            auc_rec.append(np.sum(mu))
            param_rec.append(param)
        print("best index {} (param {})".format(param_rec[np.argmax(auc_rec)].split("_")[0],
                                                param_rec[np.argmax(auc_rec)].split("_")[1]))
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


def performance_change(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], ylim=[], smooth=1.0,
                       top_runs=[0, 1.0],
                       xy_label=True, data_label=True, linewidth=1, figsize=(8, 6), emphasize=None,
                       property_perc=None, color_by_activation=False, in_plot_label=None):
    labels = [i["label"] for i in all_paths_dict]
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)

    all_goals_auc = pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels, top_runs=top_runs)

    if property_perc is not None:
        formated_path = {}
        for goal in goal_ids:
            g_path = copy.deepcopy(all_paths_dict)
            for i in range(len(all_paths_dict)):
                label = g_path[i]["label"]
                best_param_folder = all_goals_auc[goal][label][1]
                best = int(best_param_folder.split("_")[0])
                g_path[i]["control"] = [g_path[i]["control"].format(goal), best]
            formated_path[goal] = g_path

        allp_properties = {}
        for pk in property_perc.keys():
            properties, _ = load_property([formated_path[goal]], property_key=pk, early_stopped=True)
            allp_properties[pk] = copy.deepcopy(properties)
        allg_transf_perf = {}
        for goal in goal_ids:
            transf_perf, temp_lables = load_property([formated_path[goal]], property_key="return", early_stopped=True)
            allg_transf_perf[goal] = copy.deepcopy(transf_perf)
        allg_transf_perf, allp_properties = property_filter(allg_transf_perf, allp_properties, property_perc)
        filtered_merged_labels = allg_transf_perf[goal_ids[0]].keys()
        all_goals_auc = {}
        for goal in goal_ids:
            rep_auc = {}
            for label in filtered_merged_labels:
                best_auc = np.mean(
                    np.array([allg_transf_perf[goal][label][run] for run in allg_transf_perf[goal][label]]))
                rep_auc[label.split("_")[0]] = [best_auc]
            all_goals_auc[goal] = rep_auc
        filtered_labels = [k.split("_")[0] for k in allg_transf_perf[goal_ids[0]].keys()]
    else:
        filtered_labels = labels

    curves = {}
    for label in filtered_labels:
        ranked_auc = np.zeros(all_ranks.max() + 1) * np.inf
        ranked_goal = np.zeros(all_ranks.max() + 1) * np.inf
        for goal in goal_ids:
            rank = ranks[goal]
            # print(rank, goal, label, all_goals_auc[goal][label][0])
            ranked_auc[rank] = all_goals_auc[goal][label][0]
            ranked_goal[rank] = goal
        ranked_auc = exp_smooth(ranked_auc, smooth)
        curves[label] = ranked_auc

    # with open('temp.pkl', 'wb') as handle:
    #     pickle.dump(curves, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure(figsize=figsize)
    avg_decrease = {"FTA": [], "ReLU": [], "ReLU(L)": []}
    for label in filtered_labels:
        if emphasize and (label not in emphasize):
            # plt.plot(curves[label], color=violin_colors[label], linestyle=curve_styles[label], label=label, linewidth=1, alpha=0.3)
            plt.plot(curves[label], color=violin_colors["Other rep"], linestyle=curve_styles[label], label=label,
                     linewidth=1, alpha=0.3)
        elif color_by_activation:
            # if label == "ReLU+ATC":
            #     color = "purple"
            #     alpha = 1
            if label.split("+")[0] == "ReLU":
                color = "#e74c3c"
                alpha = 0.5
            elif label.split("+")[0] == "ReLU(L)":
                color = "#8e44ad"
                alpha = 0.5
            elif label.split("+")[0] == "FTA" or label.split(" ")[0] == "FTA":
                color = "#3498db"
                alpha = 0.5
            elif label in ["Scratch", "Scratch(FTA)", "Scratch(L)"]:
                color = "#34495e"  # violin_colors[label]
                alpha = 1
            else:
                color = "#bdc3c7"  # violin_colors[label]
                alpha = 0.5
            plt.plot(curves[label], color=color, alpha=alpha)
        else:
            plt.plot(curves[label], color=violin_colors[label], linestyle=curve_styles[label], label=label,
                     linewidth=linewidth, zorder=100)
        print("Label, first point {}, last point {}, decrease {}".format(label, curves[label][0], curves[label][-1],
                                                                         curves[label][-1] - curves[label][0]))
        if label.split("+")[0] in avg_decrease.keys():
            avg_decrease[label.split("+")[0]].append(curves[label][0] - curves[label][-1])
    print("\nAverage decreases:")
    for label in avg_decrease:
        print(label, np.array(avg_decrease[label]).mean())

    if ylim != []:
        plt.ylim(ylim[0], ylim[1])
    xticks_pos = list(range(0, all_ranks.max() + 1, 25))
    xticks_labels = list(range(0, all_ranks.max() + 1, 25))
    plt.xticks(xticks_pos, xticks_labels, rotation=60, fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if data_label:
        plt.legend()
    if in_plot_label:
        # plt.get_legend().remove()
        for label in in_plot_label:
            key = list(label.keys())[0]
            plt.plot([], color=label[key][0], linestyle=label[key][1], alpha=label[key][2], label=key)
        plt.legend(ncol=2, prop={'size': 20})
    # plt.show()
    # plt.xlabel('goal index\nOrdered by the similarity')
    if xy_label:
        plt.xlabel('Goal Ranks')
        plt.ylabel('AUC')
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}".format(title))
    # plt.show()
    plt.close()


def performance_change_by_percent_onegroup(all_paths_dict, goal_ids, ranks, total_param=None, xlim=[], ylim=[],
                                           smooth=1.0,
                                           baselines=[], group_label=None, x_shift=0, width=0.8):
    labels = [i["label"] for i in all_paths_dict]
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)

    all_goals_auc, all_goals_independent = pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels,
                                                           get_each_run=True)
    filtered_labels = labels
    for bs in baselines:
        filtered_labels.remove(bs)

    ranked_perc = np.zeros((len(baselines), all_ranks.max() + 1))
    ranked_goal = np.zeros(all_ranks.max() + 1)
    for goal in goal_ids:
        bases = np.zeros(len(baselines))
        labels = []
        for bidx, bl in enumerate(baselines):
            bases[bidx] = all_goals_auc[goal][bl][0]
            labels.append(bl)
        higher_count = np.zeros(len(baselines))
        total_count = 0
        for label in filtered_labels:
            runs = all_goals_independent[goal][label]
            aucs = runs.sum(axis=1)
            for bidx, bl in enumerate(baselines):
                hidx = np.where(aucs > bases[bidx])[0]
                higher_count[bidx] += len(hidx)
            total_count += len(aucs)
        rank = ranks[goal]
        ranked_perc[:, rank] = higher_count / total_count
        ranked_goal[rank] = goal
    return ranked_perc, all_ranks


def performance_change_vf_difference(targets, paths_dicts, goal_ids, ranks, title, total_param=None, xlim=[], ylim=[],
                                     smooth=1.0,
                                     xy_label=True, data_label=True, figsize=(8, 6), baselines=[]):
    plt.figure(figsize=figsize)
    width = 0.3
    ranked_perc_dic = {}
    for idx, paths_dict in enumerate(list(paths_dicts.keys())):
        ranked_perc, all_ranks = performance_change_by_percent_onegroup(label_filter(targets, paths_dicts[paths_dict]),
                                                                        goal_ids, ranks, total_param=total_param,
                                                                        xlim=xlim, ylim=ylim, smooth=smooth,
                                                                        baselines=baselines, group_label=paths_dict,
                                                                        x_shift=width * idx, width=width)
        # for bidx, bl in enumerate(baselines):
        #     # plt.plot(exp_smooth(ranked_perc[bidx], smooth), label="{}-{}".format(group_label, bl), color=violin_colors[bl], linestyle=curve_styles[group_label])
        #     plt.bar(np.arange(0, len(ranked_perc[bidx]), 1) + width*idx, ranked_perc[bidx], label="{}-{}".format(paths_dict, bl), width=width)
        # if ylim != []:
        #     plt.ylim(ylim[0], ylim[1])
        # xticks_pos = list(range(0, all_ranks.max() + 1, 25))
        # xticks_labels = list(range(0, all_ranks.max() + 1, 25))
        # plt.xticks(xticks_pos, xticks_labels, rotation=60, fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['top'].set_visible(False)

        ranked_perc_dic[paths_dict] = ranked_perc

    for bidx, bl in enumerate(baselines):
        plt.bar(np.arange(0, len(ranked_perc_dic["Nonlinear"][bidx]), 1),
                ranked_perc_dic["Nonlinear"][bidx] - ranked_perc_dic["Linear"][bidx],
                label="{}-{}".format(paths_dict, bl), width=width)
    if ylim != []:
        plt.ylim(ylim[0], ylim[1])
    xticks_pos = list(range(0, all_ranks.max() + 1, 25))
    xticks_labels = list(range(0, all_ranks.max() + 1, 25))
    plt.xticks(xticks_pos, xticks_labels, rotation=60, fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if data_label:
        plt.legend()
    if xy_label:
        plt.xlabel('Goal Ranks')
        plt.ylabel('Percent of run that higher than the baseline')
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}".format(title))
    # plt.show()
    plt.close()


def correlation_change(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], ylim=[], smooth=1.0,
                       classify=[None, 50, 50], label=True,
                       property_keys={"lipschitz": "Complexity Reduction", "distance": "Dynamics Awareness",
                                      "ortho": "Orthogonality", "interf": "Noninterference", "diversity": "Diversity",
                                      "sparsity": "Sparsity"},
                       property_perc=None, property_rank=None, plot_origin=True):
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)

    all_goals_cor, property_keys = correlation_load(all_paths_dict, goal_ids,
                                                    total_param=total_param, xlim=xlim, property_keys=property_keys,
                                                    property_perc=property_perc, property_rank=property_rank)

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

    plt.figure(figsize=(6, 6))
    pk_all = list(property_keys.keys())
    plt.plot([0] * len(curves[pk_all[0]]), "--", color="black")
    # tpk = len(pk_all)
    for pk in pk_all:
        plt.plot(curves[pk], label=property_keys[pk], linewidth=2, color=violin_colors[property_keys[pk]])
        if plot_origin:
            plt.plot(unsmooth_curves[pk], alpha=0.4, color=violin_colors[property_keys[pk]])

    xticks_pos = list(range(0, all_ranks.max() + 1, 50))
    xticks_labels = list(range(0, all_ranks.max() + 1, 50))
    plt.xticks(xticks_pos, xticks_labels, rotation=60)
    # plt.xticks(list(range(all_ranks.max()+1))[1:], all_ranks, rotation=90)

    xticks_pos = list(range(0, all_ranks.max() + 1, 50))
    xticks_labels = list(range(0, all_ranks.max() + 1, 50))
    plt.xticks(xticks_pos, xticks_labels, rotation=60, fontsize=30)
    yticks_pos = [-0.6, -0.4, 0.0, 0.4, 0.6]
    # for y in yticks_pos:
    #     plt.plot([y] * len(curves[pk_all[0]]), "--", color="black")
    yticks_labels = yticks_pos
    plt.yticks(yticks_pos, yticks_labels, fontsize=30)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if label:
        plt.legend()
        plt.xlabel('Goal Ranks')
        plt.ylabel('Correlation')
    if ylim != []:
        plt.ylim(ylim[0], ylim[1])

    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}".format(title))
    # plt.show()
    plt.close()


def correlation_overall(all_paths_dict, goal_ids, ax, total_param=None, xlim=[],
                        property_keys={"lipschitz": "Complexity Reduction", "distance": "Dynamics Awareness",
                                       "ortho": "Orthogonality", "interf": "Noninterference", "diversity": "Diversity",
                                       "sparsity": "Sparsity"},
                        property_perc=None, property_rank=None):
    # all_ranks = []
    # for goal in goal_ids:
    #     all_ranks.append(ranks[goal])
    # all_ranks = np.array(all_ranks)

    file_path = "plot/temp_data/"
    if os.path.isfile(file_path + "overall_cor.pkl") and os.path.isfile(file_path + "property_keys.pkl"):
        with open(file_path + "overall_cor_{}.pkl".format(property_rank), "rb") as f:
            overall_cor = pickle.load(f)
        with open(file_path + "property_keys.pkl", "rb") as f:
            property_keys = pickle.load(f)
    else:
        _, overall_cor, property_keys = correlation_load(all_paths_dict, goal_ids,
                                                         total_param=total_param, xlim=xlim,
                                                         property_keys=property_keys,
                                                         property_perc=property_perc, property_rank=property_rank,
                                                         get_overall=True)
        with open(file_path + "overall_cor_{}.pkl".format(property_rank), "wb") as f:
            pickle.dump(overall_cor, f)
        with open(file_path + "property_keys.pkl", "wb") as f:
            pickle.dump(property_keys, f)

    ordered_cor = []
    ordered_prop = ["lipschitz", "distance", "diversity", "ortho", "sparsity", "interf"]
    for pk in ordered_prop:
        ordered_cor.append(overall_cor[pk])

    xticks_pos = np.arange(len(ordered_cor))
    ax.bar(xticks_pos, ordered_cor, color="#3498db", capsize=10)
    ax.axhline(y=0, c="black", ls="--")
    xticks_labels = [property_keys[pk] for pk in ordered_prop]
    yticks = [-0.6, -0.4, 0, 0.4, 0.6]
    ax.set_yticks(yticks, yticks, fontsize=14)
    ax.set_xticks(xticks_pos, xticks_labels, rotation=90, fontsize=14)
    for i, c in enumerate(ordered_cor):
        ax.text(xticks_pos[i] - 0.4, 0.1, "{:.2f}".format(c), color='black', fontweight='bold')


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


def figure3(paths_dicts_all, subtitles, baselines, groups, goal_ids, ranks, title, total_param=None, xlim=[],
            xy_label=True, data_label=True, figsize=(8, 6), ax=None):
    ready_plot = []
    ready_plot_error = []
    ready_label = []
    all_bases = []
    for paths_dicts in paths_dicts_all:
        labels = [i["label"] for i in paths_dicts]
        all_ranks = []
        for goal in goal_ids:
            all_ranks.append(ranks[goal])
        all_ranks = np.array(all_ranks)

        all_goals_auc, all_goals_independent = pick_best_perfs(paths_dicts, goal_ids, total_param, xlim, labels,
                                                               get_each_run=True)

        subp_data = np.zeros((len(baselines), len(groups)))
        subp_error = np.zeros((len(baselines), len(groups)))
        subp_label = []
        key_lst = list(groups.keys())
        one_setting_bases = []
        for gk_idx, group_key in enumerate(key_lst):
            filtered_labels = groups[group_key]
            # filtered_labels = labels
            # for bs in baselines:
            #     filtered_labels.remove(bs)

            # ranked_improve = np.zeros((len(baselines), all_ranks.max() + 1))
            ranked_improve = np.zeros((1, all_ranks.max() + 1))
            for goal in goal_ids:
                bases = np.zeros(len(baselines))
                labels = []
                for bidx, bl in enumerate(baselines):
                    bases[bidx] = all_goals_auc[goal][bl][0]
                    labels.append(bl)
                one_setting_bases.append(bases)
                # improvement = np.zeros(len(baselines))
                # for bidx, bl in enumerate(baselines):
                #     impr_all_label = []
                #     for label in filtered_labels:
                #         runs = all_goals_independent[goal][label]
                #         aucs = runs.sum(axis=1)
                #         # impr = aucs - bases[bidx]
                #         impr = aucs / bases[bidx]
                #         # impr = (aucs - bases[bidx]) /bases[bidx]
                #         impr_all_label.append(impr.mean())
                #     improvement[bidx] = np.array(impr_all_label).mean()

                impr_all_label = []
                improvement = np.zeros(1)
                for label in filtered_labels:
                    runs = all_goals_independent[goal][label]
                    aucs = runs.sum(axis=1)
                    bidx = 0 if label.split("+")[0] == "ReLU" else 1
                    impr = aucs / bases[bidx]
                    impr_all_label.append(impr.mean())
                improvement[0] = np.array(impr_all_label).mean()

                rank = ranks[goal]
                ranked_improve[:, rank] = improvement
                # ranked_goal[rank] = goal
            avg_improve = ranked_improve.mean(axis=1)
            std_improve = 1.96 * np.std(ranked_improve, axis=1) / np.sqrt(len(ranked_improve[0]))
            subp_data[:, gk_idx] = avg_improve
            subp_error[:, gk_idx] = std_improve
            subp_label.append(group_key)
        ready_plot.append(subp_data)
        ready_plot_error.append(subp_error)
        ready_label.append(subp_label)
        all_bases.append(np.array(one_setting_bases))

    # fig = plt.figure()
    # for idx, subp_data in enumerate(ready_plot):
    #     xticks_pos = np.array(list(range(0, len(subp_data[0]))))
    #     for bs_idx, improv in enumerate(subp_data):
    #         ax1 = fig.add_subplot(2, len(ready_plot), idx+1)
    #         ax2 = fig.add_subplot(2, len(ready_plot), idx+len(ready_plot)+1, sharex=ax1)
    #         ax1.spines['bottom'].set_visible(False)
    #         ax2.spines['top'].set_visible(False)
    #         ax1.set_ylim(-0.8, 2)
    #         ax2.set_ylim(-5, -2.5)
    #         ax1.bar(xticks_pos+0.1*bs_idx, improv)
    #         ax2.bar(xticks_pos+0.1*bs_idx, improv)
    #
    #         # ax1.tick_params(axis='x', which='both', bottom=False)
    #         ax1.tick_params(labeltop=False)
    #         ax2.set_xticks(xticks_pos, ready_label[idx], rotation=60, fontsize=14)
    #
    # plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')

    # fig, axs = plt.subplots(nrows=2, ncols=len(ready_plot), sharex='col', figsize=(6, 3))
    # plt.subplots_adjust(hspace=0.1, wspace=0.4)
    ylims = [
        [[1, 2], [-1, 0.1]],
        [[-0.1, 0.1], [-5, -2]]
    ]
    text_y = [
        [-0.5, 0.9, 0.6, 0.8],
        [-0.8, -0.4, -0.4, -0.4]
    ]

    # for idx, subp_data in enumerate(ready_plot):
    #     xticks_pos = np.array(list(range(0, len(subp_data[0]))))
    #     for bs_idx, improv in enumerate(subp_data):
    #         ax1 = axs[0, idx]
    #         ax2 = axs[1, idx]
    #         ax1.bar(xticks_pos + 0.1 * bs_idx, improv)
    #         ax2.bar(xticks_pos + 0.1 * bs_idx, improv)
    #         ax1.set_ylim(ylims[idx][0][0], ylims[idx][0][1])
    #         ax1.set_yticks(ylims[idx][0], ylims[idx][0])
    #         ax2.set_ylim(ylims[idx][1][0], ylims[idx][1][1])
    #         ax2.set_yticks(ylims[idx][1], ylims[idx][1])
    #
    #         ax1.axhline(y=0,c="black", ls="--")
    #         ax2.axhline(y=0,c="black", ls="--")
    #
    #         # for i, v in enumerate(improv):
    #         #     ax1.text(xticks_pos[i]+0.1*bs_idx-0.1, text_y[idx][i], "{:.1f}".format(v), color='black', fontweight='bold')
    #         #     ax2.text(xticks_pos[i]+0.1*bs_idx-0.1, text_y[idx][i], "{:.1f}".format(v), color='black', fontweight='bold')
    #
    #         ax1.spines['bottom'].set_visible(False)
    #         ax2.spines['top'].set_visible(False)
    #
    #         # hide the spines between ax and ax2
    #         ax1.spines.bottom.set_visible(False)
    #         ax2.spines.top.set_visible(False)
    #         # ax1.xaxis.tick_top()
    #         ax1.tick_params(labeltop=False, bottom=False)  # don't put tick labels at the top
    #         ax2.xaxis.tick_bottom()
    #         ax2.set_xticks(xticks_pos, ready_label[idx], rotation=60, fontsize=14)
    #
    #         d = .5  # proportion of vertical to horizontal extent of the slanted line
    #         kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
    #                       linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    #         ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    #         ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    #
    # plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')

    def v_pos(v, yrange):
        p = 0.05
        h = (yrange[1] - yrange[0]) * p + yrange[0]
        return h

    ylims = [
        [0, 1.6],
        [0, 1.6]
    ]
    # ylims = [
    #     [-0.8, 2.1],
    #     [-5.2, 0.5]
    # ]
    # fig, axs = plt.subplots(nrows=1, ncols=len(ready_plot), figsize=(6, 2))
    # if len(ready_plot) == 1:
    #     axs = [axs]
    all_bases = np.array(all_bases)
    assert len(ready_plot) == 1

    for idx, subp_data in enumerate(ready_plot):
        subp_error = ready_plot_error[idx]
        xticks_pos = np.array(list(range(0, len(subp_data[0]))))
        # for bs_idx, improv in enumerate(subp_data):
        improv = subp_data[0]
        bs_idx = 0
        xtcks = []
        colors = []
        for lb in ready_label[idx]:
            xtck_lst = lb.split("+")
            if len(xtck_lst) == 1:
                xtck = "No-aux"
            else:
                # xtck = xtck_lst[1]
                xtck = "With-aux"
            xtcks.append(xtck)

            if xtck_lst[0] == "ReLU":
                color = "#e74c3c"
            else:
                color = "#3498db"
            colors.append(color)

        # axs[idx].bar(xticks_pos+0.1*bs_idx, improv, color=colors, yerr=subp_error[bs_idx], capsize=10, edgecolor=None)
        rhs_axs = ax.twinx()
        ax.bar(xticks_pos[:2] + 0.1 * bs_idx, improv[:2], color=colors[:2], yerr=subp_error[bs_idx][:2], capsize=10,
               edgecolor=None)
        rhs_axs.bar(xticks_pos[2:] + 0.1 * bs_idx, improv[2:], color=colors[2:], yerr=subp_error[bs_idx][2:],
                    capsize=10, edgecolor=None)

        # if xtck_lst[0] in ["ReLU", "FTA"] and len(xtck_lst) == 0:
        #     axs[idx].text(xticks_pos[i], v_pos(v, ylims[idx][1]-0.2), xtck_lst[0], color=colors, fontsize=12)
        ax.text(xticks_pos[0], ylims[idx][0] + 1.4, "ReLU", color="#c0392b", fontsize=14)
        ax.text(xticks_pos[2] + 0.2, ylims[idx][0] + 1.4, "FTA", color="#2980b9", fontsize=14)
        ax.set_xticks(xticks_pos, xtcks, rotation=30, fontsize=14, ha='right', rotation_mode='anchor')
        rhs_axs.set_xticks(xticks_pos, xtcks, rotation=30, fontsize=14, ha='right', rotation_mode='anchor')

        # axs[idx].axhline(y=1,c="black", ls="--")
        xlim = ax.get_xlim()
        ax.plot([xlim[0], xlim[0] + (xlim[-1] - xlim[0]) / 2], [1, 1], c="#912E23", ls="--")
        rhs_axs.plot([xlim[0] + (xlim[-1] - xlim[0]) / 2, xlim[-1]], [1, 1], c="#236591", ls="--")
        ax.set_xlim(xlim)

        ax.set_title(subtitles[idx], fontsize=14)
        ax.set_ylim(ylims[idx])
        rhs_axs.set_ylim(ylims[idx])
        ax.set_yticks([1], [1], fontsize=12, color="#912E23")
        # print(idx, all_bases[idx], all_bases[idx][:, 0].sum() / all_bases[idx][:, 1].sum())
        align_yaxis(ax, 1, rhs_axs, all_bases[idx][:, 0].sum() / all_bases[idx][:, 1].sum())
        rhs_axs.set_yticks([1], [1], fontsize=12, color="#236591")

        for i, v in enumerate(improv):
            rhs_axs.text(xticks_pos[i] - 0.4, v_pos(v, rhs_axs.get_ylim()), "{:.2f}".format(v), color='black',
                         fontsize=12)


def mountain_car():
    learning_curve(mc_learn_sweep, "mountain car learning sweep")


def simple_maze():
    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        "ReLU+ATC",
        # "ReLU(L)",
        # "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF",
        # "ReLU(L)+ATC",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
        "Scratch", "Scratch(FTA)", "Input", "Random",  # "Scratch(L)", "Random(L)",
    ]
    # learning_curve(gh_nonlinear_transfer_sweep_v13_largeReLU, "maze learning sweep")

    goal_ids = [106,
                107, 108, 118, 119, 109, 120, 121, 128, 110, 111, 122, 123, 129, 130, 142, 143, 144, 141, 140, 139, 138,
                156, 157, 158, 155, 170, 171, 172, 169, 154, 168, 153, 167, 152, 166, 137, 151,
                165, 127, 117, 105, 99, 136, 126, 150, 164, 116, 104, 86, 98, 85, 84, 87, 83, 88, 89, 97, 90, 103, 115,
                91, 92, 93, 82, 96, 102, 114, 81, 80, 71, 95, 70, 69, 68, 101, 67, 66, 113, 62,
                65, 125, 61, 132, 47, 133, 79, 48, 72, 94, 52, 146, 73, 49, 33, 100, 134, 34, 74, 50, 147, 35, 112, 75,
                135, 160, 36, 148, 76, 161, 63, 77, 124, 149, 78, 162, 163, 131, 53, 145, 159,
                54, 55, 56, 57, 58, 59, 64, 51, 38, 60, 39, 40, 46, 41, 42, 43, 44, 45, 32, 31, 37, 30, 22, 21, 23, 20,
                24, 19, 25, 18, 26, 17, 16, 27, 15, 28, 29, 7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2,
                13, 1, 14, 0,
                ]

    emphasize = [
        "ReLU+VirtualVF5", "ReLU+SF",
        "FTA eta=0.2", "FTA+SF", "FTA+NAS",
        "Scratch", "Scratch(FTA)", "Random", "Input",
    ]
    # performance_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "linear/maze transfer chosen (smooth 0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), emphasize=emphasize)
    emphasize = [
        "ReLU",
        "ReLU+VirtualVF5",
        "FTA eta=0.2",
        "FTA+VirtualVF5",
    ]
    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        "ReLU+ATC",
        "FTA eta=0.6", "FTA eta=0.8", "FTA eta=0.2", "FTA eta=0.4",
        "FTA+ATC", "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+XY", "FTA+Decoder",
        "Scratch", "Input", "Random", "Scratch(FTA)"  # "Scratch(L)", "Random(L)",
    ]
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze_transfer_chosen_(smooth0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), color_by_activation=True)
    #                    # in_plot_label=[{"ReLU": ["#e74c3c", "-", 0.5]}, {"FTA":["#3498db", "-", 0.5]},
    #                    #                {"Scratch": ["#34495e", "-", 1]}, {"Random & Input": ["#bdc3c7", "-", 0.5]}])#, emphasize=emphasize)
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/temp", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), emphasize=emphasize)

    targets = [
        "ReLU(L)",
        "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward",
        "ReLU(L)+SF",
        "ReLU(L)+ATC",
        "Input", "Scratch(L)", "Random(L)",
    ]
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze_transfer_relul_(smooth0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), color_by_activation=True)

    # draw_label(targets, "auc shared label", ncol=4,
    #            emphasize=["ReLU+VirtualVF5", "ReLU+SF", "FTA+VirtualVF5", "Scratch", "Random", "Input"])
    # targets = ["FTA+NAS"]
    # draw_label(targets, "auc linear label", ncol=2,
    #            emphasize=targets)
    # targets = ["ReLU(L)+VirtualVF5", "ReLU(L)", "FTA+SF"]
    # draw_label(targets, "auc nonlinear label", ncol=2,
    #            emphasize=targets)
    # draw_label(["ReLU+VirtualVF5", "ReLU+SF", "ReLU(L)", "ReLU(L)+VirtualVF5", "FTA+VirtualVF5", "FTA+NAS", "FTA+VirtualVF1", "FTA+SF", "Scratch", "Random", "Input", "Other rep"],
    #            "auc shared label", ncol=3)
    # draw_label([{"ReLU": ["#e74c3c", "-"]}, {"FTA":["#3498db", "-"]}, {"Scratch": ["#34495e", "-"]}, {"Random & Input": ["#bdc3c7", "-"]}],
    #            "auc nonlinear label", ncol=1)

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        "ReLU+ATC",
        "ReLU(L)",
        "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward",
        "ReLU(L)+SF", "ReLU(L)+ATC",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
        "Scratch", "Scratch(FTA)",
    ]
    baselines = ["Scratch", "Scratch(FTA)"]
    # performance_change_vf_difference(targets, {"Linear": gh_transfer_sweep_v13, "Nonlinear": gh_nonlinear_transfer_sweep_v13_largeReLU}, goal_ids, ranks, "maze better than baseline (smooth 0.1)",
    #                               xlim=[0, 11], ylim=[], smooth=0.1,
    #                               xy_label=False, data_label=False, figsize=(8, 6), baselines=baselines)
    # draw_label(["The degree that non-linear VF is better than linear VF"],
    #            "auc bar label", ncol=1, with_style=False, with_color=False)
    # draw_label(["Linear", "Nonlinear"],
    #            "auc bar label", ncol=2, with_style=False)
    targets = [
        "ReLU",
        "ReLU+VirtualVF5",
        # "ReLU+VirtualVF1", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF", "ReLU+ATC",
        "FTA eta=0.2",  # "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+SF",  # "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+ATC",
        "Scratch",
        "Scratch(FTA)",
    ]
    groups = {
        "ReLU": ["ReLU"],
        # "ReLU+Aux": ["ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF", "ReLU+ATC"],
        "ReLU+VirtualVF5": ["ReLU+VirtualVF5"],
        "FTA": ["FTA eta=0.2"],  # , "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",],
        # "FTA+Aux": ["FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC"],
        "FTA+SF": ["FTA+SF"],
    }
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 2))
    plt.subplots_adjust(wspace=0.3, hspace=None)
    figure3([label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU)], ["Non-linear"],
            baselines, groups, goal_ids, ranks, "result1",
            xlim=[], xy_label=True, data_label=True, figsize=(8, 6), ax=axs[0])
    targets = [
        "ReLU",
        "ReLU+VirtualVF5",
        # "ReLU+VirtualVF1", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF", "ReLU+ATC",
        "FTA eta=0.2",  # "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+NAS",  # "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+ATC",
        "Scratch",
        "Scratch(FTA)",
    ]
    groups = {
        "ReLU": ["ReLU"],
        # "ReLU+Aux": ["ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF", "ReLU+ATC"],
        "ReLU+VirtualVF5": ["ReLU+VirtualVF5"],
        "FTA": ["FTA eta=0.2"],  # , "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",],
        # "FTA+Aux": ["FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC"],
        "FTA+NAS": ["FTA+NAS"],
    }

    # figure3([label_filter(targets, gh_transfer_sweep_v13)], ["Linear"],
    #         baselines, groups, goal_ids, ranks, "result1",
    #         xlim=[], xy_label=True, data_label=True, figsize=(8, 6), ax=axs[1])

    plt.savefig("plot/img/result1.pdf", dpi=300, bbox_inches='tight')

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        "ReLU+ATC",
        "Scratch"
    ]
    emphasize = [
        "ReLU+VirtualVF5", "ReLU+ATC", "Scratch"
    ]
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer relu (smooth 0.1)", xlim=[0, 11], ylim=[2, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), emphasize=emphasize)
    # draw_label(emphasize+[{"Other ReLU reps": ["grey", s_default[1]]}], "auc relu label", ncol=3)

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        "ReLU+ATC",
        "ReLU(L)",
        "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward",
        "ReLU(L)+SF",
        "ReLU(L)+ATC",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
        "FTA+ATC",
    ]
    # correlation_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks,
    #                    "linear/maze correlation change (no smooth)", xlim=[0, 11], smooth=1, label=False)
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (no smooth)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=1, label=False, plot_origin=False)
    # correlation_change(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks,
    #                    "linear/maze correlation change (smooth 0.1)", xlim=[0, 11], smooth=0.1, label=False)
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (smooth 0.1)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False)

    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low otho)", xlim=[0, 11], ylim=[-0.4, 0.7], smooth=0.1, label=False, plot_origin=False,
    #                    property_rank={"ortho": [0, 89]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high otho)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_rank={"ortho": [90, np.inf]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low otho) (no smooth)", xlim=[0, 11], ylim=[-0.4, 0.7], smooth=1, label=False, plot_origin=False,
    #                    property_rank={"ortho": [0, 89]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high otho) (no smooth)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=1, label=False, plot_origin=False,
    #                    property_rank={"ortho": [90, np.inf]})
    #
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low div)", xlim=[0, 11], ylim=[-0.4, 0.7], smooth=0.1, label=False, plot_origin=False,
    #                    property_rank={"diversity": [0, 105]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high div)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_rank={"diversity": [106, np.inf]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low div) (no smooth)", xlim=[0, 11], ylim=[-0.4, 0.7], smooth=1, label=False, plot_origin=False,
    #                    property_rank={"diversity": [0, 105]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high div) (no smooth)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=1, label=False, plot_origin=False,
    #                    property_rank={"diversity": [106, np.inf]})
    #
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low comp reduc)", xlim=[0, 11], ylim=[-0.4, 0.7], smooth=0.1, label=False, plot_origin=False,
    #                    property_rank={"lipschitz": [0, 95]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high comp reduc)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_rank={"lipschitz": [96, np.inf]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low comp reduc) (no smooth)", xlim=[0, 11], ylim=[-0.4, 0.7], smooth=1, label=False, plot_origin=False,
    #                    property_rank={"lipschitz": [0, 95]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high comp reduc) (no smooth)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=1, label=False, plot_origin=False,
    #                     property_rank={"lipschitz": [96, np.inf]})

    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low dist)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_perc={"distance": [0, 70]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high dist)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_perc={"distance": [70, 100]})
    #
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low spar)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_perc={"sparsity": [0, 70]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high spar)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_perc={"sparsity": [70, 100]})
    #
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (low interf)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_perc={"interf": [0, 70]})
    # correlation_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks,
    #                    "nonlinear/maze correlation change (high interf)", xlim=[0, 11], ylim=[-0.85, 0.85], smooth=0.1, label=False, plot_origin=False,
    #                    property_perc={"interf": [70, 100]})

    # draw_label(["Complexity Reduction", "Dynamics Awareness", "Noninterference", "Diversity", "Sparsity"], "ortho slice label", ncol=3)
    # draw_label(["Complexity Reduction", "Dynamics Awareness", "Noninterference", "Orthogonality", "Sparsity"], "diversity slice label", ncol=3)

    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer high ortho (smooth 0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), property_perc={"ortho": [70, 100]})
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer low ortho (smooth 0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), property_perc={"ortho": [0, 70]})
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer high div (smooth 0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), property_perc={"diversity": [70, 100]})
    # performance_change(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ranks, "nonlinear/maze transfer low div (smooth 0.1)", xlim=[0, 11], ylim=[0, 11], smooth=0.1,
    #                    xy_label=False, data_label=False, linewidth=3, figsize=(8, 6), property_perc={"diversity": [0, 70]})
    # draw_label(targets, "auc all label", ncol=4)

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        "ReLU+ATC",
        "ReLU(L)",
        "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward",
        "ReLU(L)+SF",
        "ReLU(L)+ATC",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
        "FTA+ATC",
    ]
    # fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    # correlation_overall(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, axs[0], xlim=[0, 11],
    #                    property_rank={"ortho": [90, np.inf]})
    # axs[0].set_title("High Orthogonality")
    # correlation_overall(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, axs[1], xlim=[0, 11],
    #                    property_rank={"diversity": [106, np.inf]})
    # axs[1].set_title("High Diversity")
    # correlation_overall(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, axs[2], xlim=[0, 11],
    #                    property_rank={"lipschitz": [96, np.inf]})
    # axs[2].set_title("High Complexity Reduction")
    # plt.tight_layout()
    # plt.savefig("plot/img/nonlinear/high_prop.pdf", dpi=300, bbox_inches='tight')
    # plt.close()

    # fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    # correlation_overall(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, axs[0], xlim=[0, 11],
    #                    property_rank={"ortho": [0, 89]})
    # axs[0].set_title("Low Orthogonality")
    # correlation_overall(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, axs[1], xlim=[0, 11],
    #                    property_rank={"diversity": [0, 105]})
    # axs[1].set_title("Low Diversity")
    # correlation_overall(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, axs[2], xlim=[0, 11],
    #                    property_rank={"lipschitz": [0, 95]})
    # axs[2].set_title("Low Complexity Reduction")
    # plt.tight_layout()
    # plt.savefig("plot/img/nonlinear/low_prop.pdf", dpi=300, bbox_inches='tight')
    # plt.close()

    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # correlation_overall(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, ax, xlim=[0, 11])
    # plt.tight_layout()
    # plt.savefig("plot/img/nonlinear/all_prop.pdf", dpi=300, bbox_inches='tight')
    # plt.close()


def maze_multigoals():
    # learning_curve(ghmg_original_sweep_v13, "multigoal rep sweep result")

    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1
    goal_ids = [106,
                107, 108, 118, 119, 109, 120, 121, 128, 110, 111, 122, 123, 129, 130, 142, 143, 144, 141, 140, 139, 138,
                156, 157, 158, 155, 170, 171, 172, 169, 154, 168, 153, 167, 152, 166, 137, 151,
                165, 127, 117, 105, 99, 136, 126, 150, 164, 116, 104, 86, 98, 85, 84, 87, 83, 88, 89, 97, 90, 103, 115,
                91, 92, 93, 82, 96, 102, 114, 81, 80, 71, 95, 70, 69, 68, 101, 67, 66, 113, 62,
                65, 125, 61, 132, 47, 133, 79, 48, 72, 94, 52, 146, 73, 49, 33, 100, 134, 34, 74, 50, 147, 35, 112, 75,
                135, 160, 36, 148, 76, 161, 63, 77, 124, 149, 78, 162, 163, 131, 53, 145, 159,
                54, 55, 56, 57, 58, 59, 64, 51, 38, 60, 39, 40, 46, 41, 42, 43, 44, 45, 32, 31, 37, 30, 22, 21, 23, 20,
                24, 19, 25, 18, 26, 17, 16, 27, 15, 28, 29, 7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2,
                13, 1, 14, 0,
                ]
    # performance_change(ghmg_transfer_sweep_v13, goal_ids, ranks, "multigoal transfer change (smooth 0.2)", xlim=[0, 11], smooth=0.2)
    # performance_change(ghmg_transfer_last_sweep_v13, goal_ids, ranks, "multigoal lastrep transfer change (smooth 0.2)", xlim=[0, 11], smooth=0.2)

    performance_change(ghmg_transfer_sweep_v13, goal_ids, ranks, "multigoal transfer change", xlim=[0, 11], smooth=1)
    performance_change(ghmg_transfer_last_sweep_v13, goal_ids, ranks, "multigoal lastrep transfer change", xlim=[0, 11],
                       smooth=1)


if __name__ == '__main__':
    simple_maze()
    # maze_multigoals()

