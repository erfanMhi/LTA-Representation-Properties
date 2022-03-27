import os
import pickle
import sys
import copy
import numpy as np
import itertools
from sklearn import ensemble, metrics
import scipy
import seaborn as sns
# from sklearn.inspection import permutation_importance
# from sklearn.linear_model import BayesianRidge

import matplotlib.pyplot as plt

sys.path.insert(0, '..')
print(sys.path)
from plot.plot_paths import *
from plot.plot_utils import *

print("Change dir to", os.getcwd())

os.chdir("..")
print(os.getcwd())


def correlation_bar(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], smooth=1.0):
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)

    all_goals_cor, property_keys = correlation_load(all_paths_dict, goal_ids, total_param=total_param, xlim=xlim)

    ordered_prop = list(property_keys.keys())
    ordered_labels = [property_keys[k] for k in ordered_prop]

    x = np.arange(len(ordered_labels))  # the label locations
    width = 0.15  # the width of the bars
    label_pos = 0.03
    fontsize = 9
    rotation = 90
    fig, ax = plt.subplots()

    for j, goal in enumerate(goal_ids):
        ordered_corr = []
        for pk in ordered_prop:
            ordered_corr.append(all_goals_cor[pk][goal])
        ax.bar(x - 0.45 + width * (j + 1), ordered_corr, width, label="Rank={}".format(all_ranks[j]), color=cmap(j, 4))
        for i in range(len(x)):
            ax.text(x[i] - 0.45 + width * (0.6 + j), max(0, ordered_corr[i]) + label_pos,
                    "{:.4f}".format(ordered_corr[i]), color=cmap(j, 4), fontsize=fontsize, rotation=rotation)

    ax.plot([x[0] - 0.45, x[-1] + 0.45], [0, 0], "--", color="grey")

    ax.legend(loc=4)
    ax.set_ylabel('Correlation')
    ax.set_ylim(-1, 1.2)
    ax.set_xticks(x)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticklabels(ordered_labels, rotation=30)
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()


def xgboost_analysis(property_keys, all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], smooth=1.0,
                     top_runs=[0, 1.0]):
    ordered_goal_ids = []
    ordered_goal_ranks = []
    rank2goal = dict((v, k) for k, v in ranks.items())
    for r in range(len(rank2goal)):
        if rank2goal[r] in goal_ids:
            ordered_goal_ids.append(rank2goal[r])
            ordered_goal_ranks.append(r)

    labels = [i["label"] for i in all_paths_dict]

    all_goals_auc = pick_best_perfs(all_paths_dict, ordered_goal_ids, total_param, xlim, labels, top_runs=top_runs)

    formated_path = {}
    for goal in ordered_goal_ids:
        g_path = copy.deepcopy(all_paths_dict)
        for i in range(len(all_paths_dict)):
            label = g_path[i]["label"]
            best_param_folder = all_goals_auc[goal][label][1]
            best = int(best_param_folder.split("_")[0])
            g_path[i]["control"] = [g_path[i]["control"].format(goal), best]
        formated_path[goal] = g_path

    all_goals_perf = {}
    all_goals_prop = {}
    for pk in property_keys.keys():
        properties, _ = load_property([formated_path[ordered_goal_ids[0]]], property_key=pk, early_stopped=True)
        all_goals_prop[pk] = properties
        all_goals_perf[pk] = {}
        for goal in ordered_goal_ids:
            transf_perf, _ = load_property([formated_path[goal]], property_key="return", early_stopped=True)
            all_goals_perf[pk][goal] = transf_perf

    features = []
    prop_lst = list(all_goals_prop.keys())
    for i, prop in enumerate(prop_lst):
        col = []
        for j, rep in enumerate(list(all_goals_prop[prop].keys())):
            for k, run in enumerate(list(all_goals_prop[prop][rep].keys())):
                col.append(all_goals_prop[prop][rep][run])
        features.append(col)
    features = np.array(features).transpose()  # [n_samples, m_features]

    importance_change = np.zeros((len(prop_lst), len(ordered_goal_ids)))
    for g_idx, goal in enumerate(ordered_goal_ids):
        print("\nTraining xgboost for goal {}".format(goal))
        auc = []
        prop = prop_lst[0]
        for j, rep in enumerate(list(all_goals_prop[prop].keys())):
            for k, run in enumerate(list(all_goals_prop[prop][rep].keys())):
                auc.append(all_goals_perf[prop][goal][rep][run])
                assert auc[-1] == \
                       all_goals_perf[prop_lst[1]][goal][rep][run] == \
                       all_goals_perf[prop_lst[2]][goal][rep][run] == \
                       all_goals_perf[prop_lst[3]][goal][rep][run] == \
                       all_goals_perf[prop_lst[4]][goal][rep][run] == \
                       all_goals_perf[prop_lst[5]][goal][rep][run]
        auc = np.array(auc)
        params = {'n_estimators': 500,
                  'max_depth': 8,
                  'min_samples_split': 5,
                  'learning_rate': 0.01}
        reg = ensemble.GradientBoostingRegressor(**params)
        reg.fit(features, auc)
        # mse = metrics.mean_squared_error(auc, reg.predict(features)) # same training and test set
        # print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

        """https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html"""
        feature_importance = reg.feature_importances_
        # sorted_idx = np.argsort(feature_importance)
        # pos = np.arange(sorted_idx.shape[0]) + .5
        # fig = plt.figure(figsize=(6, 6))
        # plt.subplot(1, 1, 1)
        # plt.barh(pos, feature_importance, align='center')
        # plt.yticks(pos, np.array([property_keys[pt] for pt in prop_lst]))
        # plt.title('Feature Importance (MDI)')
        # plt.show()

        importance_change[:, g_idx] = feature_importance[:]

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    for pi, p in enumerate(prop_lst):
        plt.plot(exp_smooth(importance_change[pi], smooth), label=property_keys[p])
    plt.xticks(list(range(0, len(ordered_goal_ids), 25)),
               [ordered_goal_ranks[i] for i in list(range(0, len(ordered_goal_ids), 25))], rotation=60)
    plt.title('Feature Importance')
    plt.legend()
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')


def pair_property(property_keys, all_paths_dict, goal_ids, ranks, total_param=None, xlim=[], with_auc=True):
    labels = [i["label"] for i in all_paths_dict]

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

    all_goals_perf = {}
    all_goals_prop = {}
    for pk in property_keys.keys():
        properties, _ = load_property([formated_path[goal_ids[0]]], property_key=pk, early_stopped=True)
        all_goals_prop[pk] = properties
        all_goals_perf[pk] = {}
        for goal in goal_ids:
            transf_perf, _ = load_property([formated_path[goal]], property_key="return", early_stopped=True)
            all_goals_perf[pk][goal] = transf_perf

    prop1 = []
    prop2 = []
    goal = []
    auc = []

    p1 = list(property_keys.keys())[0]
    p2 = list(property_keys.keys())[1]

    for rep in all_goals_prop[p1].keys():
        for run in all_goals_prop[p1][rep].keys():
            for g in all_goals_perf[p1].keys():
                prop1.append(all_goals_prop[p1][rep][run])
                prop2.append(all_goals_prop[p2][rep][run])
                goal.append(g)
                auc.append(all_goals_perf[p1][g][rep][run])
                assert all_goals_perf[p1][g][rep][run] == all_goals_perf[p2][g][rep][run]

    fig, axes = plt.subplots(nrows=1, ncols=len(goal_ids), figsize=(4 * len(goal_ids), 4))
    if len(goal_ids) == 1:
        axes = [axes]
    for gi, g in enumerate(goal_ids):
        prop1_g, prop2_g, auc_g = [], [], []
        for i in range(len(auc)):
            if goal[i] == g:
                prop1_g.append(prop1[i])
                prop2_g.append(prop2[i])
                auc_g.append(auc[i])
        cor = np.corrcoef(prop1_g, prop2_g)[0][1]
        if with_auc:
            im = axes[gi].scatter(prop1_g, prop2_g, c=auc_g, cmap="Blues", vmin=0, vmax=11)
        else:
            c = "C1" if cor > 0.58 else "C0"
            im = axes[gi].scatter(prop1_g, prop2_g, c=c)
        xlabel = "\n" + property_keys[p1] if len(property_keys[p1].split(" ")) == 1 else "\n".join(
            property_keys[p1].split(" "))
        ylabel = "\n" + property_keys[p2] if len(property_keys[p2].split(" ")) == 1 else "\n".join(
            property_keys[p2].split(" "))
        axes[gi].set_xlabel(xlabel, fontsize=30)
        axes[gi].set_ylabel(ylabel, fontsize=30)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        axes[gi].set_title("Cor={:.2f}".format(cor), fontsize=30)
        # axes[gi].set_title("Goal {} Rank {}".format(g, ranks[g]))
    if with_auc:
        fig.colorbar(im)
        plt.savefig("plot/img/auc_{}-{}.png".format(p1, p2), dpi=300, bbox_inches='tight')
    else:
        plt.savefig("plot/img/{}-{}.pdf".format(p1, p2), dpi=300, bbox_inches='tight')


def property_auc_scatter(property_keys, all_paths_dict, groups, goal_ids, ranks, title, total_param=None, xlim=[]):
    labels = [i["label"] for i in all_paths_dict]

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

    all_goals_perf = {}
    all_goals_prop = {}
    for pk in property_keys.keys():
        properties, _ = load_property([formated_path[goal_ids[0]]], property_key=pk, early_stopped=True)
        all_goals_prop[pk] = properties
        all_goals_perf[pk] = {}
        for goal in goal_ids:
            transf_perf, _ = load_property([formated_path[goal]], property_key="return", early_stopped=True)
            all_goals_perf[pk][goal] = transf_perf

    fig, axes = plt.subplots(nrows=1, ncols=len(goal_ids), figsize=(6 * len(goal_ids), 4))
    p1 = list(property_keys.keys())[0]

    for name in groups:
        same_color = groups[name]
        prop1 = []
        goal = []
        auc = []
        for rep in all_goals_prop[p1].keys():
            if "_".join(rep.split("_")[:-1]) in same_color:
                for run in all_goals_prop[p1][rep].keys():
                    for g in all_goals_perf[p1].keys():
                        prop1.append(all_goals_prop[p1][rep][run])
                        goal.append(g)
                        auc.append(all_goals_perf[p1][g][rep][run])

        for gi, g in enumerate(goal_ids):
            prop1_g, prop2_g, auc_g = [], [], []
            for i in range(len(auc)):
                if goal[i] == g:
                    prop1_g.append(prop1[i])
                    auc_g.append(auc[i])
            im = axes[gi].scatter(prop1_g, auc_g, label=name)

    for gi, g in enumerate(goal_ids):
        axes[gi].set_xlabel(property_keys[p1])
        axes[gi].set_ylabel("AUC")
        axes[gi].set_title("Goal {} Rank {}".format(g, ranks[g]))
    if len(groups) > 1:
        plt.legend()
    plt.savefig("plot/img/{}-{}.png".format(title, p1), dpi=300, bbox_inches='tight')


def property_scatter(property_keys, all_paths_dict, groups, title):
    all_goals_prop = {}
    for pk in property_keys.keys():
        properties, _ = load_property([all_paths_dict], property_key=pk, early_stopped=True)
        all_goals_prop[pk] = properties

    nc = 3
    nr = int(np.ceil(len(property_keys.keys()) // nc))
    group_name = list(groups.keys())
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(8, 3 * nr))
    if nr == 1:
        axes = [axes]
    for pi, pk in enumerate(list(property_keys.keys())):
        allp = []
        allprop = []
        for ni, name in enumerate(group_name):
            same_color = groups[name]
            prop1 = []
            for rep in all_goals_prop[pk].keys():
                if "_".join(rep.split("_")[:-1]) in same_color:
                    for run in all_goals_prop[pk][rep].keys():
                        prop1.append(all_goals_prop[pk][rep][run])
                        allp.append(all_goals_prop[pk][rep][run])
            # axes[pi//nc][pi%nc].scatter(np.random.uniform(ni, ni+0.1, size=len(prop1)), prop1, s=6)
            allprop.append(prop1)
        axes[pi // nc][pi % nc].boxplot(allprop)

        axes[pi // nc][pi % nc].set_title(property_keys[pk])
        allp = np.array(allp)
        axes[pi // nc][pi % nc].set_yticks([allp.min(), allp.max()])
        axes[pi // nc][pi % nc].set_yticklabels(["{:.2f}".format(allp.min()), "{:.2f}".format(allp.max())])
        if pi // nc == (nr - 1):
            # axes[pi // nc][pi % nc].set_xticks([ni+0.25 for ni in range(len(group_name))])
            axes[pi // nc][pi % nc].set_xticklabels(group_name, fontsize=12, rotation=90)
        else:
            axes[pi // nc][pi % nc].get_xaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in {}".format(title))


def performance_scatter(all_paths_dict, groups, title, goal_ids, xlim=[], ylim=[]):
    labels = [i["label"] for i in all_paths_dict]

    # _, all_goals_independent = pick_best_perfs(all_paths_dict, goal_ids, None, xlim, labels, get_each_run=True)
    pklfile = "plot/temp_data/all_goals_independent_best_perf.pkl"
    if os.path.isfile(pklfile):
        with open(pklfile, "rb") as f:
            all_goals_independent = pickle.load(f)
        print("Load from {}".format(pklfile))
    else:
        _, all_goals_independent = pick_best_perfs(all_paths_dict, goal_ids, None, xlim, labels, get_each_run=True)
        with open(pklfile, "wb") as f:
            pickle.dump(all_goals_independent, f)

    overall_trans = []
    groups_keys = list(groups.keys())
    for gp in groups_keys:
        targets = groups[gp]
        temp = []
        for rep in targets:
            overall = []
            for g in all_goals_independent.keys():
                overall.append(all_goals_independent[g][rep].sum(axis=1))
            temp.append(np.array(overall).mean(axis=0))
        overall_trans.append(np.array(temp).flatten())

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    bp = ax.boxplot(overall_trans)
    no_aux_med = bp['medians'][0].get_ydata()[0]
    # ax.axhline(no_aux_med, c="C1", ls="--")
    # ax.text(7, 5.9, "No Aux\nMedian", c="C1", fontsize=25)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_yticks(ylim, labels=ylim, fontsize=28)
    ax.set_xticklabels(groups_keys, fontsize=28, rotation=90)

    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}".format(title))
    plt.close()

    # xticks_pos = list(range(0, all_ranks.max() + 1, 25))
    # xticks_labels = list(range(0, all_ranks.max() + 1, 25))
    # plt.xticks(xticks_pos, xticks_labels, rotation=60, fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    #
    # if data_label:
    #     plt.legend()
    # if xy_label:
    #     plt.xlabel('Goal Ranks')
    #     plt.ylabel('Percent of run that higher than the baseline')
    # plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    # print("Save in plot/img/{}".format(title))
    # # plt.show()
    # plt.close()


def performance_scatter_ste(all_paths_dict, groups, title, goal_ids, xlim=[], ylim=[]):
    labels = [i["label"] for i in all_paths_dict]

    pklfile = "plot/temp_data/all_goals_independent_best_perf.pkl"
    if os.path.isfile(pklfile):
        with open(pklfile, "rb") as f:
            all_goals_independent = pickle.load(f)
        print("Load from {}".format(pklfile))
    else:
        _, all_goals_independent = pick_best_perfs(all_paths_dict, goal_ids, None, xlim, labels, get_each_run=True)
        with open(pklfile, "wb") as f:
            pickle.dump(all_goals_independent, f)

    overall_trans = []
    overall_means = []
    overall_stes = []
    groups_keys = list(groups.keys())
    for gp in groups_keys:
        targets = groups[gp]
        temp = []
        count_ov = 0
        for rep in targets:
            overall = []
            for g in all_goals_independent.keys():
                overall.append(all_goals_independent[g][rep].sum(axis=1))
                count_ov += 1
            # temp.append(np.array(overall).mean(axis=0))
            temp += overall
        temp = np.array(temp).flatten()
        overall_trans.append(temp)
        overall_means.append(np.mean(temp))
        overall_stes.append(np.std(temp) / np.sqrt(count_ov) * 1.96)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    # plt.errorbar(np.arange(len(overall_means)), overall_means, overall_stes, fmt='ok', ecolor="grey", elinewidth=5, markersize=5)
    plt.bar(np.arange(len(overall_means)), overall_means, color="#3498db", yerr=overall_stes, capsize=10)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_yticks(ylim, labels=ylim, fontsize=28)
    ax.set_xticks(np.arange(len(overall_means)), fontsize=28, rotation=90)
    ax.set_xticklabels(groups_keys, fontsize=28, rotation=90)

    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}".format(title))
    plt.close()


def property_auc_goal(property_key, all_paths_dict, goal_ids, ranks, title,
                      total_param=None, xlim=[], classify=[50, 50], figsize=(8, 6), xy_label=True, legend=True):
    ordered_goal_ids = []
    ordered_goal_ranks = []
    rank2goal = dict((v, k) for k, v in ranks.items())
    for r in range(len(rank2goal)):
        if rank2goal[r] in goal_ids:
            ordered_goal_ids.append(rank2goal[r])
            ordered_goal_ranks.append(r)

    labels = [i["label"] for i in all_paths_dict]

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

    prop_log = []

    properties, _ = load_property([formated_path[goal_ids[0]]], property_key=property_key, early_stopped=True)
    all_goals_prop = properties
    all_goals_perf = {}
    for rep in properties:
        for run in properties[rep]:
            prop_log.append(properties[rep][run])
    for goal in goal_ids:
        transf_perf, _ = load_property([formated_path[goal]], property_key="return", early_stopped=True)
        all_goals_perf[goal] = transf_perf

    prop_log = np.array(prop_log)
    lim1 = np.percentile(prop_log, classify[0])
    lim2 = np.percentile(prop_log, classify[1])

    prop1, prop2 = [[] for i in range(len(ordered_goal_ids))], [[] for i in range(len(ordered_goal_ids))]
    auc1, auc2 = [[] for i in range(len(ordered_goal_ids))], [[] for i in range(len(ordered_goal_ids))]
    for rep in all_goals_prop.keys():
        for run in all_goals_prop[rep].keys():
            for g in all_goals_perf.keys():
                gidx = ordered_goal_ids.index(g)
                run_prop = all_goals_prop[rep][run]
                run_auc = all_goals_perf[g][rep][run]
                if run_prop <= lim1:  # below
                    prop1[gidx].append(run_prop)
                    auc1[gidx].append(run_auc)
                if run_prop >= lim2:  # above
                    prop2[gidx].append(run_prop)
                    auc2[gidx].append(run_auc)
    prop1 = np.array(prop1)
    prop2 = np.array(prop2)
    auc1 = np.array(auc1)
    auc2 = np.array(auc2)
    avg1 = np.mean(auc1, axis=1)
    avg2 = np.mean(auc2, axis=1)
    std1 = auc1.std(axis=1)
    std2 = auc2.std(axis=1)
    ste1 = std1 / np.sqrt(auc1.shape[1])
    ste2 = std2 / np.sqrt(auc2.shape[1])

    plt.figure(figsize=figsize)
    # for gidx in range(len(auc2)):
    # plt.scatter([gidx]*len(auc2[gidx]), auc2[gidx], c=prop2[gidx], cmap="Oranges", alpha=0.3, s=1, vmin=-1, vmax=3) #set a larger vmax, this colormap assigns too dark color for large values
    # plt.scatter([gidx]*len(auc2[gidx]), auc2[gidx], c="C1", alpha=0.2, s=3)
    # plt.fill_between(range(len(avg2)), avg2-ste2, avg2+ste2, alpha=0.3, color="C1")
    plt.fill_between(range(len(avg2)), avg2 - std2, avg2 + std2, alpha=0.3, color="C1")
    # for gidx in range(len(auc1)):
    # plt.scatter([gidx]*len(auc1[gidx]), auc1[gidx], c=prop1[gidx], cmap="Blues", alpha=0.6, s=1, vmin=-1, vmax=1)
    # plt.scatter([gidx]*len(auc1[gidx]), auc1[gidx], c="C0", alpha=0.2, s=3)
    # plt.fill_between(range(len(avg1)), avg1-ste1, avg1+ste1, alpha=0.3, color="C0")
    plt.fill_between(range(len(avg1)), avg1 - std1, avg1 + std1, alpha=0.3, color="C0")

    plt.plot(avg1, color="C0", label="Below {}th perc".format(classify[0]))
    plt.plot(avg2, color="C1", label="Above {}th perc".format(classify[1]))

    xticks_pos = list(range(0, ordered_goal_ranks[-1] + 1, 25))
    xticks_labels = list(range(0, ordered_goal_ranks[-1] + 1, 25))
    plt.xticks(xticks_pos, xticks_labels, rotation=60, fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if xy_label:
        plt.xlabel('Goal Ranks')
        plt.ylabel('AUC')
    if legend:
        plt.legend()
    plt.savefig("plot/img/{}_{}.pdf".format(title, property_key), dpi=300, bbox_inches='tight')
    print("Save in plot/img/{}_{}".format(title, property_key))
    # plt.show()


def property_accumulate(property_key, all_paths_dict, goal_ids, title,
                        total_param=None, xlim=[], prop_lim=None, ylim=[], figsize=(8, 6), xy_label=True, legend=True,
                        yticks=[], xticks=[],
                        pair_prop=None, highlight=None, group_color=False, given_ax=None, ax_below=None,
                        show_only=None,
                        rank_filter=None, ranks=None):
    labels = [i["label"] for i in all_paths_dict]
    pklfile = "plot/temp_data/all_goals_auc_{}.pkl".format(property_key)
    if os.path.isfile(pklfile):
        with open(pklfile, "rb") as f:
            all_goals_auc = pickle.load(f)
        print("Load from {}".format(pklfile))
    else:
        all_goals_auc = pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels)
        with open(pklfile, "wb") as f:
            pickle.dump(all_goals_auc, f)

    formated_path = {}
    for goal in goal_ids:
        if (rank_filter is None) or (rank_filter is not None and ranks[goal] in rank_filter):
            g_path = copy.deepcopy(all_paths_dict)
            for i in range(len(all_paths_dict)):
                label = g_path[i]["label"]
                best_param_folder = all_goals_auc[goal][label][1]
                best = int(best_param_folder.split("_")[0])
                g_path[i]["control"] = [g_path[i]["control"].format(goal), best]
            formated_path[goal] = g_path

    prop_log = []
    rep_idx = []
    properties, _ = load_property([formated_path[list(formated_path.keys())[0]]], property_key=property_key,
                                  early_stopped=True, fix_rep=True)

    all_goals_perf = {}
    for rep in properties:
        for run in properties[rep]:
            prop_log.append(properties[rep][run])
            rep_idx.append((rep, run))
    for goal in goal_ids:
        if (rank_filter is None) or (rank_filter is not None and ranks[goal] in rank_filter):
            transf_perf, _ = load_property([formated_path[goal]], property_key="return", early_stopped=True)
            all_goals_perf[goal] = transf_perf

    accumulate_perf = np.zeros(len(rep_idx))
    accumulate_label = []
    for i, idx in enumerate(rep_idx):
        temp = []
        rep, run = idx
        for goal in goal_ids:
            if (rank_filter is None) or (rank_filter is not None and ranks[goal] in rank_filter):
                temp.append(all_goals_perf[goal][rep][run])
        accumulate_perf[i] = np.array(temp).mean()
        accumulate_label.append(rep.split("_0")[0])

    prop_log = np.array(prop_log)
    ranks = prop_log.argsort()
    ranked_prop = prop_log[ranks]
    ranked_perf = accumulate_perf[ranks]

    if given_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = given_ax

    # smoothed = exp_smooth(ranked_perf, 0.4)
    # ax.plot(ranked_prop, smoothed, c="C0")
    # ax.text(0,0, "{}-{}".format(np.argmax(smoothed), np.max(smoothed)))

    # reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    # init = [1.0, 1e-3]
    # reg.set_params(alpha_init=init[0], lambda_init=init[1])
    # reg.fit(ranked_prop.reshape(-1, 1), ranked_perf.reshape(-1, 1))
    # ymean, ystd = reg.predict(ranked_prop.reshape(-1, 1), return_std=True)
    # ax.plot(ranked_prop, ymean.reshape(-1), c="C0")

    # z = np.polynomial.polynomial.polyfit(ranked_prop, ranked_perf, 2)
    # p = np.poly1d(z)
    # ax.plot(ranked_prop,p(ranked_prop), c="C0")

    if not group_color:
        ax.scatter(ranked_prop, ranked_perf, c="C0")
    else:
        plotted_y = []
        plotted_x = []
        plotted_c = []
        for i, p in enumerate(ranked_perf):
            lb = accumulate_label[ranks[i]]
            if lb == "ReLU" or lb.split("+")[0] == "ReLU":
                c = "#e74c3c"
                if show_only is not None and show_only == "ReLU":
                    ax.scatter([ranked_prop[i]], [p], c=c)
                    plotted_x.append(ranked_prop[i])
                    plotted_y.append(p)
                    plotted_c.append(c)
            elif lb == "ReLU(L)" or lb.split("+")[0] == "ReLU(L)":
                c = "#9b59b6"
                if show_only is not None and show_only == "ReLU(L)":
                    ax.scatter([ranked_prop[i]], [p], c=c)
                    plotted_x.append(ranked_prop[i])
                    plotted_y.append(p)
                    plotted_c.append(c)
            elif lb == "FTA" or lb.split("+")[0] == "FTA" or lb.split(" ")[0] == "FTA":
                c = "#2980b9"
                if show_only is not None and show_only == "FTA":
                    ax.scatter([ranked_prop[i]], [p], c=c)
                    plotted_x.append(ranked_prop[i])
                    plotted_y.append(p)
                    plotted_c.append(c)
            # ax.scatter([ranked_prop[i]], [p], c=c)
    # smoothed = exp_smooth(np.array(plotted_y), 0.4)
    # ax.plot(plotted_x, smoothed, c="C0")

    # xticks_pos = [ranked_prop[i] for i in range(0, len(ranked_prop), 50)] + [ranked_prop[-1]]
    # xticks_labels = ["{}({:.2f})".format(i, ranked_prop[i]) for i in range(0, len(ranked_prop), 50)]+["{}({:.2f})".format(len(ranked_prop)-1, ranked_prop[-1])]
    # plt.xticks(xticks_pos, xticks_labels, rotation=90)

    # if highlight is not None:
    # ax.scatter([ranked_prop[highlight]], [ranked_perf[highlight]], c="C1")
    # ax.vlines([ranked_prop[highlight]], 2, [ranked_perf[highlight]], ls=":", colors="C1", alpha=1, linewidth=3)
    # xticks_pos = [ranked_prop[0], ranked_prop[highlight]]
    # xticks_labels = ["{}({:.2f})".format(0, ranked_prop[0]), "{}({:.2f})".format(highlight, ranked_prop[highlight])]
    # ax.set_xticks(xticks_pos, xticks_labels, rotation=60)

    plotted_x = np.array(plotted_x)
    plotted_y = np.array(plotted_y)

    idx = np.argsort(plotted_y)[-3:]
    top3_x = plotted_x[idx]
    top3_y = plotted_y[idx]
    ax.scatter(top3_x, top3_y, marker="*", c="#27ae60", s=100)
    ax.set_xticks([], [])
    ax_below.set_xticks([], [])
    if xticks != []:
        if xticks == "min-max":
            ax_below.set_xticks([np.min(ranked_prop), np.max(ranked_prop)],
                                ["{:.2f}".format(np.min(ranked_prop)), "{:.2f}".format(np.max(ranked_prop))],
                                rotation=60)

    # kernel = scipy.stats.gaussian_kde(plotted_x.reshape(1, -1))
    # dens_x = np.linspace(plotted_x.min(), plotted_x.max(), num=100)
    # density = kernel(dens_x)
    # ax_below.plot(dens_x, density, c=plotted_c[-1])

    dens_x = np.linspace(ranked_prop.min(), ranked_prop.max(), num=30)
    density = np.zeros(len(dens_x) - 1)
    stars = []
    for idx in range(len(dens_x) - 1):
        count = 0
        for x in plotted_x:
            if x > dens_x[idx] and x <= dens_x[idx + 1]:
                count += 1
                if x in top3_x:
                    stars.append(idx)
        density[idx] = count
    bin_w = (ranked_prop.max() - ranked_prop.min()) / len(density)
    ax_below.bar(dens_x[:-1] + 0.5 * bin_w, density, color=plotted_c[-1], width=bin_w)
    ax_below.bar(dens_x[stars] + 0.5 * bin_w, density[stars], color="#27ae60", width=bin_w)

    ax_below.spines['top'].set_visible(False)
    ax_below.spines['right'].set_visible(False)
    ax_below.spines['left'].set_visible(False)
    ax_below.set_yticks([], [])

    ax.set_xlim(ranked_prop.min() - 0.01, ranked_prop.max() + 0.01)
    ax_below.set_xlim(ranked_prop.min() - 0.01, ranked_prop.max() + 0.01)
    if prop_lim:
        ax.set_xlim(prop_lim[0], prop_lim[1])
        ax_below.set_xlim(prop_lim[0], prop_lim[1])
        if xticks == "min-max":
            ax_below.set_xticks([prop_lim[0], prop_lim[1]],
                                ["{:.2f}".format(prop_lim[0]), "{:.2f}".format(prop_lim[1])],
                                rotation=60)
    if ylim != []:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_yticks(yticks)

    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax_below.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)

    if pair_prop:
        pair_properties, _ = load_property([formated_path[goal_ids[0]]], property_key=pair_prop, early_stopped=True)
        pair_prop_log = np.zeros(len(rep_idx))
        for i, idx in enumerate(rep_idx):
            rep, run = idx
            pair_prop_log[i] = pair_properties[rep][run]
        ranked_pairp = pair_prop_log[ranks]

        ax2 = ax.twinx()
        ax2.plot(ranked_prop, ranked_pairp, c="C1")

    if given_ax is None:
        # plt.title(property_keys[property_key], fontsize=30)
        plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
        # plt.show()


def pair_prop_corr(property_keys, all_paths_dict):
    def load_prop_pair(prop_pair, paths):
        all_goals_prop = {}
        for pk in prop_pair:
            properties, _ = load_property([paths], property_key=pk, early_stopped=True)
            all_goals_prop[pk] = properties

        prop1 = []
        prop2 = []

        p1 = prop_pair[0]
        p2 = prop_pair[1]

        for rep in all_goals_prop[p1].keys():
            for run in all_goals_prop[p1][rep].keys():
                prop1.append(all_goals_prop[p1][rep][run])
                prop2.append(all_goals_prop[p2][rep][run])
        return prop1, prop2

    labels = [i["label"] for i in all_paths_dict]

    all_pairs = list(itertools.combinations(property_keys.keys(), 2))
    ncol = 3  # int(np.sqrt(len(all_pairs)))+1
    nrow = int(np.ceil(len(all_pairs)))  # int(np.sqrt(len(all_pairs)))+1
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4 * ncol, 4 * nrow))
    fig.tight_layout(h_pad=6, w_pad=2)
    for i, key_pair in enumerate(all_pairs):
        prop1, prop2 = load_prop_pair(key_pair, all_paths_dict)
        cor = np.corrcoef(prop1, prop2)[0][1]
        axes[i // ncol][i % ncol].scatter(prop1, prop2)
        axes[i // ncol][i % ncol].set_xlabel(property_keys[key_pair[0]], fontsize=30)
        axes[i // ncol][i % ncol].set_ylabel(property_keys[key_pair[1]], fontsize=30)
        axes[i // ncol][i % ncol].set_title(
            "y={}\nx={}\ncor={:.3f}".format(property_keys[key_pair[0]], property_keys[key_pair[1]], cor), fontsize=30)

    plt.savefig("plot/img/prop-cor.png", dpi=300, bbox_inches='tight')


def main():
    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1
    goal_ids = [
        106,
        107, 108, 118, 119, 109, 120, 121, 128, 110, 111, 122, 123, 129, 130, 142, 143, 144, 141, 140,
        139, 138, 156, 157, 158, 155, 170, 171, 172, 169, 154, 168, 153, 167, 152, 166, 137, 151, 165,
        127, 117, 105, 99, 136, 126, 150, 164, 116, 104, 86, 98, 85, 84, 87, 83, 88, 89, 97, 90, 103,
        115, 91, 92, 93, 82, 96, 102, 114, 81, 80, 71, 95, 70, 69, 68, 101, 67, 66, 113, 62, 65, 125,
        61, 132, 47, 133, 79, 48, 72, 94, 52, 146, 73, 49, 33, 100, 134, 34, 74, 50, 147, 35, 112, 75,
        135, 160, 36, 148, 76, 161, 63, 77, 124, 149, 78, 162, 163, 131, 53, 145, 159, 54, 55, 56, 57,
        58, 59, 64, 51, 38, 60, 39, 40, 46, 41, 42, 43, 44, 45, 32, 31, 37, 30, 22, 21, 23, 20, 24, 19,
        25, 18, 26, 17, 16, 27, 15, 28, 29, 7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0,
    ]

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        "ReLU+ATC",
        "ReLU(L)",
        "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward",
        "ReLU(L)+SF", "ReLU(L)+ATC",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
    ]
    property_keys.pop("return")
    for key_pair in itertools.combinations(property_keys.keys(), 2):
        pair = {}
        pair[key_pair[0]] = property_keys[key_pair[0]]
        pair[key_pair[1]] = property_keys[key_pair[1]]
        pair_property(pair, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), [106], ranks, xlim=[0, 11], with_auc=False)

    groups = {
        "ReLU": ["ReLU", "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward",
                 "ReLU+SF", "ReLU+ATC"],
        "ReLU(L)": ["ReLU(L)", "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder",
                    "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF", "ReLU(L)+ATC"],
        "FTA": ["FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8", "FTA+VirtualVF1", "FTA+VirtualVF5",
                "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC"],
    }
    # property_keys.pop("return")
    # property_keys.pop("sparsity")
    # property_keys.pop("interf")
    # property_keys.pop("distance")
    # property_scatter(property_keys, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-activation")
    # performance_scatter(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-auc-activation",
    #                     goal_ids, xlim=[0, 11], ylim=[4, 10])
    groups = {
        "No Aux": ["ReLU"],
        "XY": ["ReLU+XY"],
        "IR": ["ReLU+Decoder"],
        "NAS": ["ReLU+NAS"],
        "Reward": ["ReLU+Reward"],
        "SF": ["ReLU+SF"],
        "VirtualVF1": ["ReLU+VirtualVF1"],
        "VirtualVF5": ["ReLU+VirtualVF5"],
        "ATC": ["ReLU+ATC"],
    }
    # property_scatter(property_keys, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-aux-relu")
    # performance_scatter(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-auc-aux-relu",
    #                     goal_ids, xlim=[0, 11], ylim=[4, 10])
    # performance_scatter_ste(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-auc-aux-relu_ci",
    #                     goal_ids, xlim=[0, 11], ylim=[4, 10])
    groups = {
        "No Aux": ["ReLU(L)"],
        "XY": ["ReLU(L)+XY"],
        "IR": ["ReLU(L)+Decoder"],
        "NAS": ["ReLU(L)+NAS"],
        "Reward": ["ReLU(L)+Reward"],
        "SF": ["ReLU(L)+SF"],
        "VirtualVF1": ["ReLU(L)+VirtualVF1"],
        "VirtualVF5": ["ReLU(L)+VirtualVF5"],
        "ATC": ["ReLU(L)+ATC"],
    }
    # property_scatter(property_keys, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-aux-relu(l)")
    # performance_scatter(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-auc-aux-relu(l)",
    #                     goal_ids, xlim=[0, 11], ylim=[4, 10])
    # performance_scatter_ste(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-auc-aux-relu(l)_ci",
    #                     goal_ids, xlim=[0, 11], ylim=[4, 10])
    groups = {
        "No Aux": ["FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8"],
        "XY": ["FTA+XY"],
        "IR": ["FTA+Decoder"],
        "NAS": ["FTA+NAS"],
        "Reward": ["FTA+Reward"],
        "SF": ["FTA+SF"],
        "VirtualVF1": ["FTA+VirtualVF1"],
        "VirtualVF5": ["FTA+VirtualVF5"],
        "ATC": ["FTA+ATC"],
    }
    # property_scatter(property_keys, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-aux-fta")
    # performance_scatter(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-auc-aux-fta",
    #                     goal_ids, xlim=[0, 11], ylim=[4, 10])
    # performance_scatter_ste(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-auc-aux-fta_ci",
    #                     goal_ids, xlim=[0, 11], ylim=[4, 10])

    property_keys.pop("return", None)
    # property_keys.pop("sparsity")
    # property_keys.pop("interf")
    # property_keys.pop("distance")
    highlight_idxs = {"lipschitz": 95,  # 72,
                      "distance": 149,
                      "ortho": 89,  # 96,
                      "interf": 1,
                      "diversity": 102,  # 90,
                      "sparsity": 134, }
    # # for key in property_keys.keys():
    # #     property_auc_goal(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13), goal_ids, ranks, "nonlinear/scatter/slice",
    # #                               xlim=[0, 11], classify=[50, 50])
    # targets = [
    #     "ReLU",
    #     "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
    # ]
    # targets = [
    #     "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
    #     "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",
    # ]
    # highlight_idxs = {"lipschitz": 10,
    #                   "distance": 59,
    #                   "ortho": 4,
    #                   "interf": 0,
    #                   "diversity": 5,
    #                   "sparsity": 13,}
    titles = {
        "lipschitz": "Complexity\nReduction",
        "distance": "Dynamics\nAwareness",
        "ortho": "Orthogonality",
        "interf": "Non-\ninterference",
        "diversity": "Diversity",
        "sparsity": "Sparsity",
    }

    # fig, axs = plt.subplots(6, 6, figsize=(28, 36))
    #
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     # property_accumulate(key, label_filter(targets, gh_transfer_sweep_v13), goal_ids, "linear/accumAUC/accum_{}".format(key),
    #     #                     xlim=[0, 11])
    #     yticks = [2, 6, 10] if i == 0 else []
    #     xticks = []
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_fta_{}".format(key),
    #                         xlim=[0, 11],
    #                         ylim=[2, 10],
    #                         group_color=True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax=axs[0,i],
    #                         show_only="FTA"
    #                         )
    #     axs[0,i].set_title(titles[key], fontsize=30)
    #
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     yticks = [2, 6, 10] if i == 0 else []
    #     xticks = "min-max"  # []#
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_relu_{}".format(key),
    #                         xlim=[0, 11],
    #                         ylim=[2, 10],
    #                         group_color=True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax=axs[1,i],
    #                         show_only="ReLU"  # "FTA"#
    #                         )
    #     axs[1,i].set_title(titles[key], fontsize=30)
    #
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     # print(ranks)
    #     yticks = [2, 6, 10] if i == 0 else []
    #     xticks = []
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_fta_{}".format(key),
    #                         xlim=[0, 11],
    #                         ylim=[2, 10],
    #                         group_color=True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax=axs[2,i],
    #                         show_only="FTA",
    #                         rank_filter=list(range(50)),
    #                         ranks=ranks
    #                         )
    #     axs[2,i].set_title(titles[key], fontsize=30)
    #
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     yticks = [2, 6, 10] if i == 0 else []
    #     xticks = "min-max"  # []#
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_relu_{}".format(key),
    #                         xlim=[0, 11],
    #                         ylim=[2, 10],
    #                         group_color=True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax=axs[3,i],
    #                         show_only="ReLU",
    #                         rank_filter=list(range(50)),
    #                         ranks=ranks
    #                         )
    #     axs[3,i].set_title(titles[key], fontsize=30)
    #
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     # print(ranks)
    #     yticks = [2, 6, 10] if i == 0 else []
    #     xticks = []
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_fta_{}".format(key),
    #                         xlim=[0, 11],
    #                         ylim=[2, 10],
    #                         group_color=True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax=axs[4,i],
    #                         show_only="FTA",
    #                         rank_filter=list(range(125, 200)),
    #                         ranks=ranks
    #                         )
    #     axs[4,i].set_title(titles[key], fontsize=30)
    #
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     yticks = [2, 6, 10] if i == 0 else []
    #     xticks = "min-max"  # []#
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_relu_{}".format(key),
    #                         xlim=[0, 11],
    #                         ylim=[2, 10],
    #                         group_color=True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax=axs[5,i],
    #                         show_only="ReLU",
    #                         rank_filter=list(range(125, 200)),
    #                         ranks=ranks
    #                         )
    #     axs[5,i].set_title(titles[key], fontsize=30)
    # fig.tight_layout()
    # plt.savefig("plot/img/nonlinear/accum_all.pdf", dpi=300, bbox_inches='tight')

    
    fig, axs = plt.subplots(4, 6, figsize=(28, 8), gridspec_kw={'height_ratios': [4, 1, 4, 1]})
    for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho", "sparsity", "interf"]):
        yticks = [5, 7.5, 10] if i == 0 else []
        xticks = []
        property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids,
                            "nonlinear/accumAUC/accum_fta_{}".format(key),
                            xlim=[0, 11],
                            ylim=[5, 10],
                            group_color=True,
                            yticks=yticks,
                            xticks=xticks,
                            highlight=highlight_idxs[key],
                            given_ax=axs[0, i], ax_below=axs[1, i],
                            show_only="FTA"
                            )
        axs[0, i].set_title(titles[key], fontsize=30)
    for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho", "sparsity", "interf"]):
        yticks = [5, 7.5, 10] if i == 0 else []
        xticks = "min-max"  # []#
        property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids,
                            "nonlinear/accumAUC/accum_relu_{}".format(key),
                            xlim=[0, 11],
                            ylim=[5, 10],
                            group_color=True,
                            yticks=yticks,
                            xticks=xticks,
                            highlight=highlight_idxs[key],
                            given_ax=axs[2, i], ax_below=axs[3, i],
                            show_only="ReLU"  # "FTA"#
                            )
        # axs[2, i].set_title(titles[key], fontsize=30)
    plt.savefig("plot/img/nonlinear/accum_all.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()


    # fig, axs = plt.subplots(1, 6, figsize=(28, 4))
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     # print(ranks)
    #     yticks = [7, 8.5, 10] if i == 0 else []
    #     xticks = []
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_fta_{}".format(key),
    #                         xlim = [0, 11],
    #                         ylim = [7, 10],
    #                         group_color = True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax = axs[i],
    #                         show_only = "FTA",
    #                         rank_filter=list(range(50)),
    #                         ranks=ranks
    #                         )
    #     axs[i].set_title(titles[key], fontsize=30)
    # plt.savefig("plot/img/nonlinear/accum_fta_similar.pdf", dpi=300, bbox_inches='tight')
    #
    # fig, axs = plt.subplots(1, 6, figsize=(28, 4))
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     # print(ranks)
    #     yticks = [2, 6, 10] if i == 0 else []
    #     xticks = []
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_fta_{}".format(key),
    #                         xlim=[0, 11],
    #                         ylim=[2, 10],
    #                         group_color=True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax = axs[i],
    #                         show_only = "FTA",
    #                         rank_filter=list(range(125, 200)),
    #                         ranks=ranks
    #                         )
    #     axs[i].set_title(titles[key], fontsize=30)
    # plt.savefig("plot/img/nonlinear/accum_fta_dissimilar.pdf", dpi=300, bbox_inches='tight')
    #
    # fig, axs = plt.subplots(1, 6, figsize=(28, 4))
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     yticks = [7, 8.5, 10] if i == 0 else []
    #     xticks = "min-max"#[]#
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_relu_{}".format(key),
    #                         xlim = [0, 11],
    #                         ylim = [7, 10],
    #                         group_color = True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax = axs[i],
    #                         show_only = "ReLU",
    #                         rank_filter = list(range(50)),
    #                         ranks = ranks
    #                         )
    #     axs[i].set_title(titles[key], fontsize=30)
    # plt.savefig("plot/img/nonlinear/accum_relu_similar.pdf", dpi=300, bbox_inches='tight')
    #
    # fig, axs = plt.subplots(1, 6, figsize=(28, 4))
    # for i, key in enumerate(["lipschitz", "diversity", "ortho", "distance", "interf", "sparsity"]):
    #     yticks = [2, 6, 10] if i == 0 else []
    #     xticks = "min-max"#[]#
    #     property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_relu_{}".format(key),
    #                         xlim=[0, 11],
    #                         ylim=[2, 10],
    #                         group_color=True,
    #                         yticks=yticks,
    #                         xticks=xticks,
    #                         highlight=highlight_idxs[key],
    #                         given_ax = axs[i],
    #                         show_only = "ReLU",
    #                         rank_filter = list(range(125, 200)),
    #                         ranks = ranks
    #                         )
    #     axs[i].set_title(titles[key], fontsize=30)
    # plt.savefig("plot/img/nonlinear/accum_relu_dissimilar.pdf", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()