import sys
import copy
import numpy as np
import itertools
from sklearn import ensemble, metrics
# from sklearn.inspection import permutation_importance

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
        ax.bar(x-0.45+width*(j+1), ordered_corr, width, label="Rank={}".format(all_ranks[j]), color=cmap(j, 4))
        for i in range(len(x)):
            ax.text(x[i]-0.45+width*(0.6+j), max(0, ordered_corr[i])+label_pos, "{:.4f}".format(ordered_corr[i]), color=cmap(j, 4), fontsize=fontsize, rotation=rotation)

    ax.plot([x[0]-0.45, x[-1]+0.45], [0, 0], "--", color="grey")

    ax.legend(loc=4)
    ax.set_ylabel('Correlation')
    ax.set_ylim(-1, 1.2)
    ax.set_xticks(x)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticklabels(ordered_labels, rotation=30)
    plt.savefig("plot/img/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

def xgboost_analysis(property_keys, all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], smooth=1.0, top_runs=1.0):
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
    features = np.array(features).transpose() # [n_samples, m_features]

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
    plt.xticks(list(range(0, len(ordered_goal_ids), 25)), [ordered_goal_ranks[i] for i in list(range(0, len(ordered_goal_ids), 25))], rotation=60)
    plt.title('Feature Importance')
    plt.legend()
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')


def pair_property(property_keys, all_paths_dict, goal_ids, ranks, total_param=None, xlim=[]):

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

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for g in set(goal):
    #     prop1_g, prop2_g, auc_g = [], [], []
    #     for i in range(len(auc)):
    #         if goal[i] == g:
    #             prop1_g.append(prop1[i])
    #             prop2_g.append(prop2[i])
    #             auc_g.append(auc[i])
    #     ax.scatter(prop1_g, prop2_g, auc_g)
    # ax.set_xlabel(p1)
    # ax.set_ylabel(p2)
    # ax.set_zlabel('AUC')
    # plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=len(goal_ids), figsize=(6*len(goal_ids), 4))
    for gi, g in enumerate(goal_ids):
        prop1_g, prop2_g, auc_g = [], [], []
        for i in range(len(auc)):
            if goal[i] == g:
                prop1_g.append(prop1[i])
                prop2_g.append(prop2[i])
                auc_g.append(auc[i])
        im = axes[gi].scatter(prop1_g, prop2_g, c=auc_g, cmap="Blues", vmin=0, vmax=11)
        axes[gi].set_xlabel(property_keys[p1])
        axes[gi].set_ylabel(property_keys[p2])
        axes[gi].set_title("Goal {} Rank {}".format(g, ranks[g]))
    fig.colorbar(im)
    plt.savefig("plot/img/auc_{}-{}.png".format(p1, p2), dpi=300, bbox_inches='tight')


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
    if len(groups)>1:
        plt.legend()
    plt.savefig("plot/img/{}-{}.png".format(title, p1), dpi=300, bbox_inches='tight')


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
    ncol = int(np.sqrt(len(all_pairs)))+1
    nrow = int(np.sqrt(len(all_pairs)))+1
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4*ncol, 4*nrow))
    fig.tight_layout(h_pad=6, w_pad=2)
    for i, key_pair in enumerate(all_pairs):
        prop1, prop2 = load_prop_pair(key_pair, all_paths_dict)
        cor = np.corrcoef(prop1, prop2)[0][1]
        axes[i//ncol][i%ncol].scatter(prop1, prop2)
        axes[i//ncol][i%ncol].set_xlabel(property_keys[key_pair[0]])
        axes[i//ncol][i%ncol].set_ylabel(property_keys[key_pair[1]])
        axes[i//ncol][i%ncol].set_title("{}-{}\ncor={:.3f}".format(property_keys[key_pair[0]], property_keys[key_pair[1]], cor))

    plt.savefig("plot/img/prop-cor.png", dpi=300, bbox_inches='tight')

def main():
    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1

    goal_ids = [106, 109, 155, 98, 147]
    # targets = [
    #     "ReLU",
    #     "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
    #     "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
    #     "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
    # ]
    # correlation_bar(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "maze correlation bar", xlim=[0, 11])
    # targets = [
    #     "ReLU",
    #     "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
    # ]
    # correlation_bar(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "maze correlation bar (relu)", xlim=[0, 11])
    # targets = [
    #     "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
    #     "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
    # ]
    # correlation_bar(label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "maze correlation bar (fta)", xlim=[0, 11])

    targets = [
        "ReLU",
        "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
    ]
    property_keys = {
        "lipschitz": "complexity reduction",
        "distance": "dynamics awareness",
        "ortho": "orthogonality",
        "interf":"noninterference",
        "diversity":"diversity",
        "sparsity":"sparsity"
    }
    # pair_prop_corr(property_keys, label_filter(targets, gh_transfer_sweep_v13))
    for key_pair in itertools.combinations(property_keys.keys(), 2):
        pair = {}
        pair[key_pair[0]] = property_keys[key_pair[0]]
        pair[key_pair[1]] = property_keys[key_pair[1]]
        pair_property(pair, label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, xlim=[0, 11])
    # groups = {
    #     "ReLU": ["ReLU", "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",],
    #     "FTA": ["FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8", "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",],
    # }
    # for key in property_keys.keys():
    #     property_auc_scatter({key: property_keys[key]}, label_filter(targets, gh_transfer_sweep_v13), groups, goal_ids, ranks, "auc-grouped", xlim=[0, 11])
    #
    # groups = {
    #     "All": ["ReLU", "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
    #             "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8", "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",],
    # }
    # for key in property_keys.keys():
    #     property_auc_scatter({key: property_keys[key]}, label_filter(targets, gh_transfer_sweep_v13), groups, goal_ids, ranks, "auc-all", xlim=[0, 11])

    # goal_ids = [106,
    #             107, 108, 118, 119, 109, 120, 121, 128, 110, 111, 122, 123, 129, 130, 142, 143, 144, 141, 140, 139, 138, 156, 157, 158, 155, 170, 171, 172, 169, 154, 168, 153, 167, 152, 166, 137, 151,
    #             165, 127, 117, 105, 99, 136, 126, 150, 164, 116, 104, 86, 98, 85, 84, 87, 83, 88, 89, 97, 90, 103, 115, 91, 92, 93, 82, 96, 102, 114, 81, 80, 71, 95, 70, 69, 68, 101, 67, 66, 113, 62,
    #             65, 125, 61, 132, 47, 133, 79, 48, 72, 94, 52, 146, 73, 49, 33, 100, 134, 34, 74, 50, 147, 35, 112, 75, 135, 160, 36, 148, 76, 161, 63, 77, 124, 149, 78, 162, 163, 131, 53, 145, 159,
    #             54, 55, 56, 57, 58, 59, 64, 51, 38, 60, 39, 40, 46, 41, 42, 43, 44, 45, 32, 31, 37, 30, 22, 21, 23, 20, 24, 19, 25, 18, 26, 17, 16, 27, 15, 28, 29, 7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2,
    #             13, 1, 14, 0,
    #             ]
    # xgboost_analysis(property_keys, label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "xgboost", xlim=[0, 11], smooth=0.2, , top_runs=[0.9, 1.0])
    # xgboost_analysis(property_keys, label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "xgboost top0.1", xlim=[0, 11], smooth=0.2, , top_runs=[0.9, 1.0])
    # targets = [
    #     "ReLU",
    #     "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
    # ]
    # xgboost_analysis(property_keys, label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "xgboost_relu", xlim=[0, 11], smooth=0.2)
    # xgboost_analysis(property_keys, label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "xgboost_relu top0.1", xlim=[0, 11], smooth=0.2, top_runs=[0.9, 1.0])
    # targets = [
    #     "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
    #     "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
    # ]
    # xgboost_analysis(property_keys, label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "xgboost_fta", xlim=[0, 11], smooth=0.2)
    # xgboost_analysis(property_keys, label_filter(targets, gh_transfer_sweep_v13), goal_ids, ranks, "xgboost_fta top0.1", xlim=[0, 11], smooth=0.2, top_runs=[0.9, 1.0])

main()