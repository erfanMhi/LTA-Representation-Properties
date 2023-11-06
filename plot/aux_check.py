import sys
sys.path.insert(0, '..')
from plot.plot_paths import *
from plot.plot_utils import *
from plot.factor_analysis import property_accumulate
# from plot.curves_sweep import performance_change
# os.chdir(os.getcwd()+"/LTA-Representation-Properties/")
# print("Change dir to", os.getcwd())

def performance_change_single_run(all_paths_dict, goal_ids, ranks, title, total_param=None, xlim=[], ylim=[], smooth=1.0, top_runs=[0, 1.0],
                       xy_label=True, data_label=True, linewidth=1, figsize=(8, 6), emphasize=None,
                       color_by_activation=False, in_plot_label=None):
    labels = [i["label"] for i in all_paths_dict]
    all_ranks = []
    for goal in goal_ids:
        all_ranks.append(ranks[goal])
    all_ranks = np.array(all_ranks)
    
    _, all_goals_independent = pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels, get_each_run=True)
    filtered_labels = labels
    
    curves = {}
    for label in filtered_labels:
        ranked_auc = np.zeros((all_ranks.max() + 1, len(all_goals_independent[106][label]))) * np.inf
        ranked_goal = np.zeros(all_ranks.max() + 1) * np.inf
        for goal in goal_ids:
            rank = ranks[goal]
            # print(rank, goal, label, all_goals_independent[goal][label])
            ranked_auc[rank] = all_goals_independent[goal][label].sum(axis=1)
            ranked_goal[rank] = goal
        for run in range(len(all_goals_independent[106][label])):
            ranked_auc[:, run] = exp_smooth(ranked_auc[:, run], smooth)
        curves[label] = ranked_auc
    
    # with open('temp.pkl', 'wb') as handle:
    #     pickle.dump(curves, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    plt.figure(figsize=figsize)
    avg_decrease = {"FTA": [], "ReLU": [], "ReLU(L)": []}
    for label in filtered_labels:
        if emphasize and (label not in emphasize):
            # plt.plot(curves[label], color=violin_colors[label], linestyle=curve_styles[label], label=label, linewidth=1, alpha=0.3)
            plt.plot(curves[label], color=violin_colors["Other rep"], linestyle=curve_styles[label], label=label, linewidth=1, alpha=0.3)
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
            plt.plot(curves[label], color=violin_colors[label], linestyle=curve_styles[label], label=label, linewidth=linewidth, zorder=100)
        print("Label, first point {}, last point {}, decrease {}".format(label, curves[label][0], curves[label][-1], curves[label][-1] - curves[label][0]))
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

titles = {
    "lipschitz": "Complexity\nReduction",
    "distance": "Dynamic\nAwareness",
    "ortho": "Orthogonality",
    "interf": "Non-\ninterference",
    "diversity": "Diversity",
    "sparsity": "Sparsity",
}

prop_lim = {
    "lipschitz": [0, 1],
    "distance": [0.32, 0.83],
    "ortho": [0, 0.75],
    "interf": [0, 1],
    "diversity": [0.05, 0.67],
    "sparsity": [0.47, 0.9],
}

targets = [
    "FTA eta=0.2",
]
fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho",]): # "sparsity", "interf"]):
    yticks = [5, 7.5, 10] if i == 0 else []
    xticks = "min-max"
    property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "",
                        xlim=[0, 11],
                        ylim=[5, 10],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="FTA"
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/nonlinear_fta_eta0.2.png", dpi=300, bbox_inches='tight')


targets = [
    "FTA+VirtualVF5",
]
fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho",]): # "sparsity", "interf"]):
    yticks = [5, 7.5, 10] if i == 0 else []
    xticks = "min-max"
    property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "",
                        xlim=[0, 11],
                        ylim=[5, 10],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="FTA"
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/nonlinear_fta_virtualvf5.png", dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho",]): # "sparsity", #"interf"]):
    yticks = [0, 2.5, 5] if i == 0 else []
    xticks = "min-max"
    property_accumulate(key, label_filter(targets, gh_transfer_sweep_v13), goal_ids, "",
                        xlim=[0, 11],
                        ylim=[0, 5],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="FTA"
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/linear_fta_virtualvf5.png", dpi=300, bbox_inches='tight')


targets = [
    "ReLU",
]
fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho",]): # "sparsity", "interf"]):
    yticks = [5, 7.5, 10] if i == 0 else []
    xticks = "min-max"  # []#
    property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_relu_{}".format(key),
                        xlim=[0, 11],
                        ylim=[5, 10],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="ReLU"  # "FTA"#
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/nonlinear_relu.png", dpi=300, bbox_inches='tight')

targets = [
    "ReLU+VirtualVF5",
]
fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho",]): # "sparsity", "interf"]):
    yticks = [5, 7.5, 10] if i == 0 else []
    xticks = "min-max"  # []#
    property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "nonlinear/accumAUC/accum_relu_{}".format(key),
                        xlim=[0, 11],
                        ylim=[5, 10],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="ReLU"  # "FTA"#
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/nonlinear_relu_virtualvf5.png", dpi=300, bbox_inches='tight')

targets = [
    "FTA+SF",
]

fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho"]): #, "sparsity", #"interf"]):
    yticks = [5, 7.5, 10] if i == 0 else []
    xticks = "min-max"
    property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "",
                        xlim=[0, 11],
                        ylim=[5, 10],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="FTA"
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/nonlinear_fta_sf.png", dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho",]): # "sparsity", # "interf"]):
    yticks = [0, 2.5, 5] if i == 0 else []
    xticks = "min-max"
    property_accumulate(key, label_filter(targets, gh_transfer_sweep_v13), goal_ids, "",
                        xlim=[0, 11],
                        ylim=[0, 5],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="FTA"
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/linear_fta_sf.png", dpi=300, bbox_inches='tight')

targets = [
    "FTA+NAS",
]

fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho", ]): #"sparsity", #"interf"]):
    yticks = [5, 7.5, 10] if i == 0 else []
    xticks = "min-max"
    property_accumulate(key, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), goal_ids, "",
                        xlim=[0, 11],
                        ylim=[5, 10],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="FTA"
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/nonlinear_fta_nas.png", dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(2, 6, figsize=(28, 4), gridspec_kw={'height_ratios': [4, 1]})
for i, key in enumerate(["lipschitz", "distance", "diversity", "ortho", ]): #"sparsity", # "interf"]):
    yticks = [0, 2.5, 5] if i == 0 else []
    xticks = "min-max"
    property_accumulate(key, label_filter(targets, gh_transfer_sweep_v13), goal_ids, "",
                        xlim=[0, 11],
                        ylim=[0, 5],
                        prop_lim=prop_lim[key],
                        group_color=True,
                        yticks=yticks,
                        xticks=xticks,
                        given_ax=axs[0, i], ax_below=axs[1, i],
                        show_only="FTA"
                        )
    axs[0, i].set_title(titles[key], fontsize=30)
plt.savefig("plot/img/aux_prop/linear_fta_nas.png", dpi=300, bbox_inches='tight')

