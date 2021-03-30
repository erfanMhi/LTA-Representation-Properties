import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

from plot.plot_utils import *
from plot.plot_paths import *
# from plot.curves_property import learning_curve_mean
os.chdir("..")
print("Change dir to", os.getcwd())

def arrange_order(dict1, dict2):
    l1, l2 = [], []
    if set(dict1.keys()) != set(dict2.keys()):
        print("Warning: Keys are different", dict1.keys(), dict2.keys())
        dk1 = list(dict1.keys()).copy()
        dk2 = list(dict2.keys()).copy()
        for k in dk1:
            if k not in dict2.keys():
                print("Pop", k, dict1.pop(k))
        for k in dk2:
            if k not in dict1.keys():
                print("Pop", k, dict2.pop(k))

    for i in dict1.keys():
        # assert not np.isnan(dict1[i]), print(i, dict1[i])
        # assert not np.isnan(dict2[i]), print(i, dict2[i])
        # v1 = 0 if np.isnan(dict1[i]) else dict1[i]
        # v2 = 0 if np.isnan(dict2[i]) else dict2[i]
        v1 = dict1[i]
        v2 = dict2[i]
        if np.isnan(v1):
            print("run {} is nan".format(i))
            v1 = 0
        if np.isnan(v2):
            print("run {} is nan".format(i))
            v2 = 0
        l1.append(v1)
        l2.append(v2)
    return l1, l2


def calculation(all_groups, title, property_key=None, perc=None, relationship=None, targets=[], early_stopped=False):
    if len(targets) > 0:
        temp = []
        for g in all_groups:
            t = []
            for i in g:
                if i["label"] in targets:
                    t.append(i)
            temp.append(t)
        all_groups = temp

    all_groups, all_group_dict = merge_groups(all_groups)
    # print(all_groups, "\n\n", all_group_dict, "\n")
    reverse = True if property_key in ["lipschitz", "interf"] else False # normalize interference and lipschitz, for non-interference and complexity reduction measure
    model_saving = load_info(all_group_dict, 0, "model", path_key="online_measure") if early_stopped else None
    properties = load_online_property(all_group_dict, property_key, reverse=reverse, cut_at_step=model_saving)
    control = load_online_property(all_group_dict, "return")
    labels = [i["label"] for i in all_group_dict]

    # all reps:
    all_control = []
    all_property = []
    all_color = []
    all_marker = []
    indexs = [0]
    for idx in range(len(labels)):
        rep = labels[idx]
        assert len(rep.split("_")) == 2
        old_rep = rep.split("_")[0]
        color = violin_colors[old_rep]
        marker = marker_styles[old_rep]
        # print("---------------", rep, control[rep], properties[rep])
        ctr, prop = arrange_order(control[rep], properties[rep])
        all_control += ctr
        all_property += prop
        all_color.append(color)
        all_marker.append(marker)
        assert len(ctr) == len(prop)
        indexs.append(len(ctr)+indexs[-1])

    all_control = np.array(all_control)
    all_property = np.array(all_property)
    all_control, all_property = perform_filter(all_control, all_property, perc=perc)
    all_control, all_property = perform_transformer(all_control, all_property, relationship=relationship)
    cor = np.corrcoef(all_control, all_property)[0][1]
    # print(len(all_control))
    print("All reps: {} correlation = {:.4f}".format(property_key, cor))

    # plt.figure()
    # for i in range(len(indexs)-1):
    #     plt.scatter(all_control[indexs[i]: indexs[i+1]], all_property[indexs[i]: indexs[i+1]], c=all_color[i], s=5, marker=all_marker[i])
    # plt.xlabel("AUC")
    # plt.ylabel("Property measure")
    # # plt.savefig("plot/img/scatter_{}_trans={}.pdf".format(title, relationship), dpi=300, bbox_inches='tight')
    # plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    # plt.close()
    # plt.clf()
    # # print("save {} in plot/img/".format(title))
    return cor

def merge_groups(all_groups):
    # to make sure the labels are different after merging
    all_group_dict = []
    all_groups_changed = []
    for i in range(len(all_groups)):
        group = all_groups[i]
        changed = []
        for p in group:
            cp = p.copy()
            cp["label"] += "_"+str(i)
            changed.append(cp)
        all_group_dict += changed
        all_groups_changed.append(changed)
    return all_groups_changed, all_group_dict

def perform_filter(controls, properties, perc):
    if perc==[0, 1] or perc is None:
        return controls, properties
    idx_sort = sorted(range(len(controls)), key=lambda k: controls[k]) # from low to high
    target = idx_sort[int(perc[0] * len(idx_sort)): int(perc[1] * len(idx_sort))]
    ctarget, ptarget = controls[target], properties[target]
    return ctarget, ptarget

def perform_transformer(controls, properties, relationship):
    if relationship is None:
        return controls, properties
    elif relationship == "log":
        properties = np.log((properties+1)*100)
        return controls, properties
    elif relationship == "sqrt":
        properties = np.sqrt(properties)
        return controls, properties
    else:
        raise NotImplementedError


def simple_maze_correlation_early(perc):
    print("\nSame task")
    same_lip = calculation([gh_same_early], "maze_early_same_lip", property_key="lipschitz", perc=perc, targets=targets, early_stopped=True)
    same_dist = calculation([gh_same_early], "maze_early_same_distance", property_key="distance", perc=perc, targets=targets, early_stopped=True)
    same_ortho = calculation([gh_same_early], "maze_early_same_ortho", property_key="ortho", perc=perc, targets=targets, early_stopped=True)
    same_interf = calculation([gh_same_early], "maze_early_same_interf", property_key="interf", perc=perc, targets=targets, early_stopped=True)
    same_diversity = calculation([gh_same_early], "maze_early_same_diversity", property_key="diversity", perc=perc, targets=targets, early_stopped=True)
    same_spars = calculation([gh_same_early], "maze_early_same_sparsity", property_key="sparsity", perc=perc, targets=targets, early_stopped=True)

    print("\nSimilar task")
    similar_lip = calculation([gh_similar_early], "maze_early_similar_lip", property_key="lipschitz", perc=perc, targets=targets, early_stopped=True)
    similar_dist = calculation([gh_similar_early], "maze_early_similar_distance", property_key="distance", perc=perc, targets=targets, early_stopped=True)
    similar_ortho = calculation([gh_similar_early], "maze_early_similar_ortho", property_key="ortho", perc=perc, targets=targets, early_stopped=True)
    similar_interf = calculation([gh_similar_early], "maze_early_similar_interf", property_key="interf", perc=perc, targets=targets, early_stopped=True)
    similar_diversity = calculation([gh_similar_early], "maze_early_similar_diversity", property_key="diversity", perc=perc, targets=targets, early_stopped=True)
    similar_spars = calculation([gh_similar_early], "maze_early_similar_sparsity", property_key="sparsity", perc=perc, targets=targets, early_stopped=True)

    print("\nDifferent task - fix")
    diff_lip = calculation([gh_diff_early], "maze_early_diff-fix_lip", property_key="lipschitz", perc=perc, targets=targets, early_stopped=True)
    diff_dist = calculation([gh_diff_early], "maze_early_diff-fix_distance", property_key="distance", perc=perc, targets=targets, early_stopped=True)
    diff_ortho = calculation([gh_diff_early], "maze_early_diff-fix_ortho", property_key="ortho", perc=perc, targets=targets, early_stopped=True)
    diff_interf = calculation([gh_diff_early], "maze_early_diff-fix_interf", property_key="interf", perc=perc, targets=targets, early_stopped=True)
    diff_diversity = calculation([gh_diff_early], "maze_early_diff-fix_diversity", property_key="diversity", perc=perc, targets=targets, early_stopped=True)
    diff_spars = calculation([gh_diff_early], "maze_early_diff-fix_sparsity", property_key="sparsity", perc=perc, targets=targets, early_stopped=True)

    print("\nDifferent task - tune")
    difftune_lip = calculation([gh_diff_tune_early], "maze_early_diff-tune_lip", property_key="lipschitz", perc=perc, targets=targets, early_stopped=True)
    difftune_dist = calculation([gh_diff_tune_early], "maze_early_diff-tune_distance", property_key="distance", perc=perc, targets=targets, early_stopped=True)
    difftune_ortho = calculation([gh_diff_tune_early], "maze_early_diff-tune_ortho", property_key="ortho", perc=perc, targets=targets, early_stopped=True)
    difftune_interf = calculation([gh_diff_tune_early], "maze_early_diff-tune_interf", property_key="interf", perc=perc, targets=targets, early_stopped=True)
    difftune_diversity = calculation([gh_diff_tune_early], "maze_early_diff-tune_diversity", property_key="diversity", perc=perc, targets=targets, early_stopped=True)
    difftune_spars = calculation([gh_diff_tune_early], "maze_early_diff-tune_sparsity", property_key="sparsity", perc=perc, targets=targets, early_stopped=True)

    labels = ["complexity reduction", "dynamics awareness", "orthogonality", "noninterference", "diversity", "sparsity"]
    same = [same_lip, same_dist, same_ortho, same_interf, same_diversity, same_spars]
    similar = [similar_lip, similar_dist, similar_ortho, similar_interf, similar_diversity, similar_spars]
    diff = [diff_lip, diff_dist, diff_ortho, diff_interf, diff_diversity, diff_spars]
    difftune = [difftune_lip, difftune_dist, difftune_ortho, difftune_interf, difftune_diversity, difftune_spars]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    label_pos = 0.03
    fontsize = 9
    rotation = 90

    fig, ax = plt.subplots()
    rects1 = ax.bar(x-0.45+width, same, width, label='same', color=cmap(0, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*0.6, max(0, same[i])+label_pos, "{:.4f}".format(same[i]), color=cmap(0, 4), fontsize=fontsize, rotation=rotation)

    rects2 = ax.bar(x-0.45+width*2, similar, width, label='similar', color=cmap(2, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*1.6, max(0, similar[i]) + label_pos, "{:.4f}".format(similar[i]), color=cmap(2, 4), fontsize=fontsize, rotation=rotation)

    rects3 = ax.bar(x-0.45+width*3, diff, width, label='dissimilar(fix)', color=cmap(1, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*2.6, max(0, diff[i]) + label_pos, "{:.4f}".format(diff[i]), color=cmap(1, 4), fontsize=fontsize, rotation=rotation)

    rects4 = ax.bar(x-0.45+width*4, difftune, width, label='dissimilar(tune)', color=cmap(3, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*3.6, max(0, difftune[i]) + label_pos, "{:.4f}".format(difftune[i]), color=cmap(3, 4), fontsize=fontsize, rotation=rotation)

    ax.plot([x[0]-0.45, x[-1]+0.45], [0, 0], "--", color="grey")


    # ax.legend()
    ax.set_ylabel('Correlation')
    ax.set_ylim(-0.5, 1)
    ax.set_xticks(x)
    # ax.set_yticks([-1, -0.7, -0.5, 0, 0.5, 0.7, 1])
    ax.set_xticklabels(labels, rotation=30)
    # plt.grid(True)
    # plt.show()
    plt.savefig("plot/img/{}.pdf".format("maze_correlation_early"), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()


def simple_maze_correlation_last(perc):
    print("\nSame task")
    same_lip = calculation([gh_same_last], "maze_last_same_lip", property_key="lipschitz", perc=perc, targets=targets)
    same_dist = calculation([gh_same_last], "maze_last_same_distance", property_key="distance", perc=perc, targets=targets)
    same_ortho = calculation([gh_same_last], "maze_last_same_ortho", property_key="ortho", perc=perc, targets=targets)
    same_interf = calculation([gh_same_last], "maze_last_same_interf", property_key="interf", perc=perc, targets=targets)
    same_diversity = calculation([gh_same_last], "maze_last_same_diversity", property_key="diversity", perc=perc, targets=targets)
    same_spars = calculation([gh_same_last], "maze_last_same_sparsity", property_key="sparsity", perc=perc, targets=targets)

    print("\nSimilar task")
    similar_lip = calculation([gh_similar_last], "maze_last_similar_lip", property_key="lipschitz", perc=perc, targets=targets)
    similar_dist = calculation([gh_similar_last], "maze_last_similar_distance", property_key="distance", perc=perc, targets=targets)
    similar_ortho = calculation([gh_similar_last], "maze_last_similar_ortho", property_key="ortho", perc=perc, targets=targets)
    similar_interf = calculation([gh_similar_last], "maze_last_similar_interf", property_key="interf", perc=perc, targets=targets)
    similar_diversity = calculation([gh_similar_last], "maze_last_similar_diversity", property_key="diversity", perc=perc, targets=targets)
    similar_spars = calculation([gh_similar_last], "maze_last_similar_sparsity", property_key="sparsity", perc=perc, targets=targets)

    print("\nDifferent task - fix")
    diff_lip = calculation([gh_diff_last], "maze_last_diff-fix_lip", property_key="lipschitz", perc=perc, targets=targets)
    diff_dist = calculation([gh_diff_last], "maze_last_diff-fix_distance", property_key="distance", perc=perc, targets=targets)
    diff_ortho = calculation([gh_diff_last], "maze_last_diff-fix_ortho", property_key="ortho", perc=perc, targets=targets)
    diff_interf = calculation([gh_diff_last], "maze_last_diff-fix_interf", property_key="interf", perc=perc, targets=targets)
    diff_diversity = calculation([gh_diff_last], "maze_last_diff-fix_diversity", property_key="diversity", perc=perc, targets=targets)
    diff_spars = calculation([gh_diff_last], "maze_last_diff-fix_sparsity", property_key="sparsity", perc=perc, targets=targets)

    print("\nDifferent task - tune")
    difftune_lip = calculation([gh_diff_tune_last], "maze_last_diff-tune_lip", property_key="lipschitz", perc=perc, targets=targets)
    difftune_dist = calculation([gh_diff_tune_last], "maze_last_diff-tune_distance", property_key="distance", perc=perc, targets=targets)
    difftune_ortho = calculation([gh_diff_tune_last], "maze_last_diff-tune_ortho", property_key="ortho", perc=perc, targets=targets)
    difftune_interf = calculation([gh_diff_tune_last], "maze_last_diff-tune_interf", property_key="interf", perc=perc, targets=targets)
    difftune_diversity = calculation([gh_diff_tune_last], "maze_last_diff-tune_diversity", property_key="diversity", perc=perc, targets=targets)
    difftune_spars = calculation([gh_diff_tune_last], "maze_last_diff-tune_sparsity", property_key="sparsity", perc=perc, targets=targets)

    # labels = ["complexity reduction", "dynamics awareness", "orthogonality", "interference", "diversity", "sparsity"]
    labels = ["complexity reduction", "dynamics awareness", "orthogonality", "noninterference", "diversity", "sparsity"]
    same = [same_lip, same_dist, same_ortho, same_interf, same_diversity, same_spars]
    similar = [similar_lip, similar_dist, similar_ortho, similar_interf, similar_diversity, similar_spars]
    diff = [diff_lip, diff_dist, diff_ortho, diff_interf, diff_diversity, diff_spars]
    difftune = [difftune_lip, difftune_dist, difftune_ortho, difftune_interf, difftune_diversity, difftune_spars]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    label_pos = 0.03
    fontsize = 9
    rotation = 90

    fig, ax = plt.subplots()
    rects1 = ax.bar(x-0.45+width, same, width, label='same', color=cmap(0, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*0.6, max(0, same[i])+label_pos, "{:.4f}".format(same[i]), color=cmap(0, 4), fontsize=fontsize, rotation=rotation)

    rects2 = ax.bar(x-0.45+width*2, similar, width, label='similar', color=cmap(2, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*1.6, max(0, similar[i]) + label_pos, "{:.4f}".format(similar[i]), color=cmap(2, 4), fontsize=fontsize, rotation=rotation)

    rects3 = ax.bar(x-0.45+width*3, diff, width, label='dissimilar(fix)', color=cmap(1, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*2.6, max(0, diff[i]) + label_pos, "{:.4f}".format(diff[i]), color=cmap(1, 4), fontsize=fontsize, rotation=rotation)

    rects4 = ax.bar(x-0.45+width*4, difftune, width, label='dissimilar(tune)', color=cmap(3, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*3.6, max(0, difftune[i]) + label_pos, "{:.4f}".format(difftune[i]), color=cmap(3, 4), fontsize=fontsize, rotation=rotation)

    ax.plot([x[0]-0.45, x[-1]+0.45], [0, 0], "--", color="grey")
    ax.legend(loc=4)
    ax.set_ylabel('Correlation')
    ax.set_ylim(-0.5, 1)
    ax.set_xticks(x)
    # ax.set_yticks([-1, -0.7, -0.5, 0, 0.5, 0.7, 1])
    ax.set_xticklabels(labels, rotation=30)
    # plt.grid(True)
    # plt.show()
    plt.savefig("plot/img/{}.pdf".format("maze_correlation_last"), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

def simple_picky_eater_correlation_last(perc, early=True):

    if not early:
        crgb_same_path = crgb_same_last
        crgb_diff_path = crgb_diff_last
        crgb_diff_tune_path = crgb_diff_tune_last
    else:
        crgb_same_path = crgb_same_early
        crgb_diff_path = crgb_diff_early
        crgb_diff_tune_path = crgb_diff_tune_early
 

    print("\nSame task")
    same_lip = calculation([crgb_same_path], "picky_eater_last_same_lip", property_key="lipschitz", perc=perc, targets=targets)
    same_dist = calculation([crgb_same_path], "picky_eater_last_same_distance", property_key="distance", perc=perc, targets=targets)
    same_ortho = calculation([crgb_same_path], "picky_eater_last_same_ortho", property_key="ortho", perc=perc, targets=targets)
    # same_interf = calculation([crgb_same_path], "picky_eater_last_same_interf", property_key="interf", perc=perc, targets=targets)
    same_diversity = calculation([crgb_same_path], "picky_eater_last_same_diversity", property_key="diversity", perc=perc, targets=targets)
    same_spars = calculation([crgb_same_path], "picky_eater_last_same_sparsity", property_key="sparsity", perc=perc, targets=targets)

    print("\nDifferent task - fix")
    diff_lip = calculation([crgb_diff_path], "picky_eater_last_diff-fix_lip", property_key="lipschitz", perc=perc, targets=targets)
    diff_dist = calculation([crgb_diff_path], "picky_eater_last_diff-fix_distance", property_key="distance", perc=perc, targets=targets)
    diff_ortho = calculation([crgb_diff_path], "picky_eater_last_diff-fix_ortho", property_key="ortho", perc=perc, targets=targets)
    # diff_interf = calculation([crgb_diff_path], "picky_eater_last_diff-fix_interf", property_key="interf", perc=perc, targets=targets)
    diff_diversity = calculation([crgb_diff_path], "picky_eater_last_diff-fix_diversity", property_key="diversity", perc=perc, targets=targets)
    diff_spars = calculation([crgb_diff_path], "picky_eater_last_diff-fix_sparsity", property_key="sparsity", perc=perc, targets=targets)

    print("\nDifferent task - tune")
    difftune_lip = calculation([crgb_diff_tune_path], "picky_eater_last_diff-tune_lip", property_key="lipschitz", perc=perc, targets=targets)
    difftune_dist = calculation([crgb_diff_tune_path], "picky_eater_last_diff-tune_distance", property_key="distance", perc=perc, targets=targets)
    difftune_ortho = calculation([crgb_diff_tune_path], "picky_eater_last_diff-tune_ortho", property_key="ortho", perc=perc, targets=targets)
    # difftune_interf = calculation([crgb_diff_tune_path], "picky_eater_last_diff-tune_interf", property_key="interf", perc=perc, targets=targets)
    difftune_diversity = calculation([crgb_diff_tune_path], "picky_eater_last_diff-tune_diversity", property_key="diversity", perc=perc, targets=targets)
    difftune_spars = calculation([crgb_diff_tune_path], "picky_eater_last_diff-tune_sparsity", property_key="sparsity", perc=perc, targets=targets)

    # labels = ["complexity reduction", "dynamics awareness", "orthogonality", "interference", "diversity", "sparsity"]
    # labels = ["complexity reduction", "dynamics awareness", "orthogonality", "noninterference", "diversity", "sparsity"]
    labels = ["complexity reduction", "dynamics awareness", "orthogonality", "diversity", "sparsity"]
    # same = [same_lip, same_dist, same_ortho, same_interf, same_diversity, same_spars]
    same = [same_lip, same_dist, same_ortho, same_diversity, same_spars]
    # similar = [similar_lip, similar_dist, similar_ortho, similar_interf, similar_diversity, similar_spars]
    diff = [diff_lip, diff_dist, diff_ortho, diff_diversity, diff_spars]
    # diff = [diff_lip, diff_dist, diff_ortho, diff_interf, diff_diversity, diff_spars]
    # difftune = [difftune_lip, difftune_dist, difftune_ortho, difftune_interf, difftune_diversity, difftune_spars]
    difftune = [difftune_lip, difftune_dist, difftune_ortho, difftune_diversity, difftune_spars]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    label_pos = 0.03
    fontsize = 9
    rotation = 90

    fig, ax = plt.subplots()
    rects1 = ax.bar(x-0.45+width, same, width, label='same', color=cmap(0, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*0.6, max(0, same[i])+label_pos, "{:.4f}".format(same[i]), color=cmap(0, 4), fontsize=fontsize, rotation=rotation)

    # rects2 = ax.bar(x-0.45+width*2, similar, width, label='similar', color=cmap(2, 4))
    # for i in range(len(x)):
    #     ax.text(x[i]-0.45+width*1.6, max(0, similar[i]) + label_pos, "{:.4f}".format(similar[i]), color=cmap(2, 4), fontsize=fontsize, rotation=rotation)

    rects3 = ax.bar(x-0.45+width*3, diff, width, label='dissimilar(fix)', color=cmap(1, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*2.6, max(0, diff[i]) + label_pos, "{:.4f}".format(diff[i]), color=cmap(1, 4), fontsize=fontsize, rotation=rotation)

    rects4 = ax.bar(x-0.45+width*4, difftune, width, label='dissimilar(tune)', color=cmap(3, 4))
    for i in range(len(x)):
        ax.text(x[i]-0.45+width*3.6, max(0, difftune[i]) + label_pos, "{:.4f}".format(difftune[i]), color=cmap(3, 4), fontsize=fontsize, rotation=rotation)

    ax.plot([x[0]-0.45, x[-1]+0.45], [0, 0], "--", color="grey")
    ax.legend()
    ax.set_ylabel('Correlation')
    ax.set_ylim(-0.5, 1)
    ax.set_xticks(x)
    # ax.set_yticks([-1, -0.7, -0.5, 0, 0.5, 0.7, 1])
    ax.set_xticklabels(labels, rotation=30)
    # plt.grid(True)
    # plt.show()
    if early:
        plt.savefig("plot/img/{}.pdf".format("correlation_early_model"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig("plot/img/{}.pdf".format("correlation_last_model"), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

if __name__ == '__main__':
    perc=[0, 1]
    targets_crgb = ["ReLU",
               # "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               "LTA",
               # "LTA+Control1g", "LTA+Control5g", "LTA+XY", "LTA+Decoder", "LTA+NAS", "LTA+Reward", "LTA+SF",
               # "Random", "Input"
                    ]
    targets_gw = ["ReLU",
               "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
               "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               # "Random", "Input",
               ]

    simple_picky_eater_correlation_last(perc)

    perc=[0, 1]
    targets = ["ReLU",
               "ReLU+Control1g", "ReLU+Control5g", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF",
               "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
               "FTA+Control1g", "FTA+Control5g", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF",
               # "Random", "Input",
               ]
    simple_maze_correlation_early(perc)
    simple_maze_correlation_last(perc)