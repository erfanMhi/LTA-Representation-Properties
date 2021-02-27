import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

from plot.plot_utils import *
from plot.plot_paths import *

os.chdir("..")
print("Change dir to", os.getcwd())

def load_fix_property(group, target_file_name, target_keywords):
    all_property = {}
    for i in group:
        path = i["property"]
        values = extract_property_setting(path, 0, target_file_name, target_keywords[target_file_name])
        all_property[i["label"]] = values
    return all_property

def load_online_property(group, target_key):
    all_property = {}
    for i in group:
        if target_key == "lipschitz":
            path = i["control"]
            values = extract_from_setting(path, 0, target_key, final_only=True)
        else:
            if "online_measure" in i.keys():
                path = i["online_measure"]
                values = extract_from_setting(path, 0, target_key, final_only=True)
            else:
                path = i["property"]
                values = extract_property_setting(path, 0, target_files[target_key], target_keywords[target_files[target_key]])

        all_property[i["label"]] = values
    return all_property

def load_auc(groups):
    assert type(groups[0]) == list, print("type of input should be a list of list(s)")
    all_auc = {}
    for group in groups:
        group_auc = {}
        temp = []
        for i in group:
            path = i["control"]
            # print("Loading returns from", path)
            returns = extract_from_setting(path, 0, "return")
            aucs = {}
            for run in returns:
                aucs[run] = np.array(returns[run]).sum()# / (i["optimal"] * len(returns[run]))
                temp.append(aucs[run])
            group_auc[i["label"]] = aucs

        # normalize auc by the max and min value in the same task
        max_auc = np.max(np.array(temp))
        min_auc = np.min(np.array(temp))
        for i in group:
            for run in group_auc[i["label"]].keys():
                old = group_auc[i["label"]][run]
                group_auc[i["label"]][run] = (old - min_auc) / (max_auc - min_auc)
        all_auc.update(group_auc)

    return all_auc


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

# def correlation(data1, data2, labels):
#     for label in labels:
#         l1, l2 = arrange_order(data1[label], data2[label])
#         assert not np.isnan(np.array(l1).sum()), print(l1)
#         assert not np.isnan(np.array(l2).sum()), print(l2)
#         cor = np.corrcoef(l1, l2)[0][1]
#         print("{}: correlation = {:.4f}".format(label, cor))

# def calculation_complexity_reduction(all_groups, perc=None, relationship=None):
#     keyword = "lipschitz"
#     property_file = "log"
#     print("\nChecking complexity reduction")
#
#     all_groups, all_group_dict = merge_groups(all_groups)
#
#     control = load_auc(all_groups)
#
#     assert type(all_groups[0]) == list, print("type of input should be a list of list(s)")
#     property = {}
#     for group in all_groups:
#         group_prop = {}
#         temp = []
#         for i in group:
#             path = i["control"]
#             print(path, keyword)
#             values = extract_property_setting(path, 0, property_file, keyword)
#             cr = {}
#             for run in values:
#                 cr[run] = values[run]
#                 temp.append(cr[run])
#             group_prop[i["label"]] = cr
#
#         max_cr = np.max(np.array(temp))
#         min_cr = np.min(np.array(temp))
#         for i in group:
#             for run in group_prop[i["label"]].keys():
#                 old = group_prop[i["label"]][run]
#                 group_prop[i["label"]][run] = (old - min_cr) / (max_cr - min_cr)
#         property.update(group_prop)
#
#     labels = [i["label"] for i in all_group_dict]
#     # correlation(control, property, labels)
#
#     # all reps:
#     all_control = []
#     all_property = []
#     all_color = []
#     # for rep in labels:
#     for idx in range(len(labels)):
#         rep = labels[idx]
#         ctr, prop = arrange_order(control[rep], property[rep])
#         all_control += ctr
#         all_property += prop
#         c = [idx] * len(ctr)
#         all_color += c
#
#     all_control = np.array(all_control)
#     # all_control = (all_control - np.min(all_control)) / (np.max(all_control) - np.min(all_control)) # normalization to [0,1]
#     all_property = np.array(all_property)
#     # all_property = (all_property - np.min(all_property)) / (np.max(all_property) - np.min(all_property)) # normalization to [0,1]
#     all_control, all_property = perform_filter(all_control, all_property, perc=perc)
#     all_control, all_property = perform_transformer(all_control, all_property, relationship=relationship)
#     cor = np.corrcoef(all_control, all_property)[0][1]
#     print("All reps: correlation = {:.4f}".format(cor))
#
#     plt.figure()
#     plt.scatter(all_control, all_property, c=all_color, cmap='viridis', s=5)
#     plt.xlabel("AUC")
#     plt.ylabel("Property measure")
#     plt.savefig("plot/img/scatter_{}_trans={}.png".format("complexity_reduction", relationship), dpi=300, bbox_inches='tight')
#     plt.close()
#     plt.clf()
#     print("save in plot/img/")
#     return cor

def calculation(all_groups, title, property_key=None, property_file=None, perc=None, relationship=None):
    # print("\nChecking", property_file, property_key)

    all_groups, all_group_dict = merge_groups(all_groups)
    # print(all_groups, "\n\n", all_group_dict, "\n")
    control = load_auc(all_groups)
    if property_file is not None:
        property = load_fix_property(all_group_dict, property_file, target_keywords)
    elif property_key is not None:
        property = load_online_property(all_group_dict, property_key)
    else:
        raise NotImplementedError
    labels = [i["label"] for i in all_group_dict]

    # all reps:
    all_control = []
    all_property = []
    all_color = []
    indexs = [0]
    for idx in range(len(labels)):
        rep = labels[idx]
        assert len(rep.split("_")) == 2
        old_rep = rep.split("_")[0]
        color = violin_colors[old_rep]
        ctr, prop = arrange_order(control[rep], property[rep])
        all_control += ctr
        all_property += prop
        # c = [idx] * len(ctr)
        all_color.append(color)
        assert len(ctr) == len(prop)
        indexs.append(len(ctr)+indexs[-1])

    all_control = np.array(all_control)
    all_property = np.array(all_property)
    all_control, all_property = perform_filter(all_control, all_property, perc=perc)
    all_control, all_property = perform_transformer(all_control, all_property, relationship=relationship)
    cor = np.corrcoef(all_control, all_property)[0][1]
    # print(len(all_control))
    print("All reps: correlation = {:.4f}".format(cor))

    plt.figure()
    # plt.scatter(all_control, all_property, c=all_color, cmap='viridis', s=5)
    for i in range(len(indexs)-1):
        plt.scatter(all_control[indexs[i]: indexs[i+1]], all_property[indexs[i]: indexs[i+1]], c=all_color[i], s=5)
    plt.xlabel("AUC")
    plt.ylabel("Property measure")
    # plt.savefig("plot/img/scatter_{}_trans={}.pdf".format(title, relationship), dpi=300, bbox_inches='tight')
    plt.savefig("plot/img/{}.png".format(title), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    print("save {} in plot/img/".format(title))
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

def simple_maze_fixed_property(perc):
    print("\nSame task")
    # calculation([maze_same], "maze_same_decorr", property_file="decorrelation.txt", perc=perc)
    # calculation([maze_same], "maze_same_distance", property_file="distance.txt", perc=perc)
    # calculation([maze_same], "maze_same_interf", property_file="interference.txt", perc=perc)
    # calculation([maze_same], "maze_same_lp", property_file="linear_probing_xy.txt", perc=perc)
    # calculation([maze_same], "maze_same_ortho", property_file="orthogonality.txt", perc=perc)
    # calculation([maze_same], "maze_same_sparsity", property_file="sparsity_instance.txt", perc=perc)
    # calculation_complexity_reduction([maze_same], "maze_same_lipschitz", perc=perc)


def simple_maze_online_property(perc):
    print("\nSame task")
    same_lip = calculation([gh_same], "maze_same_lip", property_key="lipschitz", perc=perc)
    same_dist = calculation([gh_same], "maze_same_distance", property_key="distance", perc=perc)
    same_ortho = calculation([gh_same], "maze_same_ortho", property_key="ortho", perc=perc)
    same_noninterf = calculation([gh_same], "maze_same_noninterf", property_key="noninterf", perc=perc)
    same_decorr = calculation([gh_same], "maze_same_decorr", property_key="decorr", perc=perc)
    same_spars = calculation([gh_same], "maze_same_sparsity", property_key="sparsity", perc=perc)

    print("\nSimilar task")
    similar_lip = calculation([gh_similar], "maze_similar_lip", property_key="lipschitz", perc=perc)
    similar_dist = calculation([gh_similar], "maze_similar_distance", property_key="distance", perc=perc)
    similar_ortho = calculation([gh_similar], "maze_similar_ortho", property_key="ortho", perc=perc)
    similar_noninterf = calculation([gh_similar], "maze_similar_noninterf", property_key="noninterf", perc=perc)
    similar_decorr = calculation([gh_similar], "maze_similar_decorr", property_key="decorr", perc=perc)
    similar_spars = calculation([gh_similar], "maze_similar_sparsity", property_key="sparsity", perc=perc)


    print("\nDifferent task - fix")
    diff_lip = calculation([gh_diff], "maze_diff-fix_lip", property_key="lipschitz", perc=perc)
    diff_dist = calculation([gh_diff], "maze_diff-fix_distance", property_key="distance", perc=perc)
    diff_ortho = calculation([gh_diff], "maze_diff-fix_ortho", property_key="ortho", perc=perc)
    diff_noninterf = calculation([gh_diff], "maze_diff-fix_noninterf", property_key="noninterf", perc=perc)
    diff_decorr = calculation([gh_diff], "maze_diff-fix_decorr", property_key="decorr", perc=perc)
    diff_spars = calculation([gh_diff], "maze_diff-fix_sparsity", property_key="sparsity", perc=perc)

    print("\nDifferent task - tune")
    difftune_lip = calculation([gh_diff_tune], "maze_diff-tune_lip", property_key="lipschitz", perc=perc)
    difftune_dist = calculation([gh_diff_tune], "maze_diff-tune_distance", property_key="distance", perc=perc)
    difftune_ortho = calculation([gh_diff_tune], "maze_diff-tune_ortho", property_key="ortho", perc=perc)
    difftune_noninterf = calculation([gh_diff_tune], "maze_diff-tune_noninterf", property_key="noninterf", perc=perc)
    difftune_decorr = calculation([gh_diff_tune], "maze_diff-tune_decorr", property_key="decorr", perc=perc)
    difftune_spars = calculation([gh_diff_tune], "maze_diff-tune_sparsity", property_key="sparsity", perc=perc)

    # lip = [same_lip, similar_lip, diff_lip, difftune_lip]
    # dist = [same_dist, similar_dist, diff_dist, difftune_dist]
    # ortho = [same_ortho, similar_ortho, diff_ortho, difftune_ortho]
    # noninterf = [same_noninterf, similar_noninterf, diff_noninterf, difftune_noninterf]
    # decorr = [same_decorr, similar_decorr, diff_decorr, difftune_decorr]
    # spars = [same_spars, similar_spars, diff_spars, difftune_spars]

    labels = ["complexity reduction", "dynamics awareness", "orthogonality", "noninterference", "decorrelation", "sparsity"]
    same = [same_lip, same_dist, same_ortho, same_noninterf, same_decorr, same_spars]
    similar = [similar_lip, similar_dist, similar_ortho, similar_noninterf, similar_decorr, similar_spars]
    diff = [diff_lip, diff_dist, diff_ortho, diff_noninterf, diff_decorr, diff_spars]
    difftune = [difftune_lip, difftune_dist, difftune_ortho, difftune_noninterf, difftune_decorr, difftune_spars]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x-0.45+width, same, width, label='same')
    rects2 = ax.bar(x-0.45+width*2, similar, width, label='similar')
    rects3 = ax.bar(x-0.45+width*3, diff, width, label='dissimilar(fix)')
    rects4 = ax.bar(x-0.45+width*4, difftune, width, label='dissimilar(tune)')
    ax.plot([x[0]-0.45, x[-1]+0.45], [0, 0], "--", color="grey")
    ax.legend()
    ax.set_ylabel('Correlation')
    ax.set_xticks(x)
    ax.set_yticks([-1, -0.7, -0.5, 0, 0.5, 0.7, 1])
    ax.set_xticklabels(labels, rotation=30)
    plt.grid(True)
    plt.show()

def simple_maze_eta_study(perc):
    print("\nDifferent task - fix")
    diff_lip = calculation([gh_etaStudy_diff_fix], "maze_eta_diff-fix_lip", property_key="lipschitz", perc=perc)
    diff_dist = calculation([gh_etaStudy_diff_fix], "maze_eta_diff-fix_distance", property_key="distance", perc=perc)
    diff_ortho = calculation([gh_etaStudy_diff_fix], "maze_eta_diff-fix_ortho", property_key="ortho", perc=perc)
    diff_noninterf = calculation([gh_etaStudy_diff_fix], "maze_eta_diff-fix_noninterf", property_key="noninterf", perc=perc)
    diff_decorr = calculation([gh_etaStudy_diff_fix], "maze_eta_diff-fix_decorr", property_key="decorr", perc=perc)
    diff_spars = calculation([gh_etaStudy_diff_fix], "maze_eta_diff-fix_sparsity", property_key="sparsity", perc=perc)

    print("\nDifferent task - tune")
    difftune_lip = calculation([gh_etaStudy_diff_tune], "maze_eta_diff-tune_lip", property_key="lipschitz", perc=perc)
    difftune_dist = calculation([gh_etaStudy_diff_tune], "maze_eta_diff-tune_distance", property_key="distance", perc=perc)
    difftune_ortho = calculation([gh_etaStudy_diff_tune], "maze_eta_diff-tune_ortho", property_key="ortho", perc=perc)
    difftune_noninterf = calculation([gh_etaStudy_diff_tune], "maze_eta_diff-tune_noninterf", property_key="noninterf", perc=perc)
    difftune_decorr = calculation([gh_etaStudy_diff_tune], "maze_eta_diff-tune_decorr", property_key="decorr", perc=perc)
    difftune_spars = calculation([gh_etaStudy_diff_tune], "maze_eta_diff-tune_sparsity", property_key="sparsity", perc=perc)

    # lip = [same_lip, similar_lip, diff_lip, difftune_lip]
    # dist = [same_dist, similar_dist, diff_dist, difftune_dist]
    # ortho = [same_ortho, similar_ortho, diff_ortho, difftune_ortho]
    # noninterf = [same_noninterf, similar_noninterf, diff_noninterf, difftune_noninterf]
    # decorr = [same_decorr, similar_decorr, diff_decorr, difftune_decorr]
    # spars = [same_spars, similar_spars, diff_spars, difftune_spars]

    labels = ["complexity reduction", "distance", "orthogonality", "noninterference", "decorrelation", "sparsity"]
    diff = [diff_lip, diff_dist, diff_ortho, diff_noninterf, diff_decorr, diff_spars]
    difftune = [difftune_lip, difftune_dist, difftune_ortho, difftune_noninterf, difftune_decorr, difftune_spars]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    rects3 = ax.bar(x-0.45+width*1, diff, width, label='dissimilar(fix)')
    rects4 = ax.bar(x-0.45+width*2, difftune, width, label='dissimilar(tune)')
    ax.plot([x[0]-0.45, x[-1]+0.45], [0, 0], "--", color="grey")
    ax.legend()
    ax.set_ylabel('Correlation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    plt.show()

if __name__ == '__main__':
    perc=[0, 1]
    simple_maze_online_property(perc)
    # simple_maze_eta_study(perc)
