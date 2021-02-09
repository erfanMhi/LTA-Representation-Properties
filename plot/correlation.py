import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

from plot.plot_utils import *
from plot.plot_dicts import *

os.chdir("..")
print("Change dir to", os.getcwd())

def load_property(group, target_file_name, target_keywords):

    all_property = {}
    for i in group:
        path = i["property"]
        values = extract_property_setting(path, 0, target_file_name, target_keywords[target_file_name])
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
            returns = extract_return_setting(path, 0)
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

def calculation_complexity_reduction(all_groups, perc=None, relationship=None):
    keyword = "Lipschitz:"
    property_file = "log"
    print("\nChecking complexity reduction")

    all_groups, all_group_dict = merge_groups(all_groups)

    control = load_auc(all_groups)

    assert type(all_groups[0]) == list, print("type of input should be a list of list(s)")
    property = {}
    for group in all_groups:
        group_prop = {}
        temp = []
        for i in group:
            path = i["control"]
            values = extract_property_setting(path, 0, property_file, keyword)
            cr = {}
            for run in values:
                cr[run] = values[run]
                temp.append(cr[run])
            group_prop[i["label"]] = cr

        max_cr = np.max(np.array(temp))
        min_cr = np.min(np.array(temp))
        for i in group:
            for run in group_prop[i["label"]].keys():
                old = group_prop[i["label"]][run]
                group_prop[i["label"]][run] = (old - min_cr) / (max_cr - min_cr)
        property.update(group_prop)
    # property = {}
    # for i in all_group_dict:
    #     path = i["control"]
    #     values = extract_property_setting(path, 0, property_file, keyword)
    #     property[i["label"]] = values

    labels = [i["label"] for i in all_group_dict]
    # correlation(control, property, labels)

    # all reps:
    all_control = []
    all_property = []
    all_color = []
    # for rep in labels:
    for idx in range(len(labels)):
        rep = labels[idx]
        ctr, prop = arrange_order(control[rep], property[rep])
        all_control += ctr
        all_property += prop
        c = [idx] * len(ctr)
        all_color += c

    all_control = np.array(all_control)
    # all_control = (all_control - np.min(all_control)) / (np.max(all_control) - np.min(all_control)) # normalization to [0,1]
    all_property = np.array(all_property)
    # all_property = (all_property - np.min(all_property)) / (np.max(all_property) - np.min(all_property)) # normalization to [0,1]
    all_control, all_property = perform_filter(all_control, all_property, perc=perc)
    all_control, all_property = perform_transformer(all_control, all_property, relationship=relationship)
    cor = np.corrcoef(all_control, all_property)[0][1]
    print("All reps: correlation = {:.4f}".format(cor))

    plt.figure()
    plt.scatter(all_control, all_property, c=all_color, cmap='viridis', s=5)
    plt.xlabel("AUC")
    plt.ylabel("Property measure")
    plt.savefig("plot/img/scatter_{}_trans={}.png".format("complexity_reduction", relationship), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    print("save in plot/img/")
    return cor

def calculation(all_groups, property_file, perc=None, relationship=None):
    print("\nChecking", property_file)

    all_groups, all_group_dict = merge_groups(all_groups)
    control = load_auc(all_groups)
    property = load_property(all_group_dict, property_file, target_keywords)
    labels = [i["label"] for i in all_group_dict]

    # all reps:
    all_control = []
    all_property = []
    all_color = []
    for idx in range(len(labels)):
        rep = labels[idx]
        ctr, prop = arrange_order(control[rep], property[rep])
        all_control += ctr
        all_property += prop
        c = [idx] * len(ctr)
        all_color += c

    all_control = np.array(all_control)
    all_property = np.array(all_property)
    all_control, all_property = perform_filter(all_control, all_property, perc=perc)
    all_control, all_property = perform_transformer(all_control, all_property, relationship=relationship)
    cor = np.corrcoef(all_control, all_property)[0][1]
    # print(len(all_control))
    print("All reps: correlation = {:.4f}".format(cor))

    plt.figure()
    plt.scatter(all_control, all_property, c=all_color, cmap='viridis', s=5)
    plt.xlabel("AUC")
    plt.ylabel("Property measure")
    plt.savefig("plot/img/scatter_{}_trans={}.pdf".format(property_file.split(".")[0], relationship), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    print("save in plot/img/")
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

def simple_maze(perc):
    # print("\nSame task")
    # calculation([maze_same], "decorrelation.txt", perc=perc)
    # calculation([maze_same], "distance.txt", perc=perc)
    # calculation([maze_same], "interference.txt", perc=perc)
    # calculation([maze_same], "linear_probing_xy.txt", perc=perc)
    # calculation([maze_same], "orthogonality.txt", perc=perc)
    # calculation([maze_same], "sparsity_instance.txt", perc=perc)
    # calculation_complexity_reduction([maze_same], perc=perc)

    # print("\nSimilar task")
    # calculation([maze_similar], "decorrelation.txt", perc=perc)
    # calculation([maze_similar], "distance.txt", perc=perc)
    # calculation([maze_similar], "interference.txt", perc=perc)
    # calculation([maze_similar], "linear_probing_xy.txt", perc=perc)
    # calculation([maze_similar], "orthogonality.txt", perc=perc)
    # calculation([maze_similar], "sparsity_instance.txt", perc=perc)
    # calculation_complexity_reduction([maze_similar], perc=perc)

    # print("\nDifferent task - fix")
    # calculation([maze_different_fix], "decorrelation.txt", perc=perc)
    # calculation([maze_different_fix], "distance.txt", perc=perc)
    # calculation([maze_different_fix], "interference.txt", perc=perc)
    # calculation([maze_different_fix], "linear_probing_xy.txt", perc=perc)
    # calculation([maze_different_fix], "orthogonality.txt", perc=perc)
    # calculation([maze_different_fix], "sparsity_instance.txt", perc=perc)
    # calculation_complexity_reduction([maze_different_fix], perc=perc)

    print("\nDifferent task - tune")
    calculation([maze_different_tune], "decorrelation.txt", perc=perc)
    calculation([maze_different_tune], "distance.txt", perc=perc)
    calculation([maze_different_tune], "interference.txt", perc=perc)
    calculation([maze_different_tune], "linear_probing_xy.txt", perc=perc)
    calculation([maze_different_tune], "orthogonality.txt", perc=perc)
    calculation([maze_different_tune], "sparsity_instance.txt", perc=perc)
    calculation_complexity_reduction([maze_different_tune], perc=perc)

def picky_eater(perc):

    # print("\nSame task")
    # calculation([eater_same], "decorrelation.txt", perc=perc)
    # calculation([eater_same], "distance.txt", perc=perc)
    # calculation([eater_same], "interference.txt", perc=perc)
    # calculation([eater_same], "linear_probing_xy.txt", perc=perc)
    # calculation([eater_same], "linear_probing_color.txt", perc=perc)
    # calculation([eater_same], "linear_probing_count.txt", perc=perc)
    # calculation([eater_same], "orthogonality.txt", perc=perc)
    # calculation([eater_same], "sparsity_instance.txt", perc=perc)
    # calculation_complexity_reduction([eater_same], perc=perc)
    #
    # print("\nDifferent task - fix")
    # calculation([eater_different_fix], "decorrelation.txt", perc=perc)
    # calculation([eater_different_fix], "distance.txt", perc=perc)
    # calculation([eater_different_fix], "interference.txt", perc=perc)
    # calculation([eater_different_fix], "linear_probing_xy.txt", perc=perc)
    # calculation([eater_different_fix], "linear_probing_color.txt", perc=perc)
    # calculation([eater_different_fix], "linear_probing_count.txt", perc=perc)
    # calculation([eater_different_fix], "orthogonality.txt", perc=perc)
    # calculation([eater_different_fix], "sparsity_instance.txt", perc=perc)
    # calculation_complexity_reduction([eater_different_fix], perc=perc)

    print("\nDifferent task - tune")
    calculation([eater_different_tune], "decorrelation.txt", perc=perc)
    calculation([eater_different_tune], "distance.txt", perc=perc)
    calculation([eater_different_tune], "interference.txt", perc=perc)
    calculation([eater_different_tune], "linear_probing_xy.txt", perc=perc)
    calculation([eater_different_tune], "linear_probing_color.txt", perc=perc)
    calculation([eater_different_tune], "linear_probing_count.txt", perc=perc)
    calculation([eater_different_tune], "orthogonality.txt", perc=perc)
    calculation([eater_different_tune], "sparsity_instance.txt", perc=perc)
    calculation_complexity_reduction([eater_different_tune], perc=perc)

if __name__ == '__main__':
    perc=[0, 1]
    # simple_maze(perc)
    picky_eater(perc)