import os
import re
import copy
import pickle
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from plot.plot_paths import violin_colors, curve_styles

from plot_paths import *

flatten = lambda t: [item for sublist in t for item in sublist]


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

def property_filter(allg_transf_perf, allp_properties, perc, rank):
    """
    allg_transf_perf = {goal: {rep : {run: x}}}
    allp_properties = {property: {rep : {run: x}}}
    perc = {property: [low, high]}
    """
    def filter_perc(properties, perc):
        concate = []
        keys = []
        for rep in properties.keys():
            for run in properties[rep].keys():
                concate.append(properties[rep][run])
                keys.append((rep, run))
        concate = np.array(concate)
        low = np.percentile(concate, perc[0])
        high = np.percentile(concate, perc[1])
        idx = np.where((concate>=low) & (concate<=high))[0]
        getkeys = [keys[i] for i in idx]
        print("\nProperty threshold {} - {}\n".format(low, high))
        return getkeys

    def filter_rank(properties, rank):
        concate = []
        keys = []
        for rep in properties.keys():
            for run in properties[rep].keys():
                concate.append(properties[rep][run])
                keys.append((rep, run))
        sorted_conc = np.array(concate)
        sorted_conc.sort()
        concate = np.array(concate)
        low = sorted_conc[rank[0]]
        high = sorted_conc[min(len(concate)-1, rank[1])]
        idx = np.where((concate>=low) & (concate<=high))[0]
        # print(concate)
        # print(idx)
        # input()
        getkeys = [keys[i] for i in idx]
        print("\nProperty threshold {} - {}\n".format(low, high))
        return getkeys

    if perc is None and rank is None:
        return allg_transf_perf, allp_properties
    assert not ((perc is not None) and (rank is not None))
    notnone = perc if perc is not None else rank
    for pk in notnone:
        new_trans = {}
        new_prop = {}
        if rank is not None and perc is None:
            reprun_keys = filter_rank(allp_properties[pk], rank[pk])
        elif rank is None and perc is not None:
            reprun_keys = filter_perc(allp_properties[pk], perc[pk])
        else:
            raise NotImplementedError

        for goal in allg_transf_perf.keys():
            new_trans[goal] = {}
        for prop in allp_properties.keys():
            new_prop[prop] = {}
        for k in reprun_keys:
            rep, run = k
            for goal in allg_transf_perf.keys():
                if rep not in new_trans[goal].keys():
                    new_trans[goal][rep] = {}
                new_trans[goal][rep][run] = allg_transf_perf[goal][rep][run]
            for prop in allp_properties.keys():
                if rep not in new_prop[prop].keys():
                    new_prop[prop][rep] = {}
                new_prop[prop][rep][run] = allp_properties[prop][rep][run]
        allg_transf_perf, allp_properties = new_trans, new_prop
    return allg_transf_perf, allp_properties

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



def arrange_order(dict1, cut_length=True, scale=1):
    lst = []
    min_l = np.inf
    for i in sorted(dict1):
        v1 = dict1[i]
        
#         if len(v1) == 201:
            # lst.append(v1)
        # else:
            # print('Length: ', len(v1))
#             print('Run: ', i)
        if len(v1) in [1, 11, 16, 31, 101, 151] or len(v1) > 10:
        # if len(v1) > 25:
            lst.append(v1)
        else:
            print('Length: ', len(v1), 'Run: ', i)
            continue

        l = len(v1)
        min_l = l if l < min_l else min_l
        if cut_length:
            l = len(v1)
            min_l = l if l < min_l else min_l
    if cut_length:
        for i in range(len(lst)):
            lst[i] = lst[i][:min_l]
    return np.array(lst) / float(scale)

def arrange_order_2(dict1, dict2):
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

def load_info(paths, param, key, label=None, path_key="control"):
    all_rt = {}
    for i in paths:
        path = i[path_key]
        param = param if type(path) != list else path[1]
        if type(path) == list:
            param = path[1]
            path = path[0]
        res, _ = extract_from_setting(path, param, key, label=label)
        all_rt[i["label"]] = res
    return all_rt

# def load_return(paths, total_param, start_param):
def load_return(paths, setting_list, search_lr=False, key="return", path_key="control"):
    all_rt = {}
    for i in paths:
        path = i[path_key]
        # print("Loading returns from", path)
        if search_lr and type(path) == list:
            path = path[0]
        returns = extract_return_all(path, setting_list, search_lr=search_lr, key=key)#total_param, start_param)
        all_rt[i["label"]] = returns
    return all_rt

def get_avg(all_res):
    mu = all_res.mean(axis=0)
    # mu = all_res.max(axis=0)
    std = all_res.std(axis=0)
    # ste = std / np.sqrt(len(std))
    ste = std / np.sqrt(all_res.shape[0])
    return mu, ste

def draw_curve(all_res, ax, label, color=None, style="-", alpha=1, linewidth=1.5, draw_ste=True, xcoord=None, break_point=None):
    if len(all_res) == 0:
        return None
    mu, ste = get_avg(all_res)
    if xcoord is None:
        xcoord = list(range(1, len(mu)+1))
    if break_point is None:
        if color is None:
            p = ax.plot(xcoord, mu, label=label, alpha=alpha, linestyle=style, linewidth=linewidth)
            # color = p.get_color()
        else:
            ax.plot(xcoord, mu, label=label, color=color, alpha=alpha, linestyle=style, linewidth=linewidth)
    else:
        if color is None:
            p = ax.plot(xcoord[:break_point], mu[:break_point], label=label, alpha=alpha, linestyle=style, linewidth=linewidth)
            p = ax.plot(xcoord[break_point:], mu[break_point:], label=label, alpha=alpha, linestyle=style, linewidth=linewidth)
            # color = p.get_color()
        else:
            ax.plot(xcoord[:break_point], mu[:break_point], label=label, color=color, alpha=alpha, linestyle=style, linewidth=linewidth)
            ax.plot(xcoord[break_point-1:], mu[break_point-1:], label=label, color='#353839', alpha=alpha, linestyle=style, linewidth=linewidth)
            ax.plot(xcoord[break_point-1:break_point], mu[break_point-1:break_point], marker="o", markersize=1.5, markeredgecolor=color, markerfacecolor=color)
            print(xcoord[break_point:break_point+1], mu[break_point:break_point+1])
    if draw_ste:
        ax.fill_between(list(range(len(mu))), mu - ste * 2, mu + ste * 2, color=color, alpha=0.1, linewidth=0.)
    print(label, "auc =", np.sum(mu))
    return mu

def draw_cut(cuts, all_res, ax, color, ymin):
    mu = all_res.mean(axis=0)
    x_mean = cuts.mean()
    x_max = cuts.max()
    # print('x_max: ', x_max)
    # print('mu: ', mu)
    # print(len(mu))
    # print('x_max_onward: ', mu[int(x_max):])
    ax.vlines(x_max, ymin, np.interp(x_max, list(range(len(mu))), mu), ls=":", colors=color, alpha=0.5, linewidth=1)

def is_converged(cuts, all_res, err_interval=0.04):
    """
    Checks whether the property line converged after the cut (early-stopping moment)
    """
    mu = all_res.mean(axis=0)
    x_max = cuts.max()
    # print('x_max: ', x_max)
    # print('mu: ', mu)
    # print(len(mu))
    # print('x_max_onward: ', mu[int(x_max):])
    conv_point = mu[int(x_max)]
    if x_max > 10:
        print('x_max: ', x_max)
    
    # print(mu)
    for point in mu[int(x_max):]:
        if conv_point - err_interval <= point <= conv_point + err_interval:
            continue
        else:
            return False
    return True

def convergence_intensity(cuts, all_res):
    """
    Checks whether the property line converged after the cut (early-stopping moment)
    """
    mu = all_res.mean(axis=0)
    x_max = cuts.max()
    # print('x_max: ', x_max)
    # print('mu: ', mu)
    # print(len(mu))
    # print('x_max_onward: ', mu[int(x_max):])
    conv_point = mu[int(x_max)]
    if x_max > 10:
        print('x_max: ', x_max)
    
    # print(mu)
    intensity = 0
    for point in mu[int(x_max):]:
        intensity += np.abs(conv_point-point)
    return intensity 



def extract_from_single_run(file, key, label=None, before_step=None):
    with open(file, "r") as f:
        content = f.readlines()
    returns = []

    check = True
    # check = False
    # for num, l in enumerate(content):
    #     info = l.split("|")[1].strip()
    #     if "epsilon: 0.1" in info:
    #         check=True

    for num, l in enumerate(content):
        if "|" in l:
            info = l.split("|")[1].strip()
            i_list = info.split(" ")
            #print('i_list: ', i_list)
            if key == "learning_rate" and "learning_rate:" in i_list:
                returns = i_list[1]
                return returns

            if "EVAL:" == i_list[0] or "total" == i_list[0] or "TRAIN" == i_list[0]:

                # print('i_list: ', i_list[0], "returns" in i_list)
                if key=="return" and "returns" in i_list:
                    returns.append(float(i_list[i_list.index("returns")+1].split("/")[0].strip())) # mean
                    # returns.append(float(i_list[i_list.index("returns")+1].split("/")[1].strip())) # median
                    # returns.append(float(i_list[i_list.index("returns")+1].split("/")[2].strip())) # min
                    # returns.append(float(i_list[i_list.index("returns")+1].split("/")[3].strip())) # max
                elif key == "lipschitz" and "Lipschitz:" in i_list:
                    if label is None:
                        returns.append(float(i_list[i_list.index("Lipschitz:") + 1].split("/")[1].strip()))  # mean
                    else:
                        if label in i_list:
                            returns.append(float(i_list[i_list.index("Lipschitz:") + 1].split("/")[1].strip()))  # mean
                elif key == "distance" and "Distance:" in i_list:
                    if label is None:
                        returns.append(float(i_list[i_list.index("Distance:") + 1].split("/")[0].strip()))
                    else:
                        if label in i_list:
                            returns.append(float(i_list[i_list.index("Distance:" ) + 1].split("/")[0].strip()))
                elif key == "ortho" and "Orthogonality:" in i_list:
                    if label is None:
                        returns.append(float(i_list[i_list.index("Orthogonality:") + 1].split("/")[0].strip()))
                    else:
                        if label in i_list:
                            returns.append(float(i_list[i_list.index("Orthogonality:") + 1].split("/")[0].strip()))
                # elif key == "noninterf" and "Noninterference:" in i_list:
                #     interf = float(i_list[i_list.index("Noninterference:") + 1].split("/")[0].strip())
                #     if not (np.isnan(interf) or np.isinf(interf) or np.isinf(-interf)):
                #         returns.append(interf)
                #     else:
                #         print("{} non-interference has {} value".format(file, interf))
                elif key == "interf" and "Interference:" in i_list:
                    interf = float(i_list[i_list.index("Interference:") + 1].split("/")[0].strip())
                    returns.append(interf)
                elif key == "decorr" and "Decorrelation:" in i_list:
                    returns.append(float(i_list[i_list.index("Decorrelation:") + 1].split("/")[0].strip()))
                elif key == "diversity" and "Diversity:" in i_list:
                    if label is None:
                        returns.append(float(i_list[i_list.index("Diversity:") + 1].split("/")[0].strip()))
                        if np.isnan(returns[-1]):
                            returns[-1] = 0
                    else:
                        # key_label = "%s Diversity:" % label
                        if label in i_list:
                            returns.append(float(i_list[i_list.index("Diversity:") + 1].split("/")[0].strip()))
                            if np.isnan(returns[-1]):
                                returns[-1] = 0
                elif key == "sparsity" and "Instance Sparsity:" in info:
                    if label is None:
                        returns.append(float(i_list[i_list.index("Sparsity:") + 1].split(",")[0].strip()))
                    else:
                        # key_label = "%s Instance Sparsity:" % label
                        if label in i_list:
                            returns.append(float(i_list[i_list.index("Sparsity:") + 1].split(",")[0].strip()))
                elif key == "mi" and "Mutual Info:" in info:
                    if label is None:
                        returns.append(float(i_list[i_list.index("Info:") + 1].split(",")[0].strip()))
                    else:
                        if label in i_list:
                            returns.append(float(i_list[i_list.index("Info:") + 1].split(",")[0].strip()))

                # used only when extract property for early-stopping model
                if before_step is not None:
                    current_step = int(info.split("steps")[1].split(",")[0])
                    # print('current_step: ', current_step)
                    # print('before_step: ', before_step)
                    if current_step == before_step[0]:
                        return returns
            #Fruit state-values (green, 2) LOG: steps 0, episodes   0, values 0.0001073884 (mean)
            if "Fruit" in i_list[0] and key in ["action-values", "state-values"]:

                match_fruit = re.search(r"\(([A-Za-z0-9_, ]*)\)", info)
                label_match_fruit = re.search(r"\(([A-Za-z0-9_, ]*)\)", label)

                match_rm_fruits = re.search(r"\(([A-Za-z0-9_, ]*)\)", info[match_fruit.end():])

                label_match_rm_fruits = re.search(r"\(([A-Za-z0-9_, ]*)\)", label[label_match_fruit.end():])
           #      print('---------fruit check---------')
                # print(file)
                # print(info)
                # print(match_fruit)
                # print(label_match_fruit)
                # print(match_rm_fruits)
                # print(label_match_rm_fruits)
# #                 #print(result.group(0))
                if key=="state-values" and "state-values" in i_list:
                    if label_match_fruit.group(0) == match_fruit.group(0) and label_match_rm_fruits.group(0) == match_rm_fruits.group(0):
                        returns.append(float(i_list[i_list.index("values")+1].split("/")[0].strip())) # mean
                
                if key=="action-values" and "action-values" in i_list:
                    if label_match_fruit.group(0) == match_fruit.group(0) and label_match_rm_fruits.group(0) == match_rm_fruits.group(0) and 'Fruit-undirected' in info and ') undirected' in label:
                    # if len(i_list[i_list.index("values")+1].split("/")) == 3 and ') undirected' in label:
                        returns.append(float(i_list[i_list.index("values")+1].split("/")[0].strip())) # mean
                    # if len(i_list[i_list.index("values")+1].split("/")) == 1 and ') directed' in label:
                    if label_match_fruit.group(0) == match_fruit.group(0) and label_match_rm_fruits.group(0) == match_rm_fruits.group(0) and 'Fruit-directed' in info and ') directed' in label:
                            returns.append(float(i_list[i_list.index("values")+1].split("/")[0].strip())) # mean
 
            
            if "early-stopping" in i_list and key == "model":
                cut = num - 1 # last line
                # converge_at = int(content[cut].split("total steps")[1].split(",")[0])
                converge_at = int(content[cut].split("steps")[1].split(",")[0])

                l = cut
                found = False
                while l > 0 and not found:
                    info_temp = content[l].split("|")[1].strip()
                    i_list_temp = info_temp.split(" ")
                    if "total" == i_list_temp[0]:
                        if "returns" in i_list_temp:
                            converge_return = float(i_list_temp[i_list_temp.index("returns") + 1].split("/")[0].strip())
                            found = True
                    l -= 1

                returns = [converge_at]
                # stop = False
                # above_step = [converge_at]
                # above_return = [converge_return]
                # while l > 0 and not stop:
                #     info_temp = content[l].split("|")[1].strip()
                #     i_list_temp = info_temp.split(" ")
                #     if "total" == i_list_temp[0]:
                #         if "returns" in i_list_temp:
                #             r_temp = float(i_list_temp[i_list_temp.index("returns") + 1].split("/")[0].strip())
                #             if r_temp >= 0.95 * converge_return:
                #                 above_step.append(int(content[l].split("total steps")[1].split(",")[0]))
                #                 above_return.append(r_temp)
                #             else:
                #                 # print(r_temp)
                #                 stop = True
                #     l -= 1

                # cut = above_return[-1]
                # returns = above_step[-1]
                # print("100% return ({}) at {}, 95% return ({}) at {}".format(converge_return, converge_at, cut, returns))

    # Sanity Check
    if not isinstance(returns, int):
        if len(returns) in [0]:#, 1] :
            print('Empty returns {}: '.format(key), returns)
            print('File Name: ', file)
    return returns
#
# def extract_lipschitz_single_run(file):
#     with open(file, "r") as f:
#         content = f.readlines()
#     returns = []
#     for l in content:
#         info = l.split("|")[1].strip()
#         i_list = info.split(" ")
#         if "total" == i_list[0]:
#             if "Lipschitz:" in i_list:
#                 returns.append(float(i_list[i_list.index("Lipschitz:")+1].split("/")[1].strip())) # mean
#     return returns
#

def extract_from_setting(find_in, setting, key="return", final_only=False, label=None, cut_at_step=None):
    setting_folder = "/{}_param_setting".format(setting)
    all_runs = {}
    lr = -1
    assert os.path.isdir(find_in), ("\nERROR: {} is not a directory\n".format(find_in))
    for path, subdirs, files in os.walk(find_in):
        # print(path)
        for name in files:
            if name in ["log"] and setting_folder in path:
                file = os.path.join(path, name)
                run_num = int(file.split("_run")[0].split("/")[-1])
                before_step = None if cut_at_step is None else cut_at_step[run_num]
                #print(file)
                res = extract_from_single_run(file, key, label, before_step=before_step)
                # print(res)
                # print(file)
                if final_only:
                    # print("--", res)
                    res = res[-1]
                all_runs[run_num] = res
                lr = extract_from_single_run(file, "learning_rate", label, before_step=before_step)
    return all_runs, lr

# def extract_return_all(path, total=None, start=0):
#     if total is None:
#         all_param = os.listdir(path + "/0_run")
#         setting_list = []
#         for p in all_param:
#             idx = int(p.split("_param")[0])
#             setting_list.append(idx)
#         setting_list.sort()
#     else:
#         setting_list = list(range(start, total))

def extract_return_all(path, setting_list, search_lr=False, key="return"):
    if setting_list is None:
        all_param = os.listdir(path + "/0_run")
        setting_list = []
        for p in all_param:
            idx = int(p.split("_param")[0])
            setting_list.append(idx)
        # print(setting_list)
        setting_list.sort()
    all_sets = {}
    for setting in setting_list:

        res, lr = extract_from_setting(path, setting, key=key)
        if search_lr:
            all_sets["{}_{}".format(setting, lr)] = res
        else:
            all_sets[setting] = res
    return all_sets

def extract_property_single_run(file, keyword):
    with open(file, "r") as f:
        content = f.readlines()
    for i in range(len(content)-1, -1, -1):
        l = content[i]
        if keyword in l:
            value = l.split(keyword)[-1]
            if "/" in value:
                value = value.split("/")[0]
                # print(value, i, len(content))
            return float(value)

def extract_property_setting(path, setting, file_name, keyword):
    find_in = path + "/"
    setting_folder = "{}_param_setting".format(setting)
    all_runs = {}
    for path, subdirs, files in os.walk(find_in):
        for name in files:
            if name==file_name and setting_folder in path:
                file = os.path.join(path, name)
                value = extract_property_single_run(file, keyword)
                all_runs[int(file.split("_run")[0].split("/")[-1])] = value
    return all_runs

def load_online_property(group, target_key, reverse=False, normalize=False, cut_at_step=None, p_label=None, fixed_rep=False):
    all_property = {}
    temp = []
    #print(group)
    for i in group:
        cas = cut_at_step[i["label"]] if cut_at_step else None
        if target_key in ["return"]:
            path = i["control"]
            if type(path) == list:
                setting = path[1]
                path = path[0]
            else:
                setting = 0
            returns,_ = extract_from_setting(path, setting, target_key, final_only=False)
            values = {}
            for run in returns:
                values[run] = np.array(returns[run]).sum()

        elif target_key in ["interf"]:
            if fixed_rep:
                path = i["fixrep_measure"]
                if type(path) == list:
                    setting = path[1]
                    path = path[0]
                else:
                    setting = 0
                values, _ = extract_from_setting(path, setting, target_key, final_only=True)
            else:
                path = i["online_measure"]
                if type(path) == list:
                    setting = path[1]
                    path = path[0]
                else:
                    setting = 0
                
                returns, _ = extract_from_setting(path, setting, target_key, final_only=False, cut_at_step=cas)
                values = {}
                for run in returns:
                    t = np.array(returns[run])[1:] # remove the first measure, which is always nan
                    pct = np.percentile(t, 90)
                    target_idx = np.where(t >= pct)[0]
                    values[run] = np.mean(t[target_idx]) # average over the top x percentiles only

        else:
            # print('case: ', cas)
            path = i["online_measure"]
            if type(path) == list:
                setting = path[1]
                path = path[0]
            else:
                setting = 0
            values, _ = extract_from_setting(path, setting, target_key, final_only=True, cut_at_step=cas, label=p_label)

        all_property[i["label"]] = values

        for run in values:
            temp.append(values[run])

    if (reverse or normalize):
        outlier_remove = False
        # mx = np.max(np.array(temp))
        # mn = np.min(np.array(temp))
        # print(target_key, mn, mx)
        # if target_key == "interf":
        #     srt = np.array(temp).argsort()
        #     mx = np.array(temp)[srt[-2]]
        mn = float('+inf')
        mx = float('-inf')
        max_group = ''
        min_group = ''
        for i in group:
            for run in all_property[i["label"]]:
                ori = all_property[i["label"]][run]

                if 'DA+O' in i["label"] or 'CR+O' in i["label"]:
                    print(i["label"], ' ', ori)

                if ori > mx:
                    mx = ori
                    max_group = i["label"]
                #print(all_property[i["label"]][run])
                if ori < mn:
                    mn = ori
                    min_group = i["label"]

        # print(all_property)
        print(target_key)
        print('mn: ', mn, ' mx: ', mx)
        print('mn group: ', min_group, ' mx group: ', max_group)


        for i in group:
            for run in all_property[i["label"]].keys():
                ori = all_property[i["label"]][run]
                if normalize:
                    mn, mx = normalize_prop[target_key]
                    
                    #print('mn, mx: ', mn, mx)
                # if ori> mx:
                #     print('shit')
                #     print(ori)
                if normalize and reverse:
                    all_property[i["label"]][run] = 1.0 - (ori - mn) / (mx - mn)
                    # all_property[i["label"]][run] = - ori 
                elif normalize:
                    all_property[i["label"]][run] = (ori - mn) / (mx - mn)
                elif reverse:
                    all_property[i["label"]][run] = 1.0 - ori#(ori - mn) / (mx - mn)
                if all_property[i["label"]][run] < 0:
                    print(i["label"])
                    print(ori)
                    print()
                # # print(target_key, ori, mn, mx, all_property[i["label"]][run])
                # if target_key == "interf" and all_property[i["label"]][run] <= 0:
                #     base = [i["label"], run]
                #     outlier_remove = True
        # if outlier_remove:
        #     del all_property[base[0]][base[1]]

    return all_property

def confidence_interval(path, setting, file_name, keyword):
    res_dict = extract_property_setting(path, setting, file_name, keyword)

    all_res = []
    for i in res_dict:
        res = res_dict[i]
        if not np.isnan(res):
            all_res.append(res)
        else:
            print("{}: {}th run is NaN or does not exist".format(path, i))

    all_res = np.array(all_res)
    mu = all_res.mean()
    std = all_res.std()
    interval = [mu-std*2, mu+std*2]
    print("{}: {:.3f}, [{:.3f}]".format(path.split("/")[-3:-1], mu, std))
    return mu, interval[0], interval[1], all_res

def violin_plot(ax1, color, data, xpos, width, normalize=False):
    if normalize:
        data_nomalize = []
        min = np.inf
        max = -1*np.inf
        for d in data:
            mn = d.min()
            mx = d.max()
            if mn < min:
                min = mn
            if mx > max:
                max = mx
        for d in data:
            d_nomalize = (d - min) / (max - min)
            data_nomalize.append(d_nomalize)
        data = data_nomalize

    violin_parts = ax1.violinplot(data, showmeans=False, showextrema=False, positions=xpos, widths=width)
    means = [np.mean(data[i]) for i in range(len(data))]#np.mean(np.array(data), axis=1)
    maxs = [np.max(data[i]) for i in range(len(data))]#np.max(np.array(data), axis=1)
    mins = [np.min(data[i]) for i in range(len(data))]#np.min(np.array(data), axis=1)

    for i in range(len(violin_parts['bodies'])):
        violin_parts['bodies'][i].set_facecolor(color)
        violin_parts['bodies'][i].set_alpha(0.5)
        # ax1.scatter(xpos[i], means[i], marker='_', color=color, s=20, zorder=10)
        ax1.scatter(xpos[i], maxs[i], marker='.', color=color, s=10, zorder=10)
        ax1.scatter(xpos[i], mins[i], marker='.', color=color, s=10, zorder=10)
        ax1.vlines(xpos[i], mins[i], maxs[i], color, linestyle='-', lw=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

def box_plot(ax1, color, data, xpos, width):
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        for patch in bp['boxes']:
            patch.set(facecolor=color, alpha=0.3)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
        plt.setp(bp["fliers"], markeredgecolor=color, marker=".")

    bp = ax1.boxplot(data, positions=xpos, widths=width, patch_artist=True)
    set_box_color(bp, color=color)

def draw_label(targets, save_path, ncol, emphasize=None, with_style=True, with_color=True):
    def get_linestyle(label):
        linestyle = curve_styles[label] if with_style else s_default[0]
        return linestyle

    def get_color(label):
        color = violin_colors[label] if with_color else "C0"
        return color

    plt.figure(figsize=(0.1, 2))
    if emphasize:
        for label in emphasize:
            plt.plot([], color=get_color(label), linestyle=get_linestyle(label), label=label, alpha=1, linewidth=2)
    for label in targets:
        if type(label) == dict:
            key = list(label.keys())[0]
            plt.plot([], color=label[key][0], linestyle=label[key][1], label=key)
        else:
            if emphasize and label not in emphasize:
                plt.plot([], color=get_color(label), linestyle=get_linestyle(label), label=label, alpha=0.4)
            elif emphasize and label in emphasize:
                pass
            else:
                plt.plot([], color=get_color(label), linestyle=get_linestyle(label), label=label)
    plt.axis('off')
    plt.legend(ncol=ncol)
    plt.savefig("plot/img/{}.pdf".format(save_path), dpi=300, bbox_inches='tight')
    # plt.savefig("plot/img/{}.png".format(save_path), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()

def exp_smooth(ary, alpha):
    new = np.zeros(len(ary))
    for j in range(len(ary)):
        new[j] = ary[j]
        if not np.isnan(ary[j]):
            break
    for i in range(j+1, len(ary)):
        new[i] = alpha * ary[i] + (1-alpha) * new[i-1]
        # print(alpha, new[i-1:i+1], ary[i])
    return new

def load_property(all_groups, property_key=None, perc=None, relationship=None, targets=[], early_stopped=False, p_label=None, fix_rep=False):
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
    normalize = True if property_key in ["lipschitz", "interf"] else False # normalize interference and lipschitz, for non-interference and complexity reduction measure
    # normalize = True if property_key in ["lipschitz"] else False # normalize interference and lipschitz, for non-interference and complexity reduction measure
    model_saving = load_info(all_group_dict, 0, "model", path_key="online_measure") if early_stopped else None
    # print('model saving: ', model_saving)
    properties = load_online_property(all_group_dict, property_key, reverse=reverse, normalize=normalize, cut_at_step=model_saving, p_label=p_label, fixed_rep=fix_rep)


    return properties, all_group_dict


def load_property_in_step(all_groups, property_key=None, perc=None, relationship=None, targets=[], early_stopped=False, p_label=None, fix_rep=False, step=10000):
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
    normalize = True if property_key in ["lipschitz", "interf"] else False # normalize interference and lipschitz, for non-interference and complexity reduction measure
    # normalize = True if property_key in ["lipschitz"] else False # normalize interference and lipschitz, for non-interference and complexity reduction measure
    model_saving = load_info(all_group_dict, 0, "model", path_key="online_measure") if early_stopped else None
    print(model_saving)
    for agent_key in model_saving:
        for run_key in model_saving[agent_key]:
            print(agent_key, run_key)
            model_saving[agent_key][run_key][0] = step
    
    #print('model saving: ', model_saving)

        
    properties = load_online_property(all_group_dict, property_key, reverse=reverse, normalize=normalize, cut_at_step=model_saving, p_label=p_label, fixed_rep=fix_rep)


    return properties, all_group_dict


def correlation_calc(all_group_dict, control, properties, perc=None, relationship=None):
    labels = [i["label"] for i in all_group_dict]

    # all reps:
    all_control = []
    all_property = []
    indexs = [0]
    for idx in range(len(labels)):
        rep = labels[idx]
        assert len(rep.split("_")) == 2
        ctr, prop = arrange_order_2(control[rep], properties[rep])
        all_control += ctr
        all_property += prop
        assert len(ctr) == len(prop)
        indexs.append(len(ctr)+indexs[-1])

    all_control = np.array(all_control)
    all_property = np.array(all_property)
    all_control, all_property = perform_filter(all_control, all_property, perc=perc)
    all_control, all_property = perform_transformer(all_control, all_property, relationship=relationship)
    cor = np.corrcoef(all_control, all_property)[0][1]
    return cor

def correlation_load(all_paths_dict, goal_ids, total_param=None, xlim=[],
                     property_keys = {"lipschitz": "Complexity Reduction", "distance": "Dynamics Awareness", "ortho": "Orthogonality", "interf":"Noninterference", "diversity":"Diversity", "sparsity":"Sparsity"},
                     property_perc = None, property_rank = None, get_overall=False):
    labels = [i["label"] for i in all_paths_dict]

    # all_goals_auc = pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels)
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
        g_path = copy.deepcopy(all_paths_dict)
        for i in range(len(all_paths_dict)):
            label = g_path[i]["label"]
            best_param_folder = all_goals_auc[goal][label][1]
            best = int(best_param_folder.split("_")[0])
            g_path[i]["control"] = [g_path[i]["control"].format(goal), best]
        formated_path[goal] = g_path

    allp_properties = {}
    for pk in property_keys.keys():
        properties, _ = load_property([formated_path[goal]], property_key=pk, early_stopped=True)
        allp_properties[pk] = copy.deepcopy(properties)
    allg_transf_perf = {}
    for goal in goal_ids:
        transf_perf, temp_lables = load_property([formated_path[goal]], property_key="return", early_stopped=True)
        allg_transf_perf[goal] = copy.deepcopy(transf_perf)

    allg_transf_perf, allp_properties = property_filter(allg_transf_perf, allp_properties, property_perc, property_rank)
    filtered_lables = []
    for obj in temp_lables:
        if obj["label"] in allg_transf_perf[106].keys():
            filtered_lables.append(obj)

    all_goals_cor = {}
    overall_cor = {}
    for pk in property_keys.keys():
        all_goals_cor[pk] = {}

        avg_transf_perf = {}
        for rep in allg_transf_perf[106].keys():
            avg_transf_perf[rep] = {}
            for run in allg_transf_perf[106][rep].keys():
                avg_transf_perf[rep][run] = []

        for goal in goal_ids:
            cor = correlation_calc(filtered_lables, allg_transf_perf[goal], allp_properties[pk], perc=None, relationship=None)
            all_goals_cor[pk][goal] = cor
            for rep in allg_transf_perf[goal].keys():
                for run in allg_transf_perf[goal][rep].keys():
                    avg_transf_perf[rep][run].append(allg_transf_perf[goal][rep][run])
        
        for rep in allg_transf_perf[106].keys():
            for run in allg_transf_perf[106][rep].keys():
                ary = np.array(avg_transf_perf[rep][run])
                avg_transf_perf[rep][run] = np.mean(ary)
        overall_cor[pk] = correlation_calc(filtered_lables, avg_transf_perf, allp_properties[pk], perc=None, relationship=None)
        
    if get_overall:
        return all_goals_cor, overall_cor, property_keys
    return all_goals_cor, property_keys

def pick_best_perfs(all_paths_dict, goal_ids, total_param, xlim, labels, top_runs=[0, 1.0], get_each_run=False):

    all_goals_auc = {}
    all_goals_independent = {}
    for goal in goal_ids:
        print("Loading auc from goal id {}".format(goal))
        single_goal_paths_dict = copy.deepcopy(all_paths_dict)
        for i in range(len(single_goal_paths_dict)):
            single_goal_paths_dict[i]["control"] = single_goal_paths_dict[i]["control"].format(goal)
        control = load_return(single_goal_paths_dict, total_param, search_lr=True)  # , start_param)

        rep_auc = {}
        independent_run = {}
        for idx, label in enumerate(labels):
            print("\n", idx, label)
            all_params = control[label]
            returns_rec = []
            auc_rec = []
            param_rec = []
            curve_rec = []
            for param, returns in all_params.items():
                returns = arrange_order(returns)
                mu, ste = get_avg(returns)
                if xlim != []:
                    mu, ste = mu[xlim[0]: xlim[1]], ste[xlim[0]: xlim[1]]
                    returns = returns[:, xlim[0]: xlim[1]]
                returns_rec.append(returns)
                auc_rec.append(np.sum(mu))
                param_rec.append(param)
                curve_rec.append([mu, ste])
            best_idx = np.argmax(auc_rec)
            best_param_folder = param_rec[best_idx].split("_")[0]
            best_param = param_rec[best_idx].split("_")[1]
            # if top_runs == 1.0:
            #     best_auc = auc_rec[best_idx]
            # else:
            #     best_return = returns_rec[best_idx]
            #     run_auc = np.sum(best_return, axis=1)
            #     sort_run = np.argsort(run_auc)
            #     take = int(np.ceil(len(run_auc) * top_runs))
            #     take_runs = best_return[sort_run[-take: ]]
            #     best_auc = np.average(take_runs, axis=0).sum()
            best_return = returns_rec[best_idx]
            run_auc = np.sum(best_return, axis=1)
            sort_run = np.argsort(run_auc)
            take_start = int(np.floor(len(run_auc) * top_runs[0]))
            take_end = int(np.ceil(len(run_auc) * top_runs[1]))
            take_runs = best_return[sort_run[take_start: take_end]]
            print("number of chosen runs", len(take_runs))
            best_auc = np.average(take_runs, axis=0).sum()

            rep_auc[label] = [best_auc, best_param_folder, best_param]
            independent_run[label] = take_runs
            print("{}, best param {}".format(label, best_param))
        all_goals_auc[goal] = rep_auc
        all_goals_independent[goal] = independent_run
    if get_each_run:
        return all_goals_auc, all_goals_independent
    return all_goals_auc

def label_filter(targets, all_paths):
    filtered = []
    for item in all_paths:
        if item["label"] in targets:
            filtered.append(item)
    return filtered

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    # _, y1 = ax1.transData.transform((0, v1))
    # _, y2 = ax2.transData.transform((0, v2))
    # inv = ax2.transData.inverted()
    # _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    # miny, maxy = ax2.get_ylim()
    # ax2.set_ylim(miny+dy, maxy+dy)
    ax1_ylim = ax1.get_ylim()
    ax2_ylim = ax2.get_ylim()
    prop = (v1 - ax1_ylim[0]) / (ax1_ylim[1] - ax1_ylim[0])
    # new_y_low = v2 - (ax2_ylim[1] - ax2_ylim[0]) * prop
    # new_y_high = new_y_low + (ax2_ylim[1] - ax2_ylim[0])
    # ax2.set_ylim(new_y_low, new_y_high)
    new_y_high = ax2_ylim[0] + (v2 - ax2_ylim[0]) / prop
    ax2.set_ylim(ax2_ylim[0], new_y_high)