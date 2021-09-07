import os
import re
import numpy as np
import matplotlib.pyplot as plt
from plot.plot_paths import violin_colors, curve_styles

from plot_paths import *

flatten = lambda t: [item for sublist in t for item in sublist]


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
        if len(v1) in [1, 11, 16, 31, 101, 151] or len(v1) > 25:
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

def load_info(paths, param, key, label=None, path_key="control"):
    all_rt = {}
    for i in paths:
        path = i[path_key]
        res, _ = extract_from_setting(path, param, key, label=label)
        all_rt[i["label"]] = res
    return all_rt

# def load_return(paths, total_param, start_param):
def load_return(paths, setting_list, search_lr=False):
    all_rt = {}
    for i in paths:
        path = i["control"]
        # print("Loading returns from", path)
        returns = extract_return_all(path, setting_list, search_lr=search_lr)#total_param, start_param)
        all_rt[i["label"]] = returns
    return all_rt

def get_avg(all_res):
    mu = all_res.mean(axis=0)
    # mu = all_res.max(axis=0)
    std = all_res.std(axis=0)
    # ste = std / np.sqrt(len(std))
    ste = std / np.sqrt(all_res.shape[0])
    return mu, ste

def draw_curve(all_res, ax, label, color=None, style="-", alpha=1, linewidth=1.5):
    if len(all_res) == 0:
        return None
    mu, ste = get_avg(all_res)
    if color is None:
        p = ax.plot(mu, label=label, alpha=alpha, linestyle=style, linewidth=linewidth)
        # color = p.get_color()
    else:
        ax.plot(mu, label=label, color=color, alpha=alpha, linestyle=style, linewidth=linewidth)
    ax.fill_between(list(range(len(mu))), mu - ste * 2, mu + ste * 2, color=color, alpha=0.1, linewidth=0.)
    print(label, "auc =", np.sum(mu))
    return mu

def draw_cut(cuts, all_res, ax, color, ymin):
    mu = all_res.mean(axis=0)
    x_mean = cuts.mean()
    x_max = cuts.max()
    ax.vlines(x_max, ymin, np.interp(x_max, list(range(len(mu))), mu), ls=":", colors=color, alpha=0.8, linewidth=3)

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
                    if current_step == before_step:
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
    setting_folder = "{}_param_setting".format(setting)
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
                res = extract_from_single_run(file, key, label, before_step=before_step)
                # print(res)
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

def extract_return_all(path, setting_list, search_lr=False):
    if setting_list is None:
        all_param = os.listdir(path + "/0_run")
        setting_list = []
        for p in all_param:
            idx = int(p.split("_param")[0])
            setting_list.append(idx)
        setting_list.sort()
    all_sets = {}
    for setting in setting_list:
        res, lr = extract_from_setting(path, setting)
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

def load_online_property(group, target_key, reverse=False, normalize=False, cut_at_step=None, p_label=None):
    all_property = {}
    temp = []
    for i in group:
        cas = cut_at_step[i["label"]] if cut_at_step else None

        if target_key in ["return"]:
            path = i["control"]
            returns,_ = extract_from_setting(path, 0, target_key, final_only=False)
            values = {}
            for run in returns:
                values[run] = np.array(returns[run]).sum()

        elif target_key in ["interf"]:
            path = i["online_measure"]
            returns, _ = extract_from_setting(path, 0, target_key, final_only=False, cut_at_step=cas)
            values = {}
            for run in returns:
                t = np.array(returns[run])[1:] # remove the first measure, which is always nan
                pct = np.percentile(t, 90)
                target_idx = np.where(t >= pct)[0]
                values[run] = np.mean(t[target_idx]) # average over the top x percentiles only

        else:
            path = i["online_measure"]
            values, _ = extract_from_setting(path, 0, target_key, final_only=True, cut_at_step=cas, label=p_label)

        all_property[i["label"]] = values

        for run in values:
            temp.append(values[run])

    if reverse or normalize:
        mx = np.max(np.array(temp))
        mn = np.min(np.array(temp))
        for i in group:
            for run in all_property[i["label"]]:
                ori = all_property[i["label"]][run]
                if reverse:
                    all_property[i["label"]][run] = 1.0 - (ori - mn) / (mx - mn)
                if normalize:
                    all_property[i["label"]][run] = (ori - mn) / (mx - mn)

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

def draw_label(targets, save_path, ncol):
    plt.figure(figsize=(0.1, 2))
    for label in targets:
        plt.plot([], color=violin_colors[label], linestyle=curve_styles[label], label=label)
    plt.axis('off')
    plt.legend(ncol=ncol)
    # plt.savefig("plot/img/{}.pdf".format(save_path), dpi=300, bbox_inches='tight')
    plt.savefig("plot/img/{}.png".format(save_path), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()