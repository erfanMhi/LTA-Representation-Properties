import os
import numpy as np
import matplotlib.pyplot as plt

flatten = lambda t: [item for sublist in t for item in sublist]


def arrange_order(dict1, cut_length=True, scale=1):
    lst = []
    min_l = np.inf
    for i in sorted(dict1):
        v1 = dict1[i]
        if len(v1) == 151:
            lst.append(v1)
        else:        
            print('Length: ', len(v1))
            print('Run: ', i)
            continue
# <<<<<<< HEAD
        l = len(v1)
        min_l = l if l < min_l else min_l
    # print("min length: ", min_l)
    # for i in range(len(lst)):
        # lst[i] = lst[i][:min_l]
    # return np.array(lst)
# =======
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
        res = extract_from_setting(path, param, key, label=label)
        all_rt[i["label"]] = res
    return all_rt

# def load_return(paths, total_param, start_param):
def load_return(paths, setting_list):
    all_rt = {}
    for i in paths:
        path = i["control"]
        # print("Loading returns from", path)
        returns = extract_return_all(path, setting_list)#total_param, start_param)
        all_rt[i["label"]] = returns
    return all_rt

def draw_curve(all_res, ax, label, color=None, style="-", alpha=1, linewidth=1.5):
    mu = all_res.mean(axis=0)
    std = all_res.std(axis=0)
    # ste = std / np.sqrt(len(std))
    ste = std / np.sqrt(all_res.shape[0])
    if color is None:
        p = ax.plot(mu, label=label, alpha=alpha, linestyle=style, linewidth=linewidth)
        color = p.get_color()
    else:
        ax.plot(mu, label=label, color=color, alpha=alpha, linestyle=style, linewidth=linewidth)
    ax.fill_between(list(range(len(mu))), mu - ste * 2, mu + ste * 2, color=color, alpha=0.1, linewidth=0.)
    print(label, "auc =", np.sum(mu))
    return mu

def draw_cut(cuts, all_res, ax, color, ymin):
    mu = all_res.mean(axis=0)
    x_mean = cuts.mean()
    x_max = cuts.max()
    # ax.vlines(x_mean, ymin, np.interp(x_mean, list(range(len(mu))), mu), ls="--", colors=color, alpha=0.5)
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
            # print('i_list: ', i_list[0])
            if "EVAL:" == i_list[0] or "total" == i_list[0] or "TRAIN" == i_list[0]:

                # print('i_list: ', i_list[0])
                if key=="return" and "returns" in i_list:
                    returns.append(float(i_list[i_list.index("returns")+1].split("/")[0].strip())) # mean
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

                # used only when extract property for early-stopping model
                if before_step is not None:
                    current_step = int(info.split("total steps")[1].split(",")[0])
                    if current_step == before_step:
                        return returns

            if "early-stopping" in i_list and key == "model":
                cut = num - 1 # last line
                converge_at = int(content[cut].split("total steps")[1].split(",")[0])

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
        if len(returns) in [0, 1] :
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
    print(find_in)
    assert os.path.isdir(find_in), print("\nERROR: {} is not a directory\n".format(find_in))
    for path, subdirs, files in os.walk(find_in):
        for name in files:
            if name in ["log"] and setting_folder in path:
                file = os.path.join(path, name)
                run_num = int(file.split("_run")[0].split("/")[-1])
                before_step = None if cut_at_step is None else cut_at_step[run_num]
                res = extract_from_single_run(file, key, label, before_step=before_step)
                if final_only:
                    # print("--", res)
                    res = res[-1]
                all_runs[run_num] = res
    return all_runs

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

def extract_return_all(path, setting_list):
    if setting_list is None:
        all_param = os.listdir(path + "/0_run")
        setting_list = []
        for p in all_param:
            idx = int(p.split("_param")[0])
            setting_list.append(idx)
        setting_list.sort()
    all_sets = {}
    for setting in setting_list:
        all_sets[setting] = extract_from_setting(path, setting)
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
            returns = extract_from_setting(path, 0, target_key, final_only=False)
            values = {}
            for run in returns:
                values[run] = np.array(returns[run]).sum()

        # elif target_key in []:  # ["lipschitz", "interf"]:
        #     path = i["control"]
        #     values = extract_from_setting(path, 0, target_key, final_only=True, cut_at_step=cas)

        elif target_key in ["interf"]:
            path = i["online_measure"]
            returns = extract_from_setting(path, 0, target_key, final_only=False, cut_at_step=cas)
            values = {}
            for run in returns:
                t = np.array(returns[run])[1:] # remove the first measure, which is always nan
                pct = np.percentile(t, 90)
                target_idx = np.where(t >= pct)[0]
                values[run] = np.mean(t[target_idx]) # average over the top x percentiles only

        else:
            path = i["online_measure"]
            values = extract_from_setting(path, 0, target_key, final_only=True, cut_at_step=cas, label=p_label)

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

