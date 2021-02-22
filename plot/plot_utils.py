import os
import numpy as np

flatten = lambda t: [item for sublist in t for item in sublist]


def arrange_order(dict1):
    lst = []
    min_l = np.inf
    for i in sorted(dict1):
        v1 = dict1[i]
        lst.append(v1)
        l = len(v1)
        min_l = l if l < min_l else min_l
    for i in range(len(lst)):
        lst[i] = lst[i][:min_l]
    return np.array(lst)

def load_info(paths, param, key):
    all_rt = {}
    for i in paths:
        path = i["control"]
        res = extract_from_setting(path, param, key)
        all_rt[i["label"]] = res
    return all_rt

def load_return(paths, total_param, start_param):
    all_rt = {}
    for i in paths:
        path = i["control"]
        # print("Loading returns from", path)
        returns = extract_return_all(path, total_param, start_param)
        all_rt[i["label"]] = returns
    return all_rt

def draw_curve(all_res, ax, label, color):
    mu = all_res.mean(axis=0)
    std = all_res.std(axis=0)
    # ste = std / np.sqrt(len(std))
    ste = std / np.sqrt(all_res.shape[0])
    ax.plot(mu, label=label, color=color)
    ax.fill_between(list(range(len(mu))), mu-ste*2, mu+ste*2, color=color, alpha=0.1, linewidth=0.)

def extract_from_single_run(file, key):
    with open(file, "r") as f:
        content = f.readlines()
    returns = []
    for l in content:
        info = l.split("|")[1].strip()
        i_list = info.split(" ")
        if "total" == i_list[0]:
            if key=="return" and "returns" in i_list:
                returns.append(float(i_list[i_list.index("returns")+1].split("/")[0].strip())) # mean
            elif key == "lipschitz" and "Lipschitz:" in i_list:
                returns.append(float(i_list[i_list.index("Lipschitz:") + 1].split("/")[1].strip()))  # mean
            elif key == "distance" and "Distance:" in i_list:
                returns.append(float(i_list[i_list.index("Distance:") + 1].split("/")[0].strip()))
            elif key == "otho" and "Orthogonality:" in i_list:
                returns.append(float(i_list[i_list.index("Orthogonality:") + 1].split("/")[0].strip()))
            elif key == "noninterf" and "Noninterference:" in i_list:
                returns.append(float(i_list[i_list.index("Noninterference:") + 1].split("/")[0].strip()))
            elif key == "decorr" and "Decorrelation:" in i_list:
                returns.append(float(i_list[i_list.index("Decorrelation:") + 1].split("/")[0].strip()))
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

def extract_from_setting(find_in, setting, key="return"):
    setting_folder = "{}_param_setting".format(setting)
    all_runs = {}
    assert os.path.isdir(find_in), print("\nERROR: {} is not a directory\n".format(find_in))
    for path, subdirs, files in os.walk(find_in):
        for name in files:
            if name in ["log"] and setting_folder in path:
                file = os.path.join(path, name)
                res = extract_from_single_run(file, key)
                all_runs[int(file.split("_run")[0].split("/")[-1])] = res
    return all_runs

def extract_return_all(path, total=None, start=0):
    if total is None:
        all_param = os.listdir(path+"/0_run")
        setting_list = []
        for p in all_param:
            idx = int(p.split("_param")[0])
            setting_list.append(idx)
        setting_list.sort()
    else:
        setting_list = list(range(start, total))

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

def violin_plot(ax1, colors, all_d, normalize=False):#(, xlabel, exp, env, file_name, normalize=False):
    if normalize:
        all_d_nomalize = []
        min = np.inf
        max = -1*np.inf
        for d in all_d:
            mn = d.min()
            mx = d.max()
            if mn < min:
                min = mn
            if mx > max:
                max = mx
        for d in all_d:
            d_nomalize = (d - min) / (max - min)
            all_d_nomalize.append(d_nomalize)
        all_d = all_d_nomalize

    violin_parts = ax1.violinplot(all_d, showmeans=False, showextrema=False)
    means = [np.mean(all_d[i]) for i in range(len(all_d))]#np.mean(np.array(all_d), axis=1)
    maxs = [np.max(all_d[i]) for i in range(len(all_d))]#np.max(np.array(all_d), axis=1)
    mins = [np.min(all_d[i]) for i in range(len(all_d))]#np.min(np.array(all_d), axis=1)

    for i in range(len(violin_parts['bodies'])):
        violin_parts['bodies'][i].set_facecolor(colors[i])
        ax1.scatter([i+1], means[i], marker='_', color=colors[i], s=150, zorder=10)
        ax1.scatter([i+1], maxs[i], marker='.', color=colors[i], s=50, zorder=10)
        ax1.scatter([i+1], mins[i], marker='.', color=colors[i], s=50, zorder=10)
        ax1.vlines([i+1], mins[i], maxs[i], colors[i], linestyle='-', lw=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
