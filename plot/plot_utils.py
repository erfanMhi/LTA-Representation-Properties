import os
import numpy as np

flatten = lambda t: [item for sublist in t for item in sublist]

def draw_curve(all_res, ax, label, color):
    mu = all_res.mean(axis=0)
    std = all_res.std(axis=0)
    # ste = std / np.sqrt(len(std))
    ste = std / np.sqrt(all_res.shape[0])
    ax.plot(mu, label=label, color=color)
    ax.fill_between(list(range(len(mu))), mu-ste*2, mu+ste*2, color=color, alpha=0.1, linewidth=0.)

def extract_return_single_run(file):
    with open(file, "r") as f:
        content = f.readlines()
    returns = []
    for l in content:
        info = l.split("|")[1].strip()
        i_list = info.split(" ")
        if "total" == i_list[0]:
            if "returns" in i_list:
                returns.append(float(i_list[i_list.index("returns")+1].split("/")[0].strip())) # mean
    return returns

def extract_return_setting(find_in, setting):
    setting_folder = "{}_param_setting".format(setting)
    all_runs = {}
    assert os.path.isdir(find_in), print("\nERROR: {} is not a directory\n".format(find_in))
    for path, subdirs, files in os.walk(find_in):
        for name in files:
            if name in ["log"] and setting_folder in path:
                file = os.path.join(path, name)
                returns = extract_return_single_run(file)
                all_runs[int(file.split("_run")[0].split("/")[-1])] = returns
    return all_runs

def extract_return_all(path, setting_list):
    all_sets = {}
    for setting in setting_list:
        all_sets[setting] = extract_return_setting(path, setting)
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
