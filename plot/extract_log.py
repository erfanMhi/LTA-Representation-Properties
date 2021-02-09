import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def violin_plot(ax1, colors, all_d, normalize=False):
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

    stes = [all_d[i].std()/np.sqrt(len(all_d[i])) for i in range(len(all_d))]

    for i in range(len(violin_parts['bodies'])):
        violin_parts['bodies'][i].set_facecolor(colors[i])
        ax1.scatter([i+1], means[i], marker='_', color=colors[i], s=150, zorder=10)
        ax1.scatter([i+1], maxs[i], marker='.', color=colors[i], s=50, zorder=10)
        ax1.scatter([i+1], mins[i], marker='.', color=colors[i], s=50, zorder=10)
        ax1.vlines([i+1], mins[i], maxs[i], colors[i], linestyle='-', lw=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.savefig("/home/han/Pictures/{}_{}_{}.pdf".format(exp, env, file_name), dpi=300, format='pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    return means, stes

## online property measure, extract the last evaluation and make violin plots
def extract_final_eval(root, rep_list, param=None):
    returns_all, lip_all, div_all, dec_all, distance_all, ortho_all, noninterf_all = extract_log(root, rep_list, param=param, plot=False)
    returns_lst = []
    lip_lst = []
    div_lst = []
    dec_lst = []
    distance_lst = []
    ortho_lst = []
    noninterf_lst = []
    color_lst = []
    for rep in rep_list:
        # returns_lst.append(np.array(returns_all[rep])[:, -1])
        lip_lst.append(np.array(lip_all[rep])[:, -1])
        div_lst.append(np.array(div_all[rep])[:, -1])
        # dec_lst.append(np.array(dec_all[rep])[:, -1])
        # distance_lst.append(np.array(distance_all[rep])[:, -1])
        # ortho_lst.append(np.array(ortho_all[rep])[:, -1])
        # noninterf_lst.append(np.array(noninterf_all[rep])[:, -1])
        color_lst.append(colors[path2label[rep]])

    figsize = (30, 3)
    plt.figure(figsize=figsize)
    fig, axs = plt.subplots(nrows=1, ncols=6)
    fig.set_figwidth(figsize[0])
    fig.tight_layout(pad=6)

    # violin_plot(axs[0], color_lst, returns_lst)
    means, stes = violin_plot(axs[0], color_lst, lip_lst)
    axs[0].set_title("Lipschitz")
    print("Lipschitz mean", means, "(ste", stes, ")")

    means, stes = violin_plot(axs[1], color_lst, div_lst)
    axs[1].set_title("Specialization")
    print("Specialization mean", means, "(ste", stes, ")")

    # violin_plot(axs[2], color_lst, dec_lst)
    # axs[2].set_title("Decorrelation")
    #
    # violin_plot(axs[3], color_lst, distance_lst)
    # axs[3].set_title("Distance")
    #
    # violin_plot(axs[4], color_lst, ortho_lst)
    # axs[4].set_title("Orthogonal")
    #
    # violin_plot(axs[5], color_lst, noninterf_lst)
    # axs[5].set_title("Non-Intereference")

    # fig.suptitle(root.split("/")[-1])
    # plt.show()

    plt.tight_layout()
    plt.show()
    # plt.savefig("../../Pictures/{}.png".format(root.split("/")[-2]), bbox_inches='tight')


## online property measure, extract evaluations by time
def extract_log(root, rep_list, param=None, plot=True, len_record=[16, 31, 51, 101], with_measure=True):
    lr = {}
    returns_all = {}
    lip_all = {}
    div_all = {}
    dec_all = {}
    distance_all = {}
    ortho_all = {}
    noninterf_all = {}

    for rep in rep_list:
        lr[rep] = []
        returns_all[rep] = []
        lip_all[rep] = []
        div_all[rep] = []
        dec_all[rep] = []
        distance_all[rep] = []
        ortho_all[rep] = []
        noninterf_all[rep] = []
        find_in = root + rep + "/"
        for path, subdirs, files in os.walk(find_in):
            for name in files:
                if name in ["log"] and \
                        (param is None or (param is not None and param == path.split("/")[-1].split("_")[0])):
                    file = os.path.join(path, name)
                    with open(file, "r") as f:
                        content = f.readlines()

                    returns = []
                    lip = []
                    div = []
                    dec = []
                    distance = []
                    ortho = []
                    noninterf = []

                    for l in content:
                        info = l.split("|")[1].strip()
                        i_list = info.split(" ")
                        if "learning_rate:" == i_list[0]:
                            lr[rep].append(float(i_list[1]))

                        if "total" == i_list[0]:
                            if "returns" in i_list:
                                returns.append(float(i_list[i_list.index("returns")+1].split("/")[0].strip())) # mean
                            elif "Lipschitz:" in i_list:
                                lip.append(float(i_list[i_list.index("Lipschitz:")+1].split("/")[1].strip())) # mean
                            elif "Specialization:" in i_list:
                                record = float(i_list[i_list.index("Specialization:") + 1].split("/")[0].strip())
                                if np.isnan(record) or np.isinf(record):
                                    record = 0
                                div.append(record) # mean
                            # elif "Specialization:" in i_list: # Diversity
                            #     record = 1 - float(i_list[i_list.index("Specialization:")+1].split("/")[0].strip())
                            #     if np.isnan(record) or np.isinf(record):
                            #         record = 0
                            #     div.append(record) # mean
                            elif "Decorrelation:" in i_list:
                                record = float(i_list[i_list.index("Decorrelation:")+1].split("/")[0].strip())
                                if np.isnan(record) or np.isinf(record):
                                    record = 0
                                dec.append(record) # mean
                            elif "Distance:" in i_list:
                                distance.append(float(i_list[i_list.index("Distance:")+1].split("/")[0]))
                            elif "Orthogonality:" in i_list:
                                ortho.append(float(i_list[i_list.index("Orthogonality:")+1].split("/")[0]))
                            elif "Noninterference:" in i_list:
                                record = float(i_list[i_list.index("Noninterference:")+1].split("/")[0])
                                if np.isnan(record) or np.isinf(record):
                                    record = 0
                                noninterf.append(record)

                    # print(len(returns), len(lip), len(div), len(distance), len(ortho), len(noninterf))
                    # if len(returns) == 16 or len(returns) == 31 or len(returns) == 51 or len(returns) == 101:
                    if len(returns) in len_record:
                        returns_all[rep].append(returns)
                        lip_all[rep].append(lip)
                        div_all[rep].append(div)
                        dec_all[rep].append(dec)
                        distance_all[rep].append(distance)
                        ortho_all[rep].append(ortho)
                        noninterf_all[rep].append(noninterf)

    if plot and with_measure:
        fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(30, 4))
        for rep in rep_list:
            all_res = np.array(returns_all[rep])
            learning_curve(all_res, rep, axs[0])
        axs[0].set_title("Return")

        for rep in rep_list:
            all_res = np.array(lip_all[rep])
            learning_curve(all_res, rep, axs[1])
        axs[1].set_title("Lipschitz")

        for rep in rep_list:
            all_res = np.array(div_all[rep])
            learning_curve(all_res, rep, axs[2])
        # axs[2].set_title("Diversity")
        axs[2].set_title("Specialization")

        for rep in rep_list:
            all_res = np.array(dec_all[rep])
            learning_curve(all_res, rep, axs[3])
        axs[3].set_title("Decorrelation")

        for rep in rep_list:
            all_res = np.array(distance_all[rep])
            learning_curve(all_res, rep, axs[4])
        axs[4].set_title("Distance")

        for rep in rep_list:
            all_res = np.array(ortho_all[rep])
            learning_curve(all_res, rep, axs[5])
        axs[5].set_title("Orthogonal")

        for rep in rep_list:
            all_res = np.array(noninterf_all[rep])
            learning_curve(all_res, rep, axs[6])
        axs[6].set_title("Non-Interference")

        # plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig("../../Pictures/{}.png".format(root.split("/")[-2]), bbox_inches='tight')
    elif plot and not with_measure:
        plt.figure()
        for rep in rep_list:
            all_res = np.array(returns_all[rep])
            print(len(all_res), "====", rep)
            learning_curve(all_res, rep, plt)
        plt.title("Return")
        plt.legend()
        plt.show()
    else:
        return returns_all, lip_all, div_all, dec_all, distance_all, ortho_all, noninterf_all

def learning_curve(all_res, rep, ax, cut=None, label=None):
    if cut is None:
        cut = all_res.shape[1]
    mu = all_res.mean(axis=0)[:cut]
    std = all_res.std(axis=0)[:cut]
    # ste = std / np.sqrt(len(std))
    ste = std / np.sqrt(all_res.shape[0])
    if label is None:
        l = path2label[rep]
        c = colors[path2label[rep]]
    else:
        l = label
        c = None
    ax.plot(mu, label=l, color=c)
    ax.fill_between(list(range(len(mu))), mu-ste*2, mu+ste*2, color=colors[path2label[rep]], alpha=0.1, linewidth=0.)


def param_sweep(root, rep_list, all_setting):
    returns_all = {}
    for rep in rep_list:
        returns_all[rep] = {}
        find_in = root + rep + "/"
        assert os.path.isdir(find_in), print(find_in, "is not a dir")
        for param in all_setting:
            param = str(param)
            returns_all[rep][param] = []
            for path, subdirs, files in os.walk(find_in):
                for name in files:
                    if name in ["log"] and \
                            (param == path.split("/")[-1].split("_")[0]):
                        file = os.path.join(path, name)
                        with open(file, "r") as f:
                            content = f.readlines()

                        returns = []
                        for l in content:
                            if "|" not in l: continue
                            info = l.split("|")[1].strip()
                            i_list = info.split(" ")
                            if "learning_rate:" == i_list[0]:
                                lr = i_list[1]

                            if "total" == i_list[0]:
                                if "returns" in i_list:
                                    returns.append(float(i_list[i_list.index("returns")+1].split("/")[0].strip())) # mean

                        # if len(returns) >26:
                        #     returns_all[rep][param].append(returns[:26])

                        if len(returns) in [11, 16, 31, 51, 101]:
                            returns_all[rep][param].append(returns)
                        else:
                            print("Bad length", rep, param, len(returns))

    fig, axs = plt.subplots(nrows=1, ncols=len(rep_list), figsize=(3*len(rep_list), 3))

    axs = [axs] if type(axs)!=np.ndarray else axs
    for i, rep in enumerate(rep_list):
        for k in returns_all[rep].keys():
            print(len(returns_all[rep][k]), rep, k)
            learning_curve(np.array(returns_all[rep][k]), rep, axs[i], label=k, cut=None)
        axs[i].set_title(path2label[rep])
        axs[i].legend(loc="upper right")
    plt.tight_layout()
    plt.show()

## extract loss of laplace representation training
def extract_loss(root, settings_range, num_runs):
    for setting in range(settings_range[0], settings_range[1]):
        all_total = []
        all_attractive = []
        all_repulsive = []
        all_rwds = []
        for run in range(num_runs):
            total = []
            attractive = []
            repulsive = []
            rwds = []
            logfile = "{}/{}_run/{}_param_setting/log".format(root, run, setting)
            with open(logfile, "r") as f:
                lines = f.readlines()
            for line in lines:
                if len(line.split("loss")) in [2,3]:
                    tar = line.split("loss")[1].split("(total/attractive/repuslive")[0]
                    t, a, r = tar.split("/")[:3]
                    total.append(float(t))
                    attractive.append(float(a))
                    repulsive.append(float(r))
                    if len(line.split("loss")) == 3:
                        rwd = tar.split("/")[3]
                        rwds.append(float(rwd))

            all_total.append(total)
            all_attractive.append(attractive)
            all_repulsive.append(repulsive)
            if rwds != []:
                all_rwds.append(rwds)

        all_total = np.array(all_total)
        all_attractive = np.array(all_attractive)
        all_repulsive = np.array(all_repulsive)
        all_rwds = np.array(all_rwds)

        fig, ax1 = plt.subplots()
        lns1 = ax1.plot(all_total.mean(axis=0), label="total", color="black")
        lns2 = ax1.plot(all_repulsive.mean(axis=0), "--", label="repulsive", color="black")
        ax1.set_ylabel('total & repulsive')
        ax2 = ax1.twinx()
        ax2color = "orange"
        lns3 = ax2.plot(all_attractive.mean(axis=0), "--", label="attractive", color=ax2color)
        ax2.tick_params(axis='y', labelcolor=ax2color)
        ax2.set_ylabel('attractive', color=ax2color)

        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=3)

        print(all_rwds.shape)
        if len(all_rwds) != 0:
            plt.figure()
            plt.plot(all_rwds.mean(axis=0))

    plt.show()

path2label = {
    "dqn/best": "No auxiliary",
    "dqn/best_6g": "No auxiliary",
    "dqn_aux/aux_control/best": "Single-goal control",
    "dqn_aux/aux_control/best_1g": "Single-goal control",
    "dqn_aux/aux_control/best_5g": "All-goals control",
    "dqn_aux/input_decoder/best": "Input-decoder prediction",
    "dqn_aux/input_decoder/best_longer": "Input-decoder prediction",

    "dqn_aux/nas_v2_delta/best": "Next-agent-state prediction",
    "dqn_aux/nas_v2/best": "Next-agent-state prediction (as)",
    "dqn_aux/nas_v2_reward/best": "Next-agent-state prediction (as+r)",

    "dqn_aux/successor_as/best_v3": "Successor-feature prediction",
    "dqn_aux/successor_as/best": "Successor-feature prediction",
    "dqn_aux/successor_as/vanilla": "Successor-feature prediction (sf)",
    "dqn_aux/successor_nasdelta/best": "Successor-feature prediction (sf+asdelta)",

    "dqn_aux/info/best": "Expert-xy prediction",
    "dqn_aux/info/best_xy": "Expert-xy prediction",
    "dqn_aux/info/best_xy+color": "Expert-(xy+color) prediction",
    "dqn_aux/info/best_xy+count": "Expert-(xy+count) prediction",

    "dqn/sweep": "No auxiliary",
    "dqn/sweep_1g": "No auxiliary",
    "dqn/sweep_6g": "No auxiliary",
    "dqn_aux/aux_control/sweep": "Single-goal control",
    "dqn_aux/aux_control/sweep_1g": "Single-goal control",
    "dqn_aux/aux_control/sweep_5g": "All-goals control",
    "dqn_aux/input_decoder/sweep": "Input-decoder prediction",
    "dqn_aux/input_decoder/sweep_longer": "Input-decoder prediction",
    "dqn_aux/nas_v2_delta/sweep": "Next-agent-state prediction",
    "dqn_aux/nas_v2_delta/sweep_1g": "Next-agent-state prediction",
    "dqn_aux/successor_as/sweep": "Successor-feature prediction",
    "dqn_aux/successor_as/sweep_vanilla": "Successor-feature prediction",
    "dqn_aux/successor_as/sweep_v3": "Successor-feature prediction",
    # "dqn_aux/xy/sweep": "Expert-(xy+color) prediction",
    # "dqn_aux/xy/sweep_1g": "Expert-xy prediction",
    # "dqn_aux/xy/sweep_xy": "Expert-xy prediction",
    # "dqn_aux/xy/sweep_count": "Expert-(xy+count) prediction",
    "dqn_aux/info/sweep": "Expert-xy prediction",
    "dqn_aux/info/sweep_xy": "Expert-xy prediction",
    "dqn_aux/info/sweep_xy+color": "Expert-(xy+color) prediction",
    "dqn_aux/info/sweep_xy+count": "Expert-(xy+count) prediction",

    "dqn_aux/nas_v2/sweep": "Next-agent-state prediction",
    "dqn_aux/nas_v2_reward/sweep": "Next-agent-state prediction",
    "dqn_aux/successor_nasdelta/sweep": "Successor-feature prediction",

    "laplace-control/sweep": "Laplace rep control",
    "laplace-control/sweep_largerVF": "Laplace rep control",
    "laplace_aux-control/info/sweep_rwd": "Laplace rep control",
    "laplace_aux-control/info/sweep_xy": "Laplace rep control",
    "laplace_aux-control/info/sweep_xy+color": "Laplace rep control",
    "laplace_aux-control/info/sweep_xy+count": "Laplace rep control",
    "laplace_aux-control/info/sweep_xy+rwd": "Laplace rep control",
    "laplace/sweep": "Laplace rep control",
    "laplace/sweep_nolip": "Laplace rep control",
    "laplace/best": "Laplace rep control",

    "dqn_sparse/sweep": "Sparse rep",
    "dqn_sparse/sparse0.05/sweep": "Sparse rep 0.05",
    "dqn_sparse/sparse0.1/sweep": "Sparse rep 0.1",
    "dqn_sparse/sparse0.2/sweep": "Sparse rep 0.2",
    "dqn_sparse/sparse0.4/sweep": "Sparse rep 0.4",
    "dqn_sparse_highdim/sparse0.1/sweep": "Sparse rep 0.1",
    "dqn_sparse_highdim/sparse0.2/sweep": "Sparse rep 0.2",
    "dqn_sparse_highdim/sparse0.4/sweep": "Sparse rep 0.4",
    "dqn_sparse_2conv/sparse0.05/sweep": "Sparse rep 0.05",
    "dqn_sparse_2conv/sparse0.1/sweep": "Sparse rep 0.1",
    "dqn_sparse_2conv/sparse0.2/sweep": "Sparse rep 0.2",
    "dqn_sparse_2conv/sparse0.4/sweep": "Sparse rep 0.4",
    "dqn_2conv/sweep": "No auxiliary",
    "input/sweep": "Input",
    "random/sweep": "Random",

    "dqn_sparse/best": "Sparse rep",
    "dqn_sparse/sparse0.05/best": "Sparse rep 0.05",
    "dqn_sparse/sparse0.1/best": "Sparse rep 0.1",
    "dqn_sparse/sparse0.2/best": "Sparse rep 0.2",
    "dqn_sparse/sparse0.4/best": "Sparse rep 0.4",
    "dqn_sparse_2conv/sparse0.05/best": "Sparse rep 0.05",
    "dqn_sparse_2conv/sparse0.1/best": "Sparse rep 0.1",
    "dqn_sparse_2conv/sparse0.2/best": "Sparse rep 0.2",
    "dqn_sparse_2conv/sparse0.4/best": "Sparse rep 0.4",
    "dqn_2conv/best": "No auxiliary",
}
colors = {
    "Baseline (from scratch)": "black",
    "DQN": "black",
    "No auxiliary": "purple",
    "Expert-xy prediction": "bisque",
    "Expert-(xy+color) prediction": "goldenrod",
    "Expert-(xy+count) prediction": "burlywood",

    "Input-decoder prediction": "brown",

    "Next-agent-state prediction": "orange",
    "Next-agent-state prediction (as)": "gold",
    "Next-agent-state prediction (as+r)": "palegoldenrod",

    "Successor-feature prediction": "green",
    "Successor-feature prediction (sf)": "lawngreen",
    "Successor-feature prediction (sf+asdelta)": "darkseagreen",

    "Pick-red control": "steelblue",
    "Single-goal control": "steelblue",
    "All-goals control": "dodgerblue",
    "control": "dodgerblue",

    "Random representation": "crimson",
    "NoRep": "slategray",

    "Laplace rep control": "grey",

    "Sparse rep": "black",
    "Sparse rep 0.05": "C0",
    "Sparse rep 0.1": "C1",
    "Sparse rep 0.2": "C2",
    "Sparse rep 0.4": "C3",

    "Random": "C4",
    "Input": "C5",
}

gridhard_learning_rep_list = [
    "dqn/best",
    # "dqn_aux/input_decoder/best_longer",
    "dqn_aux/input_decoder/best",
    "dqn_aux/xy/best",
    "dqn_aux/aux_control/best_1g",
    "dqn_aux/aux_control/best_5g",
    "dqn_aux/nas_v2_delta/best",
    "dqn_aux/successor_as/best",
    "dqn_aux/successor_as/vanilla",
    # "dqn_aux/nas_v2/best",
    # "dqn_aux/nas_v2_reward/best",
    # "dqn_aux/successor_nasdelta/best",
]

collect_learning_rep_list = [
    "dqn/best",
    "dqn_aux/input_decoder/best",
    "dqn_aux/xy/best",
    "dqn_aux/xy/best_count",
    "dqn_aux/xy/best_xy",
    "dqn_aux/aux_control/best",
    "dqn_aux/nas_v2_delta/best",
    "dqn_aux/successor_as/best_v2",
    "dqn_aux/successor_as/vanilla",
    # "dqn_aux/nas_v2/best",
    # "dqn_aux/nas_v2_reward/best",
    # "dqn_aux/successor_nasdelta/best",
]

gridhard_transfer_rep_list = [
    "dqn/best",
    # "dqn_sparse/best",
    "dqn_sparse/sparse0.05/best",
    "dqn_sparse/sparse0.1/best",
    "dqn_sparse/sparse0.2/best",
    "dqn_sparse/sparse0.4/best",
    # "dqn_aux/input_decoder/best_longer",
    "dqn_aux/input_decoder/best",
    "dqn_aux/info/best",
    "dqn_aux/aux_control/best_1g",
    "dqn_aux/aux_control/best_5g",
    "dqn_aux/nas_v2_delta/best",
    # "dqn_aux/successor_as/best_v3",
    "dqn_aux/successor_as/best",
    # "laplace/best"
]

collect_transfer_rep_list = [
    "dqn/best",
    "dqn_aux/input_decoder/best",
    "dqn_aux/xy/best",
    "dqn_aux/xy/best_count",
    "dqn_aux/xy/best_xy",
    "dqn_aux/aux_control/best",
    "dqn_aux/nas_v2_delta/best",
    "dqn_aux/successor_as/best_v2",
]

gridhard_sweep_rep_list = [
    "dqn/sweep",
    "dqn_aux/aux_control/sweep_1g",
    "dqn_aux/aux_control/sweep_5g",
    "dqn_aux/input_decoder/sweep",
    "dqn_aux/nas_v2_delta/sweep",
    "dqn_aux/successor_as/sweep",
    "dqn_aux/info/sweep",
    # # "dqn_sparse/sweep"
    # # "dqn_sparse/sparse0.05/sweep",
    "dqn_sparse_highdim/sparse0.1/sweep",
    "dqn_sparse_highdim/sparse0.2/sweep",
    "dqn_sparse_highdim/sparse0.4/sweep",
    # "input/sweep",
    "random/sweep"
]
collect_sweep_rep_list = [
    "dqn/sweep",
    # "dqn_aux/aux_control/sweep",
    # "dqn_aux/info/sweep_xy",
    # "dqn_aux/info/sweep_xy+color",
    # "dqn_aux/info/sweep_xy+count",
    # "dqn_aux/input_decoder/sweep",
    # "dqn_aux/nas_v2_delta/sweep",
    # "dqn_aux/successor_as/sweep",
    "dqn_sparse_highdim/sparse0.1/sweep",
    "dqn_sparse_highdim/sparse0.2/sweep",
    "dqn_sparse_highdim/sparse0.4/sweep",
    # "input/sweep",
    # "random/sweep"
]

## online property measure, extract evaluations by time
# extract_log("../data/output/tests/gridhard/representations/",
#             gridhard_learning_rep_list, with_measure=False)
# extract_log("../data/output/tests/gridhard/transfer_onlineProperty/same_task/",
#             gridhard_transfer_rep_list)
# extract_log("../data/output/tests/gridhard/transfer_onlineProperty/different_task/",
#             gridhard_transfer_rep_list)
# extract_log("../data/output/tests/gridhard/control/",
#             gridhard_transfer_rep_list, with_measure=False)

# extract_log("../data/output/tests/collect_two/representations/",
#             collect_learning_rep_list, with_measure=False)
# extract_log("../data/output/tests/gridhard/transfer_onlineProperty/same_task/",
#             collect_transfer_rep_list)
# extract_log("../data/output/tests/gridhard/transfer_onlineProperty/different_task/",
#             collect_transfer_rep_list)
# extract_log("../data/output/tests/collect_two/control/",
#             collect_transfer_rep_list, with_measure=False)

## online property measure, extract the last evaluation and make violin plots
# extract_final_eval("../data/output/tests/gridhard/representations_onlineProperty/",
#                    gridhard_learning_rep_list)
# extract_final_eval("../data/output/tests/gridhard/transfer_onlineProperty/same_task/",
#                    gridhard_transfer_rep_list)
# extract_final_eval("../data/output/tests/gridhard/control/different_task/fix_rep/",
#                    gridhard_transfer_rep_list)

# extract_final_eval("../data/output/tests/collect_two/representations_onlineProperty/",
#                    collect_learning_rep_list)
# extract_final_eval("../data/output/tests/collect_two/transfer_onlineProperty/same_task/",
#                    collect_transfer_rep_list)
# extract_final_eval("../data/output/tests/collect_two/control/different_task/fix_rep/",
#                    collect_transfer_rep_list)

# param_sweep("../data/output/tests/gridhard/transfer_onlineProperty/same_task/",
#             gridhard_sweep_rep_list, list(range(6)))
# param_sweep("../data/output/tests/gridhard/transfer_onlineProperty/same_task/",
#             gridhard_sweep_rep_list, list(range(6, 12)))

# param_sweep("../data/output/tests/collect_two/transfer_onlineProperty/same_task/",
#             collect_sweep_rep_list, list(range(8)))
# param_sweep("../data/output/tests/collect_two/transfer_onlineProperty/different_task/",
#             collect_sweep_rep_list, list(range(8)))

# param_sweep("../data/output/tests/gridhard/representations/",
#             gridhard_sweep_rep_list, list(range(5)))
# param_sweep("../data/output/tests/collect_two/representations/",
#             collect_sweep_rep_list, list(range(5)))

# param_sweep("../data/output/tests/gridhard/control/same_task/fix_rep/",
#             gridhard_sweep_rep_list, list(range(5)))
# param_sweep("../data/output/tests/collect_two/control/same_task/fix_rep/",
#             collect_sweep_rep_list, list(range(5)))
# param_sweep("../data/output/tests/collect_two/control/different_task/fine_tune/",
#             collect_sweep_rep_list, list(range(5)))

# param_sweep("../data/output/tests/gridhard/control/different_task/fix_rep/",
#             gridhard_transfer_rep_list, list(range(5)))
# param_sweep("../data/output/tests/collect_two/control/different_task/fix_rep/",
#             collect_transfer_rep_list, list(range(5)))

# param_sweep("../data/output/tests/gridhard/representations/",
#             ["laplace-control/sweep"], list(range(0, 15)))
# param_sweep("../data/output/tests/gridhard/control/same_task/fix_rep/",
#             ["laplace/sweep"], list(range(7)))
# param_sweep("../data/output/tests/gridhard/control/same_task/fix_rep/",
#             ["laplace/sweep_nolip"], list(range(3)))

# param_sweep("../data/output/tests/collect_two/representations/",
#             ["laplace-control/sweep"], list(range(15)))
# param_sweep("../data/output/tests/collect_two/representations/",
#             ["laplace_aux-control/info/sweep_rwd"], list(range(15)))
# param_sweep("../data/output/tests/collect_two/representations/",
#             ["laplace_aux-control/info/sweep_xy+rwd"], list(range(15)))
# param_sweep("../data/output/tests/collect_two/representations/",
#             ["laplace_aux-control/info/sweep_xy"], list(range(15)))
# param_sweep("../data/output/tests/collect_two/representations/",
#             ["laplace_aux-control/info/sweep_xy+color"], list(range(15)))
# param_sweep("../data/output/tests/collect_two/representations/",
#             ["laplace_aux-control/info/sweep_xy+count"], list(range(15)))

## online property measure, extract the last evaluation and make violin plots
# extract_final_eval("../data/output/tests/gridhard/control/same_task/fix_rep/",
#                    ["laplace/best"])
# extract_final_eval("../data/output/tests/gridhard/control/similar_task/fix_rep/",
#                    ["laplace/best"])
# extract_final_eval("../data/output/tests/gridhard/control/different_task/fix_rep/",
#                    ["laplace/best"])

## extract loss of laplace representation training
# extract_loss("../data/output/tests/gridhard/representations/laplace/best"
#              , [0, 1], 60)
# extract_loss("../data/output/tests/collect_two/representations/laplace/sweep_fewerData",
#             [15, 16], 1)
# extract_loss("../data/output/tests/collect_two/representations/laplace_rwd/sweep",
#             [24, 25], 1)

## extract return from log of control tasks
# extract_log("../data/output/tests/gridhard/control/same_task/fix_rep/",
#             gridhard_transfer_rep_list, with_measure=False)
# extract_log("../data/output/tests/gridhard/control/similar_task/fix_rep/",
#             gridhard_transfer_rep_list, with_measure=False)
# extract_log("../data/output/tests/gridhard/control/different_task/fix_rep/",
#             gridhard_transfer_rep_list, with_measure=False)

# extract_final_eval("../data/output/tests/gridhard/control/same_task/fix_rep/",
#                    gridhard_transfer_rep_list)
# extract_final_eval("../data/output/tests/gridhard/control/similar_task/fix_rep/",
#                    gridhard_transfer_rep_list)
# extract_final_eval("../data/output/tests/gridhard/control/different_task/fix_rep/",
#                    gridhard_transfer_rep_list)

# extract_log("../data/output/tests/collect_two/control/same_task/fix_rep/",
#             collect_transfer_rep_list, with_measure=False)
# extract_log("../data/output/tests/collect_two/control/different_task/fix_rep/",
#             collect_transfer_rep_list, with_measure=False)
# extract_log("../data/output/tests/collect_two/control/different_task/fine_tune/",
#             collect_transfer_rep_list, with_measure=False)

# extract_final_eval("../data/output/tests/collect_two/control/same_task/fix_rep/",
#                    collect_transfer_rep_list)
# extract_final_eval("../data/output/tests/collect_two/control/different_task/fix_rep/",
#                    collect_transfer_rep_list)
# extract_final_eval("../data/output/tests/collect_two/control/different_task/fine_tune/",
#                    collect_transfer_rep_list)


## linear value function
# param_sweep("../data/output/linear_vf/gridhard/representations/",
#             gridhard_sweep_rep_list, list(range(0, 5)))  # sparse: [26])
# param_sweep("../data/output/tests/gridhard/representations/",
#             ["laplace-control/sweep"], list(range(6, 15)))

# param_sweep("../data/output/linear_vf/gridhard/representations/",
#             [
#              "dqn_sparse_highdim/sparse0.1/sweep",
#              "dqn_sparse_highdim/sparse0.2/sweep",
#              "dqn_sparse_highdim/sparse0.4/sweep",
#              ], [2,14]) # sparse0.1: 2, sparse0.2: 2, sparse0.4: 2
# param_sweep("../data/output/linear_vf/collect_two/representations/",
#             [
#              "dqn_sparse_highdim/sparse0.1/sweep",
#              "dqn_sparse_highdim/sparse0.2/sweep",
#              "dqn_sparse_highdim/sparse0.4/sweep",
#              ], [5, 10, 11]) # sparse0.1: 5, sparse0.2: 5, sparse0.4: 5

# extract_log("../data/output/linear_vf/gridhard/representations/",
#             ["dqn/best",
#              "dqn_sparse/sparse0.05/best",
#              "dqn_sparse/sparse0.1/best",
#              "dqn_sparse/sparse0.2/best",
#              "dqn_sparse/sparse0.4/best",
#              ], with_measure=False)


# param_sweep("../data/output/linear_vf/gridhard/control/same_task/fix_rep/",
#             gridhard_sweep_rep_list, list(range(4)))
# param_sweep("../data/output/linear_vf/gridhard/control/similar_task/fix_rep/",
#             gridhard_sweep_rep_list, list(range(4)))
# param_sweep("../data/output/linear_vf/gridhard/control/different_task/fix_rep/",
#             gridhard_sweep_rep_list, list(range(4)))
param_sweep("../data/output/linear_vf/gridhard/control/different_task/fine_tune/",
            gridhard_sweep_rep_list, list(range(4)))



# param_sweep("../data/output/linear_vf/collect_two/representations/",
#             collect_sweep_rep_list, list(range(0, 5)))  # sparse:

# param_sweep("../data/output/linear_vf/collect_two/control/same_task/fix_rep/",
#             collect_sweep_rep_list, list(range(3)))
# param_sweep("../data/output/linear_vf/collect_two/control/different_task/fix_rep/",
#             collect_sweep_rep_list, list(range(3)))
# param_sweep("../data/output/linear_vf/collect_two/control/different_task/fine_tune/",
#             collect_sweep_rep_list, list(range(3)))

# extract_log("../data/output/linear_vf/gridhard/control/same_task/fix_rep/",
#             gridhard_transfer_rep_list, with_measure=False)
# extract_log("../data/output/linear_vf/gridhard/control/similar_task/fix_rep/",
#             gridhard_transfer_rep_list, with_measure=False)
# extract_log("../data/output/linear_vf/gridhard/control/different_task/fix_rep/",
#             gridhard_transfer_rep_list, with_measure=False)
