import os
import pickle
import sys
import copy
import numpy as np
import itertools
from sklearn import ensemble, metrics
import scipy
import seaborn as sns
# from sklearn.inspection import permutation_importance
# from sklearn.linear_model import BayesianRidge

import matplotlib.pyplot as plt

sys.path.insert(0, '..')
print(sys.path)
from plot.plot_paths import *
from plot.plot_utils import *
from plot.radar_plot import radar_factory
from matplotlib.animation import FuncAnimation 

print("Change dir to", os.getcwd())

os.chdir("..")
print(os.getcwd())


def property_scatter_radar_circle(property_keys, all_paths_dict, groups, title):
    plt.style.use('ggplot')

    all_goals_prop = {}
    for pk in property_keys.keys():
        properties, _ = load_property([all_paths_dict], property_key=pk, early_stopped=True)
        all_goals_prop[pk] = properties

    nc = 3
    nr = int(np.ceil(len(property_keys.keys()) // nc))
    group_name = list(groups.keys())
    # fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(8, 3*nr))
    
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(polar=True)
    property_key = list(property_keys.keys())
    property_names = [property_keys[key] for key in property_key]
    angles=np.linspace(0,2*np.pi,len(property_names), endpoint=False)
    print(angles)
    angles=np.concatenate((angles,[angles[0]]))
    property_names.append(property_names[0])
    # if nr == 1:
    #     axes = [axes]
    for ni, name in enumerate(group_name):
        allp = []
        allprop_mean = []
        allprop_std = []
        for pi, pk in enumerate(property_key):
            same_color = groups[name]
            prop1 = []
            for rep in all_goals_prop[pk].keys():
                if "_".join(rep.split("_")[:-1]) in same_color:
                    for run in all_goals_prop[pk][rep].keys():
                        prop1.append(all_goals_prop[pk][rep][run])
                        # allp.append(all_goals_prop[pk][rep][run])
        
            allprop_mean.append(np.mean(prop1))
            allprop_std.append(np.std(prop1)/np.sqrt(len(prop1))*1.96)

            # axes[pi//nc][pi%nc].scatter(np.random.uniform(ni, ni+0.1, size=len(prop1)), prop1, s=6)
            # allprop.append(prop1)
        allprop_mean.append(allprop_mean[0])
        ax.plot(angles, allprop_mean, 'o--', color=c_default[ni], label=name)
        ax.fill(angles, allprop_mean, alpha=0.25, color=c_default[ni])
        #axes[pi//nc][pi%nc].boxplot(allprop)

        # axes[pi // nc][pi % nc].set_title(property_keys[pk])
        # allp = np.array(allp)
        # axes[pi // nc][pi % nc].set_yticks([allp.min(), allp.max()])
        # axes[pi // nc][pi % nc].set_yticklabels(["{:.2f}".format(allp.min()), "{:.2f}".format(allp.max())])
        # if pi // nc == (nr - 1):
        #     # axes[pi // nc][pi % nc].set_xticks([ni+0.25 for ni in range(len(group_name))])
        #     axes[pi // nc][pi % nc].set_xticklabels(group_name, fontsize=12, rotation=90)
        # else:
        #     axes[pi // nc][pi % nc].get_xaxis().set_visible(False)
    
    print(property_names)
    ax.set_thetagrids(angles * 180/np.pi, property_names)
    ax.tick_params(axis='both', which='major', pad=30)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig("plot/img/radar_plot/{}.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in {}".format(title))

def property_radar_animation(property_keys, all_paths_dict, groups, title):
    # plt.style.use('ggplot')


    #print(load_property_in_step)

    N = len(property_keys)
    theta = radar_factory(N, frame='polygon')
    group_name = list(groups.keys())
    # fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(8, 3*nr))
    
    fig, ax = plt.subplots(figsize=(11, 11), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    property_key = list(property_keys.keys())
    property_names = [property_keys[key] for key in property_key]
    #angles=np.linspace(0,2*np.pi,len(property_names), endpoint=False)
    #print(angles)
    #angles=np.concatenate((angles,[angles[0]]))
    #property_names.append(property_names[0])
    ax.set_rgrids([0, 0.2, 0.4, 0.6, 0.8, 1])
    title = ax.set_title('Episode: ', weight='bold', position=(0.5, 1.1),
                horizontalalignment='center', verticalalignment='center', fontsize=28, pad=7)

    radar_lines = []
    radar_fills = []
    for ni, name in enumerate(group_name):
        line, = ax.plot([], [], 'o--', color=c_contrast[ni], label=name)
        #line.set_data(theta, [0.1, 0.3, 0.5, 0.7, 0.9])
        #print(theta)
        #break
        #return
        radar_lines.append(line)
        fill, = ax.fill([], [], alpha=0.25, color=c_contrast[ni])
        radar_fills.append(fill)
    #plt.show()
    #return
    # # if nr == 1:
    # #     axes = [axes]
    # allprop_means = []
    # allprop_stds = []
    # for ni, name in enumerate(group_name):
    #     allp = []
    #     allprop_mean = []
    #     allprop_std = []
    #     for pi, pk in enumerate(property_key):
    #         same_color = groups[name]
    #         prop1 = []
    #         for rep in all_goals_prop[pk].keys():
    #             if "_".join(rep.split("_")[:-1]) in same_color:
    #                 for run in all_goals_prop[pk][rep].keys():
    #                     prop1.append(all_goals_prop[pk][rep][run])
    #                     # allp.append(all_goals_prop[pk][rep][run])
        
    #         allprop_mean.append(np.mean(prop1))
    #         allprop_std.append(np.std(prop1)/np.sqrt(len(prop1))*1.96)

    #         # axes[pi//nc][pi%nc].scatter(np.random.uniform(ni, ni+0.1, size=len(prop1)), prop1, s=6)
    #         # allprop.append(prop1)
    #     #allprop_mean.append(allprop_mean[0])
    #     allprop_mean, allprop_std = np.array(allprop_mean), np.array(allprop_std)
    #     ax.plot(theta, allprop_mean, 'o--', color=c_contrast[ni], label=name)
    #     ax.fill(theta, allprop_mean, alpha=0.25, color=c_contrast[ni])
    #     allprop_means.append(allprop_mean)
    #     allprop_stds.append(allprop_std)
    #     # ax.fill(theta, allprop_mean, alpha=0.25, color=c_default[ni])
    #     # low = allprop_mean-allprop_std
        
    #     # high = allprop_mean+allprop_std
        
    #     #ax.fill_between(list(theta) + list(theta[:1]), list(low) + list(low)[:1], list(high) + list(high[:1]), alpha=0.25, color=c_default[ni*4])


    #     #axes[pi//nc][pi%nc].boxplot(allprop)

    #     # axes[pi // nc][pi % nc].set_title(property_keys[pk])
    #     # allp = np.array(allp)
    #     # axes[pi // nc][pi % nc].set_yticks([allp.min(), allp.max()])
    #     # axes[pi // nc][pi % nc].set_yticklabels(["{:.2f}".format(allp.min()), "{:.2f}".format(allp.max())])
    #     # if pi // nc == (nr - 1):
    #     #     # axes[pi // nc][pi % nc].set_xticks([ni+0.25 for ni in range(len(group_name))])
    #     #     axes[pi // nc][pi % nc].set_xticklabels(group_name, fontsize=12, rotation=90)
    #     # else:
    #     #     axes[pi // nc][pi % nc].get_xaxis().set_visible(False)
    group_lines = []
    for ni, name in enumerate(group_name): 
        # allprop_mean = allprop_means[ni]
        # allprop_std = allprop_stds[ni]
        lines = []
        for th_idx, th in enumerate(theta): # Generating Error Bars
            # print(th_idx)
            line, = ax.plot([], [], '.-', color=c_contrast[ni], alpha=0.65, linewidth=4, markersize=8)
            lines.append(line)
        group_lines.append(lines) 

    # print(property_names)
    
    ax.set_varlabels(property_names)
    # ax.set_thetagrids(angles * 180/np.pi, property_names)
    ax.tick_params(axis='both', which='major', pad=25)
    # plt.grid(True)
    # plt.tight_layout()
    plt.ylim(0., 1.)
    plt.xticks(fontsize=20)
    plt.legend(bbox_to_anchor=(0.75, 0.95), loc='upper left', fontsize=20)
    # legend = ax.legend(labels, loc=(0.9, .95),
    #                           labelspacing=0.1, fontsize='small')
    ax.patch.set_alpha(0.9) #background color

    # initializing a figure in 
    # which the graph will be plotted
    # fig = plt.figure() 
    
    # # marking the x-axis and y-axis
    # axis = plt.axes(xlim =(0, 4), 
    #                 ylim =(-2, 2)) 
    
    # # initializing a line variable
    # line, = axis.plot([], [], lw = 3) 
    
    # data which the line will 
    # contain (x, y)
    def init():
        for line in radar_lines:
            line.set_data([], [])

        return radar_lines + radar_fills
    
    def animate(i):
        print('frame: ', i)
        title.set_text('Steps: {}'.format(i*10000))
        all_goals_prop = {}
        for pk in property_keys.keys():
            print(i*10000)
            properties, _ = load_property_in_step([all_paths_dict], property_key=pk, early_stopped=True, step=(i+1)*10000)
            all_goals_prop[pk] = properties
            # print(properties)
            # return radar_lines
        allprop_means = []
        allprop_stds = []
        for ni, name in enumerate(group_name):
            allp = []
            allprop_mean = []
            allprop_std = []
            for pi, pk in enumerate(property_key):
                same_color = groups[name]
                prop1 = []
                for rep in all_goals_prop[pk].keys():
                    if "_".join(rep.split("_")[:-1]) in same_color:
                        for run in all_goals_prop[pk][rep].keys():
                            prop1.append(all_goals_prop[pk][rep][run])
                            # allp.append(all_goals_prop[pk][rep][run])
            
                allprop_mean.append(np.mean(prop1))
                allprop_std.append(np.std(prop1)/np.sqrt(len(prop1))*1.96)

                # axes[pi//nc][pi%nc].scatter(np.random.uniform(ni, ni+0.1, size=len(prop1)), prop1, s=6)
                # allprop.append(prop1)
            #allprop_mean.append(allprop_mean[0])
            #allprop_mean, allprop_std = np.array(allprop_mean), np.array(allprop_std)
            #ax.plot(theta, allprop_mean, 'o--', color=c_contrast[ni], label=name)
            radar_lines[ni].set_data(theta, allprop_mean)
            ax._close_line(radar_lines[ni])
            # print(type(theta))
            # print('thet: ', theta)
            # print('whut: ', [theta[0]])
            # print('theta: ', theta + theta[:1])
            #print(np.array([list(theta) + [theta[0]], list(allprop_mean) + [allprop_mean[0]] ]))
            
            radar_fills[ni].set_xy(np.array([list(theta) + [theta[0]], list(allprop_mean) + [allprop_mean[0]] ]).T)
            
            # ax.fill(theta, allprop_mean, alpha=0.25, color=c_contrast[ni])
            allprop_means.append(allprop_mean)
            allprop_stds.append(allprop_std)
            # ax.fill(theta, allprop_mean, alpha=0.25, color=c_default[ni])
            # low = allprop_mean-allprop_std
            
            # high = allprop_mean+allprop_std
            
            #ax.fill_between(list(theta) + list(theta[:1]), list(low) + list(low)[:1], list(high) + list(high[:1]), alpha=0.25, color=c_default[ni*4])


            #axes[pi//nc][pi%nc].boxplot(allprop)

            # axes[pi // nc][pi % nc].set_title(property_keys[pk])
            # allp = np.array(allp)
            # axes[pi // nc][pi % nc].set_yticks([allp.min(), allp.max()])
            # axes[pi // nc][pi % nc].set_yticklabels(["{:.2f}".format(allp.min()), "{:.2f}".format(allp.max())])
            # if pi // nc == (nr - 1):
            #     # axes[pi // nc][pi % nc].set_xticks([ni+0.25 for ni in range(len(group_name))])
            #     axes[pi // nc][pi % nc].set_xticklabels(group_name, fontsize=12, rotation=90)
            # else:
            #     axes[pi // nc][pi % nc].get_xaxis().set_visible(False)
        for ni, name in enumerate(group_name): 
            allprop_mean = allprop_means[ni]
            allprop_std = allprop_stds[ni]

            for th_idx, th in enumerate(theta): # Generating Error Bars
                group_lines[ni][th_idx].set_data([th, th], [allprop_mean[th_idx]-allprop_std[th_idx], allprop_mean[th_idx]+allprop_std[th_idx]])
    
        # plots a sine graph
        # y = np.sin(2 * np.pi * (x - 0.01 * i))
        # line.set_data(x, y)
        
        return radar_lines + radar_fills + group_lines[0] + group_lines[1]
    
    anim = FuncAnimation(fig, animate, init_func = init,
                        frames = 16, interval = 20)
    
    #plt.show()
    anim.save('radar_plot.mp4', 
            writer = 'ffmpeg', fps = 2)

    # plt.savefig("plot/img/radar_plot/{}_poly.pdf".format(title), dpi=300, bbox_inches='tight')
    print("Save in {}".format(title))

def main():
    ranks = np.load("data/dataset/gridhard/srs/goal(9, 9)_simrank.npy", allow_pickle=True).item()
    for i in ranks:
        ranks[i] += 1
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

    targets = [
        "ReLU",
        "ReLU+VirtualVF1", "ReLU+VirtualVF5", "ReLU+XY", "ReLU+Decoder", "ReLU+NAS", "ReLU+Reward", "ReLU+SF", "ReLU+ATC",
        "ReLU(L)",
        "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF", "ReLU(L)+ATC",
        "FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8",
        "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC",

        # New
        # "ReLU+CompOrtho", "ReLU+CR",
        # "ReLU+Laplacian", "ReLU+DynaOrtho",
    ]
    # property_keys.pop("return")
    # for key_pair in itertools.combinations(property_keys.keys(), 2):
    #     pair = {}
    #     pair[key_pair[0]] = property_keys[key_pair[0]]
    #     pair[key_pair[1]] = property_keys[key_pair[1]]
    #     pair_property(pair, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), [106], ranks, xlim=[0, 11], with_auc=False)

    groups = {
        "ReLU": ["ReLU"],
        # "ReLU(L)": ["ReLU(L)", "ReLU(L)+VirtualVF1", "ReLU(L)+VirtualVF5", "ReLU(L)+XY", "ReLU(L)+Decoder", "ReLU(L)+NAS", "ReLU(L)+Reward", "ReLU(L)+SF", "ReLU(L)+ATC"],
        # "FTA": ["FTA eta=0.2", "FTA eta=0.4", "FTA eta=0.6", "FTA eta=0.8", "FTA+VirtualVF1", "FTA+VirtualVF5", "FTA+XY", "FTA+Decoder", "FTA+NAS", "FTA+Reward", "FTA+SF", "FTA+ATC"],# "ReLU+CR", "ReLU+DynaOrtho"],
    }
    # groups = {
    #     "FTA": ["ReLU+CR"],
    # }
    # property_keys.pop("return")
    # property_keys.pop("sparsity")
    # property_keys.pop("interf")
    # property_keys.pop("distance")
    groups = {
        # "FTA": ["FTA eta=0.8"],
        # "FTA+VirtualVF5": ["FTA+VirtualVF5"],
        "No Aux": ["ReLU"],
        # "LAP": ["ReLU+Laplacian"],
        # "LAP(L)": ["ReLU(L)+Laplacian"],
        # "ReLU+CR+O": ["ReLU+CompOrtho"],
        "ReLU+DynaOrtho": ["ReLU+DynaOrtho"],

        # "CompOrtho": ["ReLU+CR"],

    # #    "ReLU+ATC": ["ReLU+ATC"]
    }

    groups = {
        # "FTA": ["FTA eta=0.8"],
        # "FTA+VirtualVF5": ["FTA+VirtualVF5"],
        "No Aux": ["ReLU"],
        # "LAP": ["ReLU+Laplacian"],
        # "LAP(L)": ["ReLU(L)+Laplacian"],
        # "ReLU+CR+O": ["ReLU+CompOrtho"],
        "ReLU+ATC": ["ReLU+ATC"],

        # "CompOrtho": ["ReLU+CR"],

    # #    "ReLU+ATC": ["ReLU+ATC"]
    }
    groups = {
        # "FTA": ["FTA eta=0.8"],
        # "FTA+VirtualVF5": ["FTA+VirtualVF5"],
        "No Aux": ["ReLU"],
        "VF5": ["ReLU+VirtualVF5"],
    #    "ReLU+ATC": ["ReLU+ATC"]
    }

    property_keys.pop("return")
    property_keys.pop("interf")

    # property_scatter(property_keys, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-activation")
    property_radar_animation(property_keys, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "ReLU")
    # property_scatter_radar_polygon(property_keys, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "ReLU")
    # groups = {
    #     "No Aux": ["FTA eta=0.8"],
    #     # "FTA+VirtualVF5": ["FTA+VirtualVF5"],
    #     # "No Aux": ["ReLU"],
    #     "VF5": ["FTA+VirtualVF5"],
    # #    "ReLU+ATC": ["ReLU+ATC"]
    # }
    # property_scatter_radar_polygon(property_keys, label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "FTA")

    # performance_scatter(label_filter(targets, gh_nonlinear_transfer_sweep_v13_largeReLU), groups, "nonlinear/group-auc-activation",
    #                     goal_ids, xlim=[0, 11], ylim=[4, 10])
if __name__ == '__main__':
    main()