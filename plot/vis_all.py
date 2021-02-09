import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def all_visualization(root, run_num, setting_list=None):
    run_path = os.path.join(root, "{}_run".format(0))
    for param in os.listdir(run_path):
        setting = int(param.split("_")[0])
        if setting_list is None:
            check = True
        else:
            check = True if setting in setting_list else False

        if check:
            fig, ax = plt.subplots(1, run_num)
            for r in range(run_num):
                run_path = os.path.join(root, "{}_run".format(r))
                vis = os.path.join(run_path, "{}/visualizations/visualization.png".format(param))
                if os.path.isfile(vis):
                    img = plt.imread(vis)
                    ax[r].imshow(img)
            plt.title(param)
            plt.show()

# all_visualization("../data/output/tests/collect_two/representations/laplace_aux/info/sweep_xy+rwd", 3, setting_list=[5,11,3,6,0])
all_visualization("../data/output/linear_vf/gridhard/representations/laplace/sweep", 3, setting_list=[13,33,34,0,26,29])
