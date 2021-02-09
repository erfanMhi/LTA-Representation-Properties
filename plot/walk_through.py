import os
import shutil

def walk_through():
    from distutils.dir_util import copy_tree
    root = "../data/output/paper_v2/collect_two/property/"
    assert os.path.isdir(root)
    for path, subdirs, files in os.walk(root):

        # for f in files:
        #     if f in ["linear_probing_xy.txt"]:
        #         file1 = os.path.join(path, f)
        #         file2 = os.path.join(path, "linear_probing_color.txt")
        #         # os.rename(file1, file2)

        for name in subdirs:
            # if "0_param_setting" in name:
                # set = int(name.split("_param_setting")[0])
                # set += 2
                # file1 = os.path.join(path, name)
                # print(file1)
                # file2 = root+"/../best/"+file1.split("/")[-2] + "/{}_param_setting".format(0)
                # print(file2, "\n")

                # shutil.copy(file1+"/linear_probing_count.txt", file2+"/linear_probing_count.txt")
                # shutil.copy(file1+"/parameters/linear_probing_count", file2+"/parameters/linear_probing_count")

                # if not os.path.isdir(file2):
                #     os.makedirs(file2)
                # copy_tree(file1, file2)

            if name == "0_run":
                file1 = os.path.join(path, name)
                if "best" in file1:
                    print(file1)
                    print(os.system("cat {}/0_param_setting/log | grep learning_rate".format(file1)))


walk_through()
