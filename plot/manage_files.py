import os
import shutil

def walk_through():
    from distutils.dir_util import copy_tree
    root = "../data/output/test/gridhard/control/last/different_task/fine_tune/dqn_aux/aux_control/sweep_5g/"
    assert os.path.isdir(root)
    for path, subdirs, files in os.walk(root):

        # for f in files:
        #     if f in ["interference.txt"]:
        #         file1 = os.path.join(path, f)
        #         file2 = os.path.join(path, "noninterference.txt")
        #         # print(file1, file2)
        #         os.rename(file1, file2)

        for name in subdirs:
            if "2_param_setting" in name:
                # set = int(name.split("_param_setting")[0])
                # set += 5
                file1 = os.path.join(path, name)
                print(file1)
                file2 = root+"/../best_5g/"+file1.split("/")[-2] + "/{}_param_setting".format(0)
                # file2 = root+"/../sweep/"+file1.split("/")[-2] + "/{}_param_setting".format(set)
                print(file2, "\n")
                # shutil.rmtree(file1)

                # shutil.copy(file1+"/linear_probing_count.txt", file2+"/linear_probing_count.txt")
                # shutil.copy(file1+"/parameters/linear_probing_count", file2+"/parameters/linear_probing_count")

                if not os.path.isdir(file2):
                    os.makedirs(file2)
                copy_tree(file1, file2)

            # if name == "0_run":
            #     file1 = os.path.join(path, name)
            #     if "best" in file1:
            #         print(file1)
            #         print(os.system("cat {}/0_param_setting/log | grep learning_rate".format(file1)))

def check_log():
    files = [
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_aux/aux_control/sweep_1g/0_run/2_param_setting/log",
        # "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_aux/aux_control/sweep_5g/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_aux/info/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_aux/input_decoder/sweep/0_run/4_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_aux/nas_v2_delta/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_aux/reward/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_aux/successor_as/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.2_sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.4_sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.6_sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.8_sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta_aux/info/sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta_aux/input_decoder/sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta_aux/reward/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/same_task/fix_rep/dqn_lta_aux/successor_as/sweep/0_run/2_param_setting/log",

        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_aux/aux_control/sweep_1g/0_run/1_param_setting/log",
        # "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_aux/aux_control/sweep_5g/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_aux/info/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_aux/input_decoder/sweep/0_run/4_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_aux/nas_v2_delta/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_aux/reward/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_aux/successor_as/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.2_sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.4_sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.6_sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.8_sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta_aux/info/sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta_aux/input_decoder/sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta_aux/reward/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/similar_task/fix_rep/dqn_lta_aux/successor_as/sweep/0_run/2_param_setting/log",

        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn/sweep/0_run/0_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/aux_control/sweep_1g/0_run/1_param_setting/log",
        # "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/aux_control/sweep_5g/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/info/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/input_decoder/sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep/0_run/0_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/reward/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_aux/successor_as/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.2_sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.4_sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.6_sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.8_sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_1g/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_5g/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/info/sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/input_decoder/sweep/0_run/3_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/reward/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fix_rep/dqn_lta_aux/successor_as/sweep/0_run/2_param_setting/log",

        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_aux/aux_control/sweep_1g/0_run/1_param_setting/log",
        # "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_aux/aux_control/sweep_5g/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_aux/info/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_aux/input_decoder/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_aux/nas_v2_delta/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_aux/reward/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_aux/successor_as/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.2_sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.4_sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.6_sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.8_sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/sweep_1g/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/sweep_5g/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta_aux/info/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta_aux/input_decoder/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/sweep/0_run/1_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta_aux/reward/sweep/0_run/2_param_setting/log",
        "../data/output/test_v3/gridhard/control/different_task/fine_tune/dqn_lta_aux/successor_as/sweep/0_run/1_param_setting/log",
    ]

    for f in files:
        print(f)
        os.system("cat {} | grep learning_rate".format(f))
        print()

def check_json():
    files = [
        # "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn/best.json",
        # "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_aux/aux_control/best_1g.json",
        # # "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_aux/aux_control/best_5g.json",
        # "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_aux/info/best.json",
        # "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_aux/input_decoder/best.json",
        # "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_aux/nas_v2_delta/best.json",
        # "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_aux/reward/best.json",
        # "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_aux/successor_as/best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.2_best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.4_best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.6_best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta/eta_study_0.8_best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/aux_control/best_1g.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/aux_control/best_5g.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/info/best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/input_decoder/best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/nas_v2_delta/best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/reward/best.json",
        "../experiment/config/test/gridhard/control/same_task/fix_rep/dqn_lta_aux/successor_as/best.json",

        # "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn/best.json",
        # "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_aux/aux_control/best_1g.json",
        # # "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_aux/aux_control/best_5g.json",
        # "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_aux/info/best.json",
        # "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_aux/input_decoder/best.json",
        # "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_aux/nas_v2_delta/best.json",
        # "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_aux/reward/best.json",
        # "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_aux/successor_as/best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.2_best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.4_best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.6_best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta/eta_study_0.8_best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/aux_control/best_1g.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/aux_control/best_5g.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/info/best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/input_decoder/best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/nas_v2_delta/best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/reward/best.json",
        "../experiment/config/test/gridhard/control/similar_task/fix_rep/dqn_lta_aux/successor_as/best.json",

        # "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_aux/aux_control/best_1g.json",
        # # "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_aux/aux_control/best_5g.json",
        # "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_aux/info/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_aux/input_decoder/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_aux/nas_v2_delta/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_aux/reward/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_aux/successor_as/best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.2_best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.4_best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.6_best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta/eta_study_0.8_best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/aux_control/best_1g.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/aux_control/best_5g.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/info/best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/input_decoder/best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/nas_v2_delta/best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/reward/best.json",
        "../experiment/config/test/gridhard/control/different_task/fix_rep/dqn_lta_aux/successor_as/best.json",

        # "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_aux/aux_control/best_1g.json",
        # # "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_aux/aux_control/best_5g.json",
        # "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_aux/info/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_aux/input_decoder/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_aux/nas_v2_delta/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_aux/reward/best.json",
        # "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_aux/successor_as/best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.2_best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.4_best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.6_best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta/eta_study_0.8_best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/best_1g.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/aux_control/best_5g.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/info/best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/input_decoder/best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/nas_v2_delta/best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/reward/best.json",
        "../experiment/config/test/gridhard/control/different_task/fine_tune/dqn_lta_aux/successor_as/best.json",
    ]

    for f in files:
        print(f)
        # os.system("cat {} | grep learning_rate".format(f))
        # os.system("cat {} | grep gridhard/online_property/dqn".format(f))
        os.system("cat {} | grep eta".format(f))
        print()

walk_through()
# check_log()
# check_json()