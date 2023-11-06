import os,sys,inspect, math
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir+"/../..")
from experiment.sweeper import Sweeper

save_in_folder = "tasks_{}.sh"

def rep_learning(all_configs, prev_file=0, line_per_file=1, num_run=30, device=0):

    count = 0
    file = open(save_in_folder.format(int(prev_file)), 'w')

    for exp_name, config_file in all_configs:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        sweeper = Sweeper(project_root, config_file)
        total_comb = sweeper.total_combinations * num_run
        for i in range(total_comb):
            file.write("python "+ exp_name +".py" +
                       " --id " + str(i) +
                       " --config-file experiment/" + str(config_file) +
                       " --device {}".format(device) +
                       "\n"
                       )
            count += 1
            if count % line_per_file == 0:
                file.close()
                # print(save_in_folder.format(str(prev_file), " done"))
                prev_file += 1
                # if (i+1) * (all_configs.index(config_file)+1) < total_comb * num_run * len(all_configs):
                file = open(save_in_folder.format(str(prev_file)), 'w')
                # print("open new file number", prev_file)
    if not file.closed:
        file.close()
    print("last script:", save_in_folder.format(str(prev_file)))

def best_learning(all_configs, prev_file=0, line_per_file=1, start_id=0, num_run=30):

    count = 0
    file = open(save_in_folder.format(int(prev_file)), 'w')

    for exp_name, config_file in all_configs:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        sweeper = Sweeper(project_root, config_file)

        assert sweeper.total_combinations == 1, print("total combination should be 1. Now there are", sweeper.total_combinations)
        total_comb = sweeper.total_combinations * num_run
        for i in range(start_id, total_comb):
            file.write("python "+ exp_name +".py" +
                       " --id " + str(i) +
                       " --config-file experiment/" + str(config_file) +
                       "\n"
                       )
            count += 1
            if count % line_per_file == 0:
                file.close()
                # print(save_in_folder.format(str(prev_file), " done"))
                prev_file += 1
                # if (i+1) * (all_configs.index(config_file)+1) < total_comb * num_run * len(all_configs):
                file = open(save_in_folder.format(str(prev_file)), 'w')
                # print("open new file number", prev_file)
    if not file.closed:
        file.close()
    print("last script:", save_in_folder.format(str(prev_file)))

def lp_test(all_configs, prev_file=1000, line_per_file=1):

    num_run = 60
    # count_start = count
    count = 0
    file = open(save_in_folder.format(int(prev_file)), 'w')

    for exp, agent, config_file in all_configs:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        sweeper = Sweeper(project_root, config_file)
        total_comb = sweeper.total_combinations * num_run
        for i in range(0, total_comb):
            file.write("python test_"+ agent +"_"+ exp +".py" +
                       " --id " + str(i) +
                       " --config-file experiment/" + str(config_file) +
                       "\n"
                       )
            count += 1
            if count % line_per_file == 0:
                file.close()
                # print(save_in_folder.format(str(prev_file), " done"))
                prev_file += 1
                # if (i+1) * (all_configs.index(config_file)+1) < total_comb * num_run * len(all_configs):
                file = open(save_in_folder.format(str(prev_file)), 'w')
                # print("open new file number", prev_file)
    if not file.closed:
        file.close()
    print("last script:", save_in_folder.format(str(prev_file)))

def separate_task(task_lst):
    all = []
    for chunk in task_lst:
        if len(chunk) == 1:
            all.append(chunk[0])
        elif len(chunk) == 2:
            all += list(range(chunk[0], chunk[1]+1))
        else:
            raise IOError
    return all

def write_script(start_script, num_script, start_task=None, total_tasks=None, tasks_list=None, hours=1,
                    min_node=1, parallel=30, account="rrg-whitem", virt_env="torch1env"):
    if start_task is not None and total_tasks is not None:
        CONDITION = 1
        assert tasks_list is None
        task_per_script = math.ceil((total_tasks-start_task+1) / float(num_script))
    elif tasks_list is not None:
        CONDITION = 2
        assert start_task is None and total_tasks is None
        separated = separate_task(tasks_list)
        start_task = 0
        total_tasks = len(separated)
        print("{} tasks in total".format(total_tasks))
        task_per_script = math.ceil(total_tasks / float(num_script))
    else:
        raise IOError

    count = start_task
    fi = start_script
    while count < total_tasks:
        if CONDITION == 1:
            node_jobs = "$(seq " + str(int(count)) + " " + str(int(min(count + task_per_script - 1, total_tasks))) + ")"
        else:
            node_jobs = " ".join(map(str, separated[count: count + task_per_script]))

        f = open("run_node_{}.sh".format(fi), "w")
        f.writelines(
            ["#!/bin/bash \n",
             "#SBATCH --account={}\n".format(account),
             "#SBATCH --mail-type=ALL\n",
             "#SBATCH --mail-user=han8@ualberta.ca\n",
             "#SBATCH --nodes={}\n".format(min_node),
             "#SBATCH --ntasks-per-node={}\n".format(parallel),
             "#SBATCH --cpus-per-task=1\n",
             "#SBATCH --time={}:55:00\n".format(hours-1),
             "#SBATCH --mem-per-cpu=4000M\n",
             # "#SBATCH --mem={}M\n".format(4000*parallel),
             "#SBATCH --job-name lta{}\n".format(fi),
             "#SBATCH --output=out.txt\n",
             "#SBATCH --error=err.txt\n",

             # "chmod +x tasks*\n",
             "cd $SLURM_SUBMIT_DIR/../../\n",
             "export OMP_NUM_THREADS=1\n",
             "source $HOME/{}/bin/activate\n".format(virt_env),
             "parallel --jobs "+str(parallel)+" --results ./experiment/scripts/outputs"+str(fi)+"/ ./experiment/scripts/tasks_{}.sh ::: " + node_jobs + " &\n",
             "sleep {}h\n".format(hours)])
        f.close()
        count += task_per_script
        fi += 1
    return


if __name__ == '__main__':

    # rep_learning([
    #     # ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/fixrep_property/input/best.json"],
    #     # ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/fixrep_property/random/best.json"],
    #     # ["run_dqn", "config/test_v13/gridhard/linear_vf/fixrep_property/input/best.json"],
    #     # ["run_dqn", "config/test_v13/gridhard/linear_vf/fixrep_property/random/best.json"],
    #
    #     ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/best_1g.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/aux_control/best_5g.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/info/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/reward/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/input_decoder/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/nas_v2_delta/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_aux/successor_as/best.json"],
    #     ["run_dqn_ul", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_cl/best.json"],
    #
    #     ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.2_best.json"],
    #     ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.4_best.json"],
    #     ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.6_best.json"],
    #     ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta/eta_study_0.8_best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/best_1g.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/aux_control/best_5g.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/info/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/reward/best.json"],
    #     # ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/best.json"],
    #     # ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/best.json"],
    #     # ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/successor_as/best.json"],
    #     # ["run_dqn_ul", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_cl/best.json"],
    #
    #     ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/aux_control/best_1g.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/aux_control/best_5g.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/info/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/reward/best.json"],
    #     # ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/input_decoder/best.json"],
    #     # ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/nas_v2_delta/best.json"],
    #     # ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/successor_as/best.json"],
    #     # ["run_dqn_ul", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_cl/best.json"],
    #
    # ], prev_file=0, line_per_file=1, num_run=5, device=0) # 1 per 8 hours
    #
    # rep_learning([
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/input_decoder/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/nas_v2_delta/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_aux/successor_as/best.json"],
    #     ["run_dqn_ul", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_lta_cl/best.json"],
    #
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/input_decoder/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/nas_v2_delta/best.json"],
    #     ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_aux/successor_as/best.json"],
    #     ["run_dqn_ul", "config/test_v13/gridhard/nonlinear_vf/original_0909/online_property/dqn_large_cl/best.json"],
    # ], prev_file=110, line_per_file=3, num_run=5, device=1)

    rep_learning([
        ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/aux_control/sweep_1g.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/aux_control/sweep_5g.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/info/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/reward/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/input_decoder/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/nas_v2_delta/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_aux/successor_as/sweep.json"],
        ["run_dqn_ul", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_cl/sweep.json"],

        ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.2_sweep.json"],
        ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.4_sweep.json"],
        ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.6_sweep.json"],
        ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta/eta_study_0.8_sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/aux_control/sweep_1g.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/aux_control/sweep_5g.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/info/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/reward/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/input_decoder/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/nas_v2_delta/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_aux/successor_as/sweep.json"],
        ["run_dqn_ul", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_lta_cl/sweep.json"],

        ["run_dqn", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/aux_control/sweep_1g.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/aux_control/sweep_5g.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/info/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/reward/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/input_decoder/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/nas_v2_delta/sweep.json"],
        ["run_dqn_aux", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_aux/successor_as/sweep.json"],
        ["run_dqn_ul", "config/test_v13/gridhard/nonlinear_vf/original_0909/fixrep_property/dqn_large_cl/sweep.json"],

    ], prev_file=0, line_per_file=1, num_run=5, device=0)
