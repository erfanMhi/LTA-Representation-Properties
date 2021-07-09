import os,sys,inspect, math
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir+"/../..")
from experiment.sweeper import Sweeper

save_in_folder = "tasks_{}.txt"

def rep_learning(all_configs, prev_file=0, line_per_file=1, num_run=30):

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
                       " --device 0" +
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

def write_script(num_node, start, total_tasks, hours):
    task_per_node = math.ceil((total_tasks-start+1) / float(num_node))
    count = start
    fi = 0
    while count < total_tasks:
        f = open("run_node_{}.sh".format(fi), "w")
        f.writelines(
            ["#!/bin/bash \n",
             "#SBATCH --account=rrg-whitem\n",
             "#SBATCH --mail-type=ALL\n",
             "#SBATCH --mail-user=\n",
             "#SBATCH --nodes=1\n",
             "#SBATCH --ntasks=32\n",
             "#SBATCH --time={}:55:00\n".format(hours-1),
             "#SBATCH --mem-per-cpu=4G\n",
             "#SBATCH --job-name lta{}\n".format(fi),
             "#SBATCH --output=out.txt\n",
             "#SBATCH --error=err.txt\n",

             "chmod +x tasks*\n",
             "cd $SLURM_SUBMIT_DIR/../../\n",
             "chmod +x main\n",
             "export OMP_NUM_THREADS=1\n",
             "source $HOME/<>/bin/activate\n",
             "parallel --jobs 32 --results ./experiment/scripts/outputs"+str(fi)+"/ ./experiment/scripts/tasks_{}.sh ::: $(seq "+str(int(count))+" "+str(int(min(count+task_per_node-1, total_tasks)))+") &\n",
             "sleep {}h\n".format(hours)])
        f.close()
        count += task_per_node
        fi += 1
    return


if __name__ == '__main__':

    # use "run_dqn" when you are learning representation for dqn or dqn_lta, or running transfer learning with any representation.
    # use "run_dqn_aux" when you are learning representation for any representation with auxiliary tasks.
    # rep_learning([
        # ["run_dqn", "config/test/picky_eater/online_property/dqn_lta/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/online_property/dqn/sweep.json"],
        # # ["property_measure", "config_files/linear_vf/gridhard_xy/property/dqn_aux/aux_control/sweep_1g.json"],
    # ], prev_file=0, line_per_file=1, num_run=30)
    
    # rep_learning([
        # ["run_dqn", "config/test/picky_eater/online_property/dqn/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/online_property/dqn/best.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fine_tune/dqn_lta/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fine_tune/dqn_lta/best_early.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fix_rep/dqn_lta/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fix_rep/dqn_lta/best_early.json"],
        # ["run_dqn", "config/test/picky_eater/control/same_task/fix_rep/dqn_lta/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/same_task/fix_rep/dqn_lta/best_early.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/reward/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/successor_as/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/aux_control/initial.json"],
    # ], prev_file=0, line_per_file=1, num_run=30)
    

    #rep_learning([
        # ["run_dqn", "config/test/picky_eater/online_property/dqn_lta/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/aux_control/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/info/sweep_xy.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/info/sweep_xy.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/input_decoder/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/input_decoder/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/nas_v2_delta/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/reward/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/reward/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/successor_as/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/successor_as/sweep.json"],
    # ], prev_file=0, line_per_file=1, num_run=30)

    # rep_learning([
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/info/sweep_xy.json"],
        # ["run_dqn", "config/test/picky_eater/control/last/different_task/fine_tune/dqn/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/last/different_task/fix_rep/dqn/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/last/same_task/fix_rep/dqn/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/last/different_task/fine_tune/dqn_lta/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_lta/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/last/different_task/fix_rep/dqn_lta/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_lta/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/last/same_task/fix_rep/dqn_lta/sweep.json"],
        # ["run_dqn", "config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_lta/sweep.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/aux_control/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/aux_control/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/info/best_xy.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/input_decoder/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/input_decoder/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/nas_v2_delta/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/reward/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/reward/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/successor_as/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/successor_as/best.json"],
    # ], prev_file=0, line_per_file=1, num_run=30)

#    rep_learning(
            # [['run_dqn',
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/reward/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/sweep.json'],
            
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/reward/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/nas_v2_delta/sweep.json']
            # ,
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/input_decoder/sweep.json'
            # ],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/reward/sweep.json'],
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/sweep.json'],
            
             # ['run_dqn',
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/successor_as/sweep.json']
            # ], prev_file=0, line_per_file=1, num_run=30)

    # rep_learning([
            # ["run_dqn", "config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/info/sweep_xy.json"],
            # ["run_dqn", "config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/info/sweep_xy.json"],
            # ["run_dqn", "config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/info/sweep_xy.json"],
            # ], prev_file=0, line_per_file=1, num_run=30)

#          rep_learning( 
            # config/test/picky_eater/online_property/dqn_aux/aux_control/ 
            # [['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/nas_v2_delta/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/input_decoder/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/reward/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/aux_control/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/same_task/fix_rep/dqn_aux/successor_as/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/nas_v2_delta/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/input_decoder/sweep.json'], 
             
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/reward/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/aux_control/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fix_rep/dqn_aux/successor_as/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/nas_v2_delta/sweep.json'] 
            # , 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/input_decoder/sweep.json' 
            # ], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/reward/sweep.json'], 
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/aux_control/sweep.json'], 
             
             # ['run_dqn', 
              # 'config/test/picky_eater/control/early_stop/different_task/fine_tune/dqn_aux/successor_as/sweep.json'] 
            # ], prev_file=0, line_per_file=1, num_run=30) 

       

    # rep_learning([
        # ["run_dqn", "config/test/picky_eater/control/different_task/fine_tune/dqn/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fine_tune/dqn/best_early.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fix_rep/dqn/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fix_rep/dqn/best_early.json"],
        # ["run_dqn", "config/test/picky_eater/control/same_task/fix_rep/dqn/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/same_task/fix_rep/dqn/best_early.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/aux_control/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/info/initial_xy.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/input_decoder/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/nas_v2_delta/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/reward/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/successor_as/initial.json"],
    # ], prev_file=0, line_per_file=1, num_run=30)


#     rep_learning([
        # ["run_dqn", "config/test/picky_eater/online_property/dqn/sweep_1f.json"],
        # ["run_dqn", "config/test/picky_eater/online_property/dqn/sweep_4f.json"],
        # ["run_dqn", "config/test/picky_eater/online_property/dqn_lta/sweep_1f.json"],
        # ["run_dqn", "config/test/picky_eater/online_property/dqn_lta/sweep_4f.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/aux_control/sweep_1f.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/aux_control/sweep_4f.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep_1f.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/aux_control/sweep_4f.json"],
    # ], prev_file=0, line_per_file=1, num_run=30)

#    rep_learning([
#        ["run_dqn", "config/test/picky_eater/control_test/decayep/last/different_task/fix_rep/dqn_lta_aux/aux_control/sweep_2f.json"],
#    ], prev_file=0, line_per_file=1, num_run=30)
    
    #rep_learning([
    #["run_dqn", "config/test/picky_eater_color_switch/control/last/different_task/fix_rep/dqn_aux/aux_control/sweep_3f.json"],
    #["run_dqn", "config/test/picky_eater_color_switch/control/last/different_task/fix_rep/dqn/best_3f.json"],
    #["run_dqn", "config/test/picky_eater_color_switch/representation/dqn/best_3f.json"],
    #["run_dqn_aux", "config/test/picky_eater_color_switch/representation/dqn_aux/aux_control/best_3f.json"]
#    ["run_dqn_aux", "config/test/picky_eater_color_switch/representation/dqn_lta_aux/aux_control/sweep_3f.json"]
    #], prev_file=0, line_per_file=2, num_run=30)

    # rep_learning([
        # ["run_dqn", "config/test/picky_eater/control/different_task/fine_tune/dqn/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fine_tune/dqn/best_early.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fix_rep/dqn/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/different_task/fix_rep/dqn/best_early.json"],
        # ["run_dqn", "config/test/picky_eater/control/same_task/fix_rep/dqn/best_final.json"],
        # ["run_dqn", "config/test/picky_eater/control/same_task/fix_rep/dqn/best_early.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/aux_control/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/info/initial_xy.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/input_decoder/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/nas_v2_delta/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/reward/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/successor_as/initial.json"],
    # ], prev_file=0, line_per_file=1, num_run=30)

    #rep_learning([
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/aux_control/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/info/initial_xy.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/input_decoder/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/nas_v2_delta/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/reward/initial.json"],
    #    ["run_dqn_aux", "config/test_v7/picky_eater/representation/dqn_lta_aux/aux_control/sweep_3f.json"],
    #    ["run_dqn_aux", "config/test_v7/picky_eater/representation/dqn_aux/aux_control/sweep_3f.json"],
    #], prev_file=0, line_per_file=1, num_run=10)

    rep_learning([
         ["run_dqn_aux", "config/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/nas_v2_delta/sweep_3f.json"],
         ["run_dqn_aux", "config/result/picky_eater/nonlinear_vf/representation/dqn_lta_aux/successor_as/sweep_3f.json"],
    ], prev_file=0, line_per_file=2, num_run=5)


    # rep_learning([
        # ["run_dqn", "config/test/picky_eater/online_property/dqn_lta/best.json"],
        # ["run_dqn", "config/test/picky_eater/online_property/dqn/best.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/aux_control/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/aux_control/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/info/initial_xy.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/info/initial_xy.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/input_decoder/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/input_decoder/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/nas_v2_delta/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/nas_v2_delta/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/reward/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/reward/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_aux/successor_as/initial.json"],
        # ["run_dqn_aux", "config/test/picky_eater/online_property/dqn_lta_aux/successor_as/initial.json"],
    # ], prev_file=0, line_per_file=1, num_run=30)


    # This function is for writing script asking for node. Please fill in your email account after --mail-user= ,
    # feel free to modify the content after --job-name .
    # And please fill in the name of your virtual env in "source $HOME/<>/bin/activate\n", if you're using any
    # The script for array jobs is in run.sh
    # write_script(1, 0, 180, 4)
