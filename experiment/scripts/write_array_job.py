import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir+"/../..")
from experiment.sweeper import Sweeper

save_in_folder = "tasks_{}.sh"

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
                       "\n"
                       )
            count += 1
            if count % line_per_file == 0:
                file.close()
                print(save_in_folder.format(str(prev_file), " done"))
                prev_file += 1
                # if (i+1) * (all_configs.index(config_file)+1) < total_comb * num_run * len(all_configs):
                file = open(save_in_folder.format(str(prev_file)), 'w')
                print("open new file number", prev_file)
    if not file.closed:
        file.close()

def best_learning(all_configs, prev_file=0, line_per_file=1, start_id=0, num_run=30):

    count = 0
    file = open(save_in_folder.format(int(prev_file)), 'w')

    for exp_name, config_file in all_configs:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        sweeper = Sweeper(project_root, config_file)
        assert sweeper.total_combinations == 1
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
                print(save_in_folder.format(str(prev_file), " done"))
                prev_file += 1
                # if (i+1) * (all_configs.index(config_file)+1) < total_comb * num_run * len(all_configs):
                file = open(save_in_folder.format(str(prev_file)), 'w')
                print("open new file number", prev_file)
    if not file.closed:
        file.close()

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
                print(save_in_folder.format(str(prev_file), " done"))
                prev_file += 1
                # if (i+1) * (all_configs.index(config_file)+1) < total_comb * num_run * len(all_configs):
                file = open(save_in_folder.format(str(prev_file)), 'w')
                print("open new file number", prev_file)
    if not file.closed:
        file.close()

if __name__ == '__main__':
    rep_learning([
        ["run_dqn", "config_files/linear_vf/gridhard/representations/dqn_lta/bin8/sweep.json"],
        # ["property_measure", "config_files/linear_vf/gridhard/property/dqn_aux/aux_control/sweep_1g.json"],
        # ["run_dqn", "config_files/linear_vf/collect_two/control/same_task/fix_rep/dqn_aux/aux_control/best.json"],
    ], prev_file=0, line_per_file=15, num_run=5)