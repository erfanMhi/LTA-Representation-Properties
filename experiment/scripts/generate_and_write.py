import os,sys,inspect
import json
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir+"/../..")
from experiment.sweeper import Sweeper
from write_array_job import rep_learning, write_script


def generate_conf(parents, key):
    conf_list = []
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    for exp_name, config_file, new_conf_file in parents:
        # Read config.ini file
        # config_object = ConfigParser()
        # config_object.read()
        with open(os.path.join(project_root, config_file)) as f:
            config_dict = json.load(f)

        children = config_dict["generate"][key]
        new_exp_name = config_dict["generate"]["{}_exp_name".format(key)]

        for child in children:
            new_conf_folder_c = "../"+"/".join(new_conf_file.format(child).split("/")[:-1])
            new_conf_file_c = "{}/{}".format(new_conf_folder_c, new_conf_file.format(child).split("/")[-1])
            if not os.path.isdir(new_conf_folder_c):
                os.makedirs(new_conf_folder_c)

            new_exp_name_c = new_exp_name.format(child)
            # new_exp_name_c = os.path.join(project_root, new_exp_name_c)

            config_dict["sweep_parameters"][key] = [child]
            config_dict["fixed_parameters"]["exp_name"] = new_exp_name_c
            config_dict.pop('generate', None)

            with open(new_conf_file_c, 'w') as conf:
                json.dump(config_dict, conf, indent=4)
            conf_list.append([exp_name, new_conf_file_c.strip("../")])
    return conf_list

def generate_and_write(parent_configs, gener_key, prev_file=0, line_per_file=1, num_run=30, device=0):
    conf_list = generate_conf(parent_configs, gener_key)
    rep_learning(conf_list, prev_file=prev_file, line_per_file=line_per_file, num_run=num_run, device=device)
    return

if __name__ == '__main__':
    generate_and_write([
        ["run_dqn", "config/test_v13/gridhard/linear_vf/learning_scratch/dqn/sweep.json", "config/test_v13/gridhard/linear_vf/learning_scratch_generated/goal_id_{}/dqn/sweep.json"],
        # # ["run_dqn", "config/test_v13/gridhard/linear_vf/learning_scratch/dqn/fix_eps_sweep.json", "config/test_v13/gridhard/linear_vf/learning_scratch_generated/goal_id_{}/dqn/fix_eps_sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/learning_scratch/random/sweep.json", "config/test_v13/gridhard/linear_vf/learning_scratch_generated/goal_id_{}/random/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/learning_scratch/input/sweep.json", "config/test_v13/gridhard/linear_vf/learning_scratch_generated/goal_id_{}/input/sweep.json"],

        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_aux/aux_control/sweep_1g.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_aux/aux_control/sweep_1g.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_aux/aux_control/sweep_5g.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_aux/aux_control/sweep_5g.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_aux/info/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_aux/info/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_aux/input_decoder/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_aux/input_decoder/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_aux/nas_v2_delta/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_aux/nas_v2_delta/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_aux/reward/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_aux/reward/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_aux/successor_as/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_aux/successor_as/sweep.json"],
        #
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta/eta_study_0.2_sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta/eta_study_0.2_sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta/eta_study_0.4_sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta/eta_study_0.4_sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta/eta_study_0.6_sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta/eta_study_0.6_sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta/eta_study_0.8_sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta/eta_study_0.8_sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta_aux/aux_control/sweep_1g.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta_aux/aux_control/sweep_1g.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta_aux/aux_control/sweep_5g.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta_aux/aux_control/sweep_5g.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta_aux/info/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta_aux/info/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta_aux/input_decoder/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta_aux/input_decoder/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta_aux/nas_v2_delta/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta_aux/nas_v2_delta/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta_aux/reward/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta_aux/reward/sweep.json"],
        # ["run_dqn", "config/test_v13/gridhard/linear_vf/original_0909/transfer/dqn_lta_aux/successor_as/sweep.json", "config/test_v13/gridhard/linear_vf/original_0909/transfer_generated/goal_id_{}/dqn_lta_aux/successor_as/sweep.json"],
    ], "goal_id", prev_file=41325, line_per_file=1, num_run=5, device=0) # 5 per hour, 2.6 per hour

