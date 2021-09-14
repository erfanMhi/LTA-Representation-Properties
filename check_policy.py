import copy
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import core.environment.env_factory as environment
import core.network.net_factory as network
import core.network.activations as activations
import core.component.replay as replay
import core.utils.normalizer as normalizer
import core.component.representation as representation
from core.agent import trained_dqn
from core.utils import torch_utils, schedule, logger, run_funcs, test_funcs
from experiment.sweeper import Sweeper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies param_setting number and parameter configuration')
    parser.add_argument('--config-file', default='experiment/config/example/gridhard/representations/dqn/best.json')
    parser.add_argument('--device', default=-1, type=int, )
    args = parser.parse_args()

    torch_utils.set_one_thread()
    # torch_utils.random_seed(args.id)

    project_root = os.path.abspath(os.path.dirname(__file__))
    cfg = Sweeper(project_root, args.config_file).parse(args.id)
    cfg.device = torch_utils.select_device(args.device)
    torch_utils.random_seed(cfg.seed)

    # if cfg.train_test_split:
    #     cfg.task_data_path = cfg.task_data_path.format(cfg.run)
    #     path = (cfg.task_data_path)
    #     if not os.path.isfile(path):
    #         print("Data file {} doesn't exist. {}".format(cfg.run, path))
    #         exit(1)

    assert cfg.rep_config["load_params"]
    run_num = int(args.id / cfg.cumulative)
    cfg.rep_config["path"] = cfg.rep_config["path"].format(run_num)
    path = os.path.join(cfg.data_root, cfg.rep_config["path"])
    if not os.path.isfile(path):
        print("Run {} doesn't exist. {}".format(run_num, path))
        exit(1)
 
    assert "load_params" in cfg.val_fn_config.keys()
    assert cfg.val_fn_config["load_params"]
    run_num = int(args.id / cfg.cumulative)
    cfg.val_fn_config["path"] = cfg.val_fn_config["path"].format(run_num)
    path = os.path.join(cfg.data_root, cfg.val_fn_config["path"])
    if not os.path.isfile(path):
        print("Run {} doesn't exist. {}".format(run_num, path))
        exit(1)

    cfg.rep_activation_fn = activations.ActvFactory.get_activation_fn(cfg)
    cfg.rep_fn = representation.RepFactory.get_rep_fn(cfg)
    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)
    cfg.val_fn = network.NetFactory.get_val_fn(cfg)
    cfg.replay_fn = replay.ReplayFactory.get_replay_fn(cfg)
    cfg.eps_schedule = schedule.ScheduleFactory.get_eps_schedule(cfg)
    cfg.state_normalizer, cfg.reward_normalizer = normalizer.NormalizerFactory.get_normalizer(cfg)

    # Setting up the logger
    cfg.logger = logger.Logger(cfg)
    cfg.log_config()

    # Initializing the agent and running the experiment
    blue = np.array([0, 0, 255.])
    red = np.array([255., 0, 0])
    green = np.array([0, 255., 0])
    if cfg.check == "visit":
        agent = trained_dqn.TrainedDQNAgent(cfg)
        run_funcs.data_collection_steps(agent)
        all_traj = np.array(agent.all_trajectory)
        agent_pos = np.where(np.all(all_traj == blue, axis=3) == True)

        visit_count = np.zeros((15, 15))

        for i in range(len(all_traj)):
            x, y = agent_pos[1][i], agent_pos[2][i]
            visit_count[x, y] += 1
        plt.figure()
        plt.imshow(visit_count)
        plt.show()
        plt.close()
        plt.clf()
    elif cfg.check == "value":
        cfg.total_samples = 1
        agent = trained_dqn.TrainedDQNAgent(cfg)
        run_funcs.data_collection_steps(agent)
        sample = np.array(agent.all_trajectory)[0]
        agent_pos = np.where(np.all(sample == blue, axis=2) == True)
        sample[agent_pos] = green
        open_space = np.where(np.all(sample == green, axis=2) == True)
        poss = []
        states = []
        for i in range(len(open_space[0])):
            p = [open_space[0][i], open_space[1][i]]
            new_s = copy.deepcopy(sample)
            new_s[p[0], p[1], :] = blue
            poss.append(p)
            states.append(new_s)
        # # Test
        # for i,p in enumerate(poss):
        #     print(p)
        #     plt.imshow(states[i])
        #     plt.show()

        value_map = np.zeros((15, 15))
        for i, p in enumerate(poss):
            max_q = agent.get_q_values(states[i]).max()
            value_map[p[0], p[1]] = max_q
        plt.figure()
        plt.imshow(value_map, interpolation='nearest', cmap="Blues")
        for k in range(value_map.shape[0]):
            for j in range(value_map.shape[1]):
                if value_map[k, j] != 0 and not (k==9 and j==9):
                    plt.text(j, k, "{:1.3f}".format(value_map[k, j]),
                             ha="center", va="center", color="orange")
        plt.show()
        plt.close()
        plt.clf()

    else:
        raise NotImplementedError
