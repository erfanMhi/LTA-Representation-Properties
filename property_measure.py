import os
import argparse
from importlib import import_module

import core.environment.env_factory as environment
import core.network.net_factory as network
import core.network.optimizer as optimizer
import core.component.replay as replay
import core.component.representation as representation
import core.component.linear_probing_tasks as linear_probing_tasks
import core.utils.normalizer as normalizer
from core.agent import linear_probing, dqn, laplace
from core.utils import torch_utils, schedule, logger, test_funcs
from experiment.sweeper import Sweeper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies run number and parameter configuration')
    parser.add_argument('--config-file', default='experiment/config_files/dqn_test.json')
    parser.add_argument('--device', default=-1, type=int, )
    args = parser.parse_args()

    torch_utils.set_one_thread()
    torch_utils.random_seed(args.id)

    project_root = os.path.abspath(os.path.dirname(__file__))
    cfg = Sweeper(project_root, args.config_file).parse(args.id)
    cfg.device = torch_utils.select_device(args.device)

    if cfg.rep_config["load_params"]:
        run_num = int(args.id / cfg.cumulative)
        cfg.rep_config["path"] = cfg.rep_config["path"].format(run_num)
        path = os.path.join(cfg.data_root, cfg.rep_config["path"])
        if not os.path.isfile(path):
            print("Run {} doesn't exist. {}".format(args.id, path))
            exit(1)

    cfg.rep_fn = representation.RepFactory.get_rep_fn(cfg)
    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)

    cfg.val_fn = network.NetFactory.get_val_fn(cfg)
    cfg.optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)

    cfg.replay_fn = replay.ReplayFactory.get_replay_fn(cfg)
    cfg.state_normalizer, cfg.reward_normalizer = normalizer.NormalizerFactory.get_normalizer(cfg)

    # # Setting up the logger
    cfg.logger = logger.Logger(cfg)
    cfg.log_config()

    if cfg.linear_probing_parent == "LaplaceEvaluate":
        parent = import_module("core.agent.laplace")
        parent = getattr(parent, "LaplaceEvaluate")
        agent = laplace.LaplaceEvaluate(cfg)
    elif cfg.linear_probing_parent == "DQNAgent":
        cfg.vf_loss_fn = optimizer.OptFactory.get_vf_loss_fn(cfg)
        cfg.vf_constr_fn = optimizer.OptFactory.get_constr_fn(cfg)
        cfg.eps_schedule = schedule.ScheduleFactory.get_eps_schedule(cfg)
        parent = import_module("core.agent.dqn")
        parent = getattr(parent, "DQNAgent")
        agent = dqn.DQNRepDistance(cfg)
    else:
        raise NotImplementedError

    # # linear probing
    lptask_all = cfg.linearprob_tasks
    for lptask in lptask_all:
        if "learning_rate" in lptask.keys(): # Sweeping does not use this block
            cfg.learning_rate = lptask["learning_rate"]
        cfg.linearprob_tasks = [lptask]
        cfg.linear_prob_task = linear_probing_tasks.get_linear_probing_task(cfg)
        cfg.logger.info("Linear Probing: training {}".format(lptask["task"]))
        lpagent = linear_probing.linear_probing(parent, cfg)
        test_funcs.run_linear_probing(lpagent)
        test_funcs.test_linear_probing(lpagent)
    cfg.logger.info("Linear Probing Ends")

    # distance
    """ Generate distance plot """
    # test_funcs.dqn_rep_distance_viz(agent)
    """ Evaluate representation distance """
    test_funcs.test_dqn_distance(agent)

    # Orthogonality
    test_funcs.test_orthogonality(agent)

    # Noninterference
    test_funcs.test_noninterference(agent)
    #
    # Decorrelation
    test_funcs.test_decorrelation(agent)

    # Sparsity
    test_funcs.test_sparsity(agent)

    # # Robustness
    # agent = dqn.DQNAgent(cfg)
    # test_funcs.test_robustness(agent)

