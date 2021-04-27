import os
import argparse

import core.environment.env_factory as environment
import core.network.net_factory as network
import core.network.optimizer as optimizer
import core.network.activations as activations
import core.component.replay as replay
import core.utils.normalizer as normalizer
import core.component.representation as representation
from core.agent import dqn_switch
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

    if cfg.rep_config["load_params"]:
        run_num = int(args.id / cfg.cumulative)
        cfg.rep_config["path"] = cfg.rep_config["path"].format(run_num)
        path = os.path.join(cfg.data_root, cfg.rep_config["path"])
        if not os.path.isfile(path):
            print("Run {} doesn't exist. {}".format(run_num, path))
            exit(1)

    if "load_params" in cfg.val_fn_config.keys() and cfg.val_fn_config["load_params"]:
        run_num = int(args.id / cfg.cumulative)
        cfg.val_fn_config["path"] = cfg.val_fn_config["path"].format(run_num)
        path = os.path.join(cfg.data_root, cfg.val_fn_config["path"])
        if not os.path.isfile(path):
            print("Run {} doesn't exist. {}".format(run_num, path))
            exit(1)

    cfg.rep_activation_fn = activations.ActvFactory.get_activation_fn(cfg)
    cfg.rep_fn = representation.RepFactory.get_rep_fn(cfg)
    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)
    cfg.val_fn = None
    cfg.val_fn_1 = network.NetFactory.get_val_fn(cfg)
    cfg.val_fn_2 = network.NetFactory.get_val_fn(cfg)
    cfg.optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)
    cfg.vf_loss_fn_1 = optimizer.OptFactory.get_vf_loss_fn(cfg)
    cfg.vf_loss_fn_2 = optimizer.OptFactory.get_vf_loss_fn(cfg)
    cfg.vf_constr_fn = optimizer.OptFactory.get_constr_fn(cfg)
    cfg.replay_fn = replay.ReplayFactory.get_replay_fn(cfg)
    cfg.eps_schedule = schedule.ScheduleFactory.get_eps_schedule(cfg)
    cfg.state_normalizer, cfg.reward_normalizer = normalizer.NormalizerFactory.get_normalizer(cfg)

    # Setting up the logger
    cfg.logger = logger.Logger(cfg)
    cfg.log_config()

    # Initializing the agent and running the experiment
    # agent = dqn.DQNAgent(cfg)
    agent = getattr(dqn_switch, cfg.agent)(cfg)
    if cfg.online_property:
        test_funcs.run_steps_onlineProperty(agent)
    else:
        run_funcs.run_steps(agent)