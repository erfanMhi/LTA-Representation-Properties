import os
import argparse

import core.environment.env_factory as environment
import core.network.net_factory as network
import core.network.optimizer as optimizer
import core.component.replay as replay
import core.utils.normalizer as normalizer
import core.component.representation as representation
import core.component.auxiliary_tasks as auxiliary_tasks
from core.agent import dqn_aux
from core.utils import torch_utils, schedule, logger, run_funcs
from experiment.sweeper import Sweeper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies param_setting number and parameter configuration')
    parser.add_argument('--config-file', default='experiment/config_files/dqn_test.json')
    parser.add_argument('--device', default=-1, type=int, )
    args = parser.parse_args()

    torch_utils.set_one_thread()
    torch_utils.random_seed(args.id)

    project_root = os.path.abspath(os.path.dirname(__file__))
    cfg = Sweeper(project_root, args.config_file).parse(args.id)
    cfg.device = torch_utils.select_device(args.device)

    cfg.rep_fn = representation.RepFactory.get_rep_fn(cfg)
    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)
    cfg.val_fn = network.NetFactory.get_val_fn(cfg)
    cfg.aux_fns = auxiliary_tasks.AuxFactory.get_aux_task(cfg)
    cfg.optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)
    cfg.vf_loss_fn = optimizer.OptFactory.get_vf_loss_fn(cfg)
    cfg.vf_constr_fn = optimizer.OptFactory.get_constr_fn(cfg)
    cfg.replay_fn = replay.ReplayFactory.get_replay_fn(cfg)
    cfg.eps_schedule = schedule.ScheduleFactory.get_eps_schedule(cfg)
    cfg.state_normalizer, cfg.reward_normalizer = normalizer.NormalizerFactory.get_normalizer(cfg)

    # Setting up the logger
    cfg.logger = logger.Logger(cfg)
    cfg.log_config()

    # Initializing the agent and running the experiment
    agent = getattr(dqn_aux, cfg.agent)(cfg)
    run_funcs.run_steps(agent)
