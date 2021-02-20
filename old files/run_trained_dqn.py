import os
import argparse
import numpy as np

import core.environment.env_factory as environment
import core.network.net_factory as network
import core.network.optimizer as optimizer
import core.component.replay as replay
import core.utils.normalizer as normalizer
import core.component.representation as representation
from core.agent import trained_dqn
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
    cfg.optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)
    cfg.vf_loss_fn = optimizer.OptFactory.get_vf_loss_fn(cfg)
    cfg.replay_fn = replay.ReplayFactory.get_replay_fn(cfg)
    cfg.eps_schedule = schedule.ScheduleFactory.get_eps_schedule(cfg)
    cfg.state_normalizer, cfg.reward_normalizer = normalizer.NormalizerFactory.get_normalizer(cfg)

    # Setting up the logger
    cfg.logger = logger.Logger(cfg)
    cfg.log_config()


    """
    Collect World
    """
    # agent = trained_dqn.TrainedDQNAgent(cfg)
    # run_funcs.run_steps(agent)
    # sampled_states_green = np.array(agent.sampled_states)
    # sampled_next_green = np.array(agent.sampled_next_states)
    # sampled_different_green = np.array(agent.sampled_different)
    # sampled_action_green = np.array(agent.sampled_actions)
    # sampled_reward_green = np.array(agent.sampled_rewards)
    # sampled_termin_green = np.array(agent.sampled_termins)
    #
    # torch_utils.random_seed(args.id + 5)
    # cfg.id = args.id + 5
    # agent = trained_dqn.TrainedDQNAgent(cfg)
    # run_funcs.run_steps(agent)
    # sampled_states_red = np.array(agent.sampled_states)
    # sampled_next_red = np.array(agent.sampled_next_states)
    # sampled_different_red = np.array(agent.sampled_different)
    # sampled_action_red = np.array(agent.sampled_actions)
    # sampled_reward_red = np.array(agent.sampled_rewards)
    # sampled_termin_red = np.array(agent.sampled_termins)
    #
    #
    # for k in range(len(sampled_states_red)):
    #     im = np.copy(sampled_states_red[k])
    #     sampled_states_red[k][:, :, 0] = im[:, :, 1]
    #     sampled_states_red[k][:, :, 1] = im[:, :, 0]
    #     im2 = np.copy(sampled_next_red[k])
    #     sampled_next_red[k][:, :, 0] = im2[:, :, 1]
    #     sampled_next_red[k][:, :, 1] = im2[:, :, 0]
    #     im3 = np.copy(sampled_different_red[k])
    #     sampled_different_red[k][:, :, 0] = im2[:, :, 1]
    #     sampled_different_red[k][:, :, 1] = im2[:, :, 0]
    #
    # # save_path = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "lip_sampled_states.npy")
    # # sampled_states = np.concatenate([sampled_states_green, sampled_states_red])
    # # np.save(save_path, sampled_states)
    #
    # # id 10
    # # save_path = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "retaining_test_states.npy")
    # # sampled_states = np.concatenate([sampled_states_green, sampled_states_red])
    # # np.save(save_path, sampled_states)
    #
    # # # id 0
    # save_path_s = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_current_states_sameEP.npy")
    # save_path_sp = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_next_states_sameEP.npy")
    # save_path_diff = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_different_states_sameEP.npy")
    # save_path_a = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_actions_sameEP.npy")
    # save_path_r = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_rewards_sameEP.npy")
    # save_path_g = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_terminals_sameEP.npy")
    # sampled_states = np.concatenate([sampled_states_green, sampled_states_red])
    # next_states = np.concatenate([sampled_next_green, sampled_next_red])
    # different_states = np.concatenate([sampled_different_green, sampled_different_red])
    # actions = np.concatenate([sampled_action_green, sampled_action_red])
    # rewards = np.concatenate([sampled_reward_green, sampled_reward_red])
    # termins = np.concatenate([sampled_termin_green, sampled_termin_red])
    # np.save(save_path_s, sampled_states)
    # np.save(save_path_sp, next_states)
    # np.save(save_path_diff, different_states)
    # np.save(save_path_a, actions)
    # np.save(save_path_r, rewards)
    # np.save(save_path_g, termins)

    """
    Grid world
    """
    # id 0
    agent = trained_dqn.RandomAgent(cfg)
    run_funcs.run_steps(agent)
    sampled_states = np.array(agent.sampled_states)
    save_path = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "lip_sampled_states.npy")
    np.save(save_path, sampled_states)

    # id 10
    agent = trained_dqn.RandomAgent(cfg)
    run_funcs.run_steps(agent)
    sampled_states = np.array(agent.sampled_states)
    save_path = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "retaining_test_states.npy")
    np.save(save_path, sampled_states)

    # id 0
    agent = trained_dqn.RandomAgent(cfg)
    run_funcs.run_steps(agent)
    sampled_states = np.array(agent.sampled_states)
    next_states = np.array(agent.sampled_next_states)
    actions = np.array(agent.sampled_actions)
    rewards = np.array(agent.sampled_rewards)
    terminals = np.array(agent.sampled_termins)
    save_path_s = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_current_states.npy")
    save_path_sp = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_next_states.npy")
    save_path_a = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_actions.npy")
    save_path_r = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_rewards.npy")
    save_path_g = os.path.join(cfg.get_log_dir(), os.pardir, os.pardir, os.pardir, "distance_terminals.npy")
    np.save(save_path_s, sampled_states)
    np.save(save_path_sp, next_states)
    np.save(save_path_a, actions)
    np.save(save_path_r, rewards)
    np.save(save_path_g, terminals)
