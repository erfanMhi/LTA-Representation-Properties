import os
from collections import namedtuple
from itertools import chain, combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import seaborn as sns
import torch
import copy

from core.agent import base
from core.utils import torch_utils
from core.utils.lipschitz import compute_lipschitz


class QLearningAgent(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

        rep_net = cfg.rep_fn()
        if cfg.rep_config['load_params']:
            path = os.path.join(cfg.data_root, cfg.rep_config['path'])
            print("loading from", path)
            rep_net.load_state_dict(torch.load(path, map_location=cfg.device))

        val_net = cfg.val_fn()
        if 'load_params' in cfg.val_fn_config:
            if cfg.val_fn_config['load_params']:
                path = os.path.join(cfg.data_root, cfg.val_fn_config['path'])
                print("loading value function from", path)
                val_net.load_state_dict(torch.load(path, map_location=cfg.device))

        params = list(rep_net.parameters()) + list(val_net.parameters())
        optimizer = cfg.optimizer_fn(params)

        self.rep_net = rep_net
        self.val_net = val_net
        self.optimizer = optimizer

        self.vf_loss = cfg.vf_loss_fn()
        self.vf_constr = cfg.vf_constr_fn()

        self.state = None
        self.action = None
        self.next_state = None

        if self.cfg.evaluate_interference:
            self.ac_last_sample = None
            self.ac_last_td2 = None
            self.update_interfs = []
            self.itera_interfs = []

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
        
        action = self.policy(self.state, self.cfg.eps_schedule())

        next_state, reward, done, _ = self.env.step([action])
        self.current_trans = [self.state.reshape((1, -1)), np.array([action]), np.array([reward]), next_state.reshape((1, -1)), np.array([int(done)])]
        self.state = next_state
        self.update_stats(reward, done)
        if self.cfg.update_network:
            self.update()

    def policy(self, state, eps):
        
        with torch.no_grad():
            phi = self.rep_net(self.cfg.state_normalizer(state))
            q_values = self.val_net(phi)

        q_values = torch_utils.to_np(q_values).flatten()

        if np.random.rand() < eps:
            action = np.random.randint(0, len(q_values))
        else:
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        return action

    def update(self):
        states, actions, rewards, next_states, terminals = self.current_trans
        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)

        actions = torch_utils.tensor(actions, self.cfg.device).long()

        if not self.cfg.rep_config['train_rep']:
            with torch.no_grad():
                phi = self.rep_net(states)
        else:
            phi = self.rep_net(states)

        q = self.val_net(phi)[self.batch_indices, actions]

        # Constructing the target
        with torch.no_grad():
            q_next = self.val_net(self.rep_net(next_states))
            q_next = q_next.max(1)[0]
            terminals = torch_utils.tensor(terminals, self.cfg.device)
            rewards = torch_utils.tensor(rewards, self.cfg.device)
            target = self.cfg.discount * q_next * (1 - terminals).float()
            target.add_(rewards.float())

        loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()
        constr = self.vf_constr(q, target, phi)
        loss += constr

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('q/loss/val_loss', loss.item(), self.total_steps)

    def log_tensorboard(self):
        rewards = self.ep_returns_queue
        mean, median, min, max = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)
        self.cfg.logger.tensorboard_writer.add_scalar('q/reward/average_reward', mean, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('q/reward/median_reward', median, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('q/reward/min_reward', min, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('q/reward/max_reward', max, self.total_steps)
                
    def log_file(self, elapsed_time=-1):
        rewards = self.ep_returns_queue
        total_episodes = len(self.episode_rewards)
        mean, median, min, max = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)

        log_str = 'TRAIN LOG: steps %d, episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'

        self.cfg.logger.info(log_str % (self.total_steps, total_episodes, mean, median,
                                        min, max, len(rewards),
                                        elapsed_time))

        return mean, median, min, max

    def log_lipschitz(self):
        lips, ratio_dv_dphi, corr = compute_lipschitz(self.cfg, self.rep_net, self.val_net, self.env)
        lipschitz_upper = np.prod(lips)
        mean, median, min, max = np.mean(ratio_dv_dphi), np.median(ratio_dv_dphi), \
                                 np.min(ratio_dv_dphi), np.max(ratio_dv_dphi)
        log_str = 'total steps %d, total episodes %3d, ' \
                  'Lipschitz: %.3f/%.5f/%.5f/%.5f/%.5f (upper/mean/median/min/max)'
        self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), lipschitz_upper, mean, median, min, max))

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        if early:
            path = os.path.join(parameters_dir, "rep_net_earlystop")
        elif self.cfg.checkpoints:
            path = os.path.join(parameters_dir, "rep_net_{}".format(self.total_steps))
        else:
            path = os.path.join(parameters_dir, "rep_net")
        torch.save(self.rep_net.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "val_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "val_net")
        torch.save(self.val_net.state_dict(), path)

    def load(self, parameters_dir, early):
        if early:
            path = os.path.join(parameters_dir, "rep_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "rep_net")
        self.rep_net.load_state_dict(torch.load(path, map_location=self.cfg.device))

        if early:
            path = os.path.join(parameters_dir, "val_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "val_net")
        self.val_net.load_state_dict(torch.load(path, map_location=self.cfg.device))


