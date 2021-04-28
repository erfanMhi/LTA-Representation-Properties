import os
from collections import namedtuple
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


class DQNSwitchHeadAgent(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

        rep_net = cfg.rep_fn()
        if cfg.rep_config['load_params']:
            path = os.path.join(cfg.data_root, cfg.rep_config['path'])
            print("loading from", path)
            rep_net.load_state_dict(torch.load(path))

        val_net_1 = cfg.val_fn_1()
        val_net_2 = cfg.val_fn_2()
        if 'load_params' in cfg.val_fn_config:
            if cfg.val_fn_config['load_params']:
                path1 = os.path.join(cfg.data_root, cfg.val_fn_config['path1'])
                path2 = os.path.join(cfg.data_root, cfg.val_fn_config['path2'])
                print("loading value function from", path1)
                val_net_1.load_state_dict(torch.load(path1))
                val_net_2.load_state_dict(torch.load(path2))

        params = list(rep_net.parameters()) + list(val_net_1.parameters()) + list(val_net_2.parameters())
        optimizer = cfg.optimizer_fn(params)

        # Creating target networks for value, representation, and auxiliary val_net_1
        rep_net_target = cfg.rep_fn()
        rep_net_target.load_state_dict(rep_net.state_dict())
        val_net_1_target = cfg.val_fn_1()
        val_net_1_target.load_state_dict(val_net_1.state_dict())
        val_net_2_target = cfg.val_fn_2()
        val_net_2_target.load_state_dict(val_net_2.state_dict())

        # Creating Target Networks
        TargetNets = namedtuple('TargetNets', ['rep_net', 'val_net_1', 'val_net_2'])
        targets = TargetNets(rep_net=rep_net_target, val_net_1=val_net_1_target, val_net_2=val_net_2_target)

        self.rep_net = rep_net
        self.val_net_1 = val_net_1
        self.val_net_2 = val_net_2
        self.optimizer = optimizer
        self.targets = targets

        self.vf_loss_1 = cfg.vf_loss_fn_1()
        self.vf_loss_2 = cfg.vf_loss_fn_2()
        self.vf_constr = cfg.vf_constr_fn()
        self.replay = cfg.replay_fn()

        self.state = None
        self.action = None
        self.next_state = None
        self.head = 0
        def flip(x):
            if x >= 0.999: return -1.001
            elif x <= -1.001: return 0.999
            else: return x
        self.flip = np.vectorize(flip)

        if self.cfg.evaluate_interference:
            self.ac_last_sample = None
            self.ac_last_td2 = None
            self.update_interfs = []
            self.itera_interfs = []

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
            self.head = self.env.info("use_head")

        action = self.policy(self.state, self.cfg.eps_schedule())

        next_state, reward, done, _ = self.env.step([action])
        left_green, left_red = self.env.info("left_fruit")
        if self.head == 0:
            reward_main = reward
        else:
            reward_main = self.flip(reward)
        # self.replay.feed([self.state, action, reward_main, next_state, int(done)])
        self.replay.feed([self.state, action, reward_main, next_state, int(left_green), int(left_red)])
        self.state = next_state
        # print('action: ', action)
        self.update_stats(reward, done)

        self.update()

    def policy(self, state, eps):
        if self.head == 0:
            with torch.no_grad():
                phi = self.rep_net(self.cfg.state_normalizer(state))
                q_values = self.val_net_1(phi)
        elif self.head == 1:
            with torch.no_grad():
                phi = self.rep_net(self.cfg.state_normalizer(state))
                q_values = self.val_net_2(phi)
        else:
            raise NotImplementedError

        q_values = torch_utils.to_np(q_values).flatten()

        if np.random.rand() < eps:
            action = np.random.randint(0, len(q_values))
        else:
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        return action

    def update(self):
        # states, actions, rewards_1, next_states, terminals = self.replay.sample()
        states, actions, rewards_1, next_states, left_green, left_red = self.replay.sample()
        terminals_1 = left_green == 0
        terminals_1 = terminals_1.astype(int)
        terminals_2 = left_red == 0
        terminals_2 = terminals_2.astype(int)
        rewards_2 = self.flip(rewards_1)
        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()
        # if len(np.where(terminals_1==1)[0])!=0 or len(np.where(terminals_2==1)[0])!=0:
        #     print("head", self.head)
        #     print(rewards_1)
        #     print(left_green)
        #     print(terminals_1)
        #     print()
        #     print(rewards_2)
        #     print(left_red)
        #     print(terminals_2)
        #     print()
        #     print()

        phi = self.rep_net(states)
        with torch.no_grad():
            phi_target = self.targets.rep_net(next_states)
            # terminals = torch_utils.tensor(terminals, self.cfg.device)

        # first head (true reward)
        q1 = self.val_net_1(phi)[self.batch_indices, actions]
        # Constructing the target for value function 1
        with torch.no_grad():
            q_next_1 = self.targets.val_net_1(phi_target)
            q_next_1 = q_next_1.max(1)[0]
            rewards_1 = torch_utils.tensor(rewards_1, self.cfg.device)
            terminals_1 = torch_utils.tensor(terminals_1, self.cfg.device)
            # target_1 = self.cfg.discount * q_next_1 * (1 - terminals).float()
            target_1 = self.cfg.discount * q_next_1 * (1 - terminals_1).float()
            target_1.add_(rewards_1.float())
        loss1 = self.vf_loss_1(q1, target_1)  # (q_next - q).pow(2).mul(0.5).mean()

        # self.optimizer.zero_grad()
        # loss1.backward()
        # self.optimizer.step()

        # second head (flip reward)
        q2 = self.val_net_2(phi)[self.batch_indices, actions]
        # Constructing the target for value function 2
        with torch.no_grad():
            q_next_2 = self.targets.val_net_2(phi_target)
            q_next_2 = q_next_2.max(1)[0]
            rewards_2 = torch_utils.tensor(rewards_2, self.cfg.device)
            terminals_2 = torch_utils.tensor(terminals_2, self.cfg.device)
            # target_2 = self.cfg.discount * q_next_2 * (1 - terminals).float()
            target_2 = self.cfg.discount * q_next_2 * (1 - terminals_2).float()
            target_2.add_(rewards_2.float())
        loss2 = self.vf_loss_2(q2, target_2)  # (q_next - q).pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        # loss2.backward()
        (loss1+loss2).backward()
        self.optimizer.step()

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn/loss/val_loss_1', loss1.item(), self.total_steps)
            self.cfg.logger.tensorboard_writer.add_scalar('dqn/loss/val_loss_2', loss2.item(), self.total_steps)

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net_1.load_state_dict(self.val_net_1.state_dict())
            self.targets.val_net_2.load_state_dict(self.val_net_2.state_dict())

    # # For debug
    # def update(self):
    #     states, actions, rewards_1, next_states, terminals = self.replay.sample()
    #     rewards_2 = self.flip(rewards_1)
    #     states = self.cfg.state_normalizer(states)
    #     next_states = self.cfg.state_normalizer(next_states)
    #
    #     actions = torch_utils.tensor(actions, self.cfg.device).long()
    #
    #     phi = self.rep_net(states)
    #
    #     # q1 = self.val_net_1(phi)[self.batch_indices, actions]
    #     q2 = self.val_net_2(phi)[self.batch_indices, actions]
    #
    #     # Constructing the target
    #     with torch.no_grad():
    #         # q_next_1 = self.targets.val_net_1(self.targets.rep_net(next_states))
    #         # q_next_1 = q_next_1.max(1)[0]
    #         q_next_2 = self.targets.val_net_2(self.targets.rep_net(next_states))
    #         q_next_2 = q_next_2.max(1)[0]
    #         terminals = torch_utils.tensor(terminals, self.cfg.device)
    #         # rewards_1 = torch_utils.tensor(rewards_1, self.cfg.device)
    #         rewards_2 = torch_utils.tensor(rewards_2, self.cfg.device)
    #         # target_1 = self.cfg.discount * q_next_1 * (1 - terminals).float()
    #         target_2 = self.cfg.discount * q_next_2 * (1 - terminals).float()
    #         # target_1.add_(rewards_1.float())
    #         target_2.add_(rewards_2.float())
    #
    #     # loss = self.vf_loss_1(q1, target_1)  # (q_next - q).pow(2).mul(0.5).mean()
    #     # loss += self.vf_loss_2(q2, target_2)  # (q_next - q).pow(2).mul(0.5).mean()
    #     loss = self.vf_loss_2(q2, target_2)  # (q_next - q).pow(2).mul(0.5).mean()
    #     # constr = self.vf_constr(q1, target_1, phi)
    #     # loss += constr
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
    #         self.cfg.logger.tensorboard_writer.add_scalar('dqn/loss/val_loss', loss.item(), self.total_steps)
    #
    #     if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
    #         self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
    #         self.targets.val_net_1.load_state_dict(self.val_net_1.state_dict())
    #         self.targets.val_net_2.load_state_dict(self.val_net_2.state_dict())

    def log_tensorboard(self):
        rewards = self.ep_returns_queue
        mean, median, min, max = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)
        self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/average_reward', mean, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/median_reward', median, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/min_reward', min, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/max_reward', max, self.total_steps)

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
        return
        # lips, ratio_dv_dphi, corr = compute_lipschitz(self.cfg, self.rep_net, self.val_net_1, self.env)
        # lipschitz_upper = np.prod(lips)
        # mean, median, min, max = np.mean(ratio_dv_dphi), np.median(ratio_dv_dphi), \
        #                          np.min(ratio_dv_dphi), np.max(ratio_dv_dphi)
        # log_str = 'total steps %d, total episodes %3d, ' \
        #           'Lipschitz: %.3f/%.5f/%.5f/%.5f/%.5f (upper/mean/median/min/max)'
        # self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), lipschitz_upper, mean, median, min, max))

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        if early:
            path = os.path.join(parameters_dir, "rep_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "rep_net")
        torch.save(self.rep_net.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "val_net_1_earlystop")
        else:
            path = os.path.join(parameters_dir, "val_net_1")
        torch.save(self.val_net_1.state_dict(), path)
        if early:
            path = os.path.join(parameters_dir, "val_net_2_earlystop")
        else:
            path = os.path.join(parameters_dir, "val_net_2")
        torch.save(self.val_net_2.state_dict(), path)


# class DQNSwitchHeadAgent(base.Agent):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#
#         rep_net = cfg.rep_fn()
#         if cfg.rep_config['load_params']:
#             path = os.path.join(cfg.data_root, cfg.rep_config['path'])
#             print("loading from", path)
#             rep_net.load_state_dict(torch.load(path))
#
#         val_net_1_1 = cfg.val_fn_1()
#         val_net_1_2 = cfg.val_fn_2()
#         if 'load_params' in cfg.val_fn_config:
#             if cfg.val_fn_config['load_params']:
#                 path1 = os.path.join(cfg.data_root, cfg.val_fn_config['path1'])
#                 path2 = os.path.join(cfg.data_root, cfg.val_fn_config['path2'])
#                 print("loading value function from", path1, path2)
#                 val_net_1_1.load_state_dict(torch.load(path1))
#                 val_net_1_2.load_state_dict(torch.load(path2))
#
#         params = list(rep_net.parameters()) + \
#                  list(val_net_1_1.parameters()) + \
#                  list(val_net_1_2.parameters())
#         optimizer = cfg.optimizer_fn(params)
#
#         rep_net_target = cfg.rep_fn()
#         rep_net_target.load_state_dict(rep_net.state_dict())
#         val_net_1_1_target = cfg.val_fn_1()
#         val_net_1_1_target.load_state_dict(val_net_1_1.state_dict())
#         val_net_1_2_target = cfg.val_fn_2()
#         val_net_1_2_target.load_state_dict(val_net_1_2.state_dict())
#
#         # Creating Target Networks
#         TargetNets = namedtuple('TargetNets', ['rep_net', 'val_net_1_1', 'val_net_1_2'])
#         targets = TargetNets(rep_net=rep_net, val_net_1_1=val_net_1_1_target, val_net_1_2=val_net_1_2_target)
#
#         self.rep_net = rep_net
#         self.val_net_1_1 = val_net_1_1
#         self.val_net_1_2 = val_net_1_2
#         self.optimizer = optimizer
#         self.targets = targets
#
#         # self.vf1_loss = copy.deepcopy(cfg.vf_loss_fn())
#         # self.vf2_loss = copy.deepcopy(cfg.vf_loss_fn())
#         # del self.vf_loss
#         self.vf_loss_1 = cfg.vf_loss_fn_1()
#         self.vf_loss_2 = cfg.vf_loss_fn_2()
#         self.vf_constr = cfg.vf_constr_fn()
#         self.replay = cfg.replay_fn()
#
#         self.state = None
#         self.action = None
#         self.next_state = None
#
#         if self.cfg.evaluate_interference:
#             self.ac_last_sample = None
#             self.ac_last_td2 = None
#             self.update_interfs = []
#             self.itera_interfs = []
#
#         self.head = None
#         # self.policy_head = self.val_net_1_1
#         def flip(x):
#             if x >= 0.999: return -1.001
#             elif x <= -1.001: return 0.999
#             else: return x
#
#         self.flip = np.vectorize(flip)
#
#     def step(self):
#         if self.reset is True:
#             self.state = self.env.reset()
#             self.reset = False
#             self.head = self.env.info("use_head")
#
#         action = self.policy(self.state, self.cfg.eps_schedule())
#
#         next_state, reward, done, _ = self.env.step([action])
#         if self.head == 0:
#             reward_main = reward
#         else:
#             reward_main = self.flip(reward)
#         self.replay.feed([self.state, action, reward_main, next_state, int(done)])
#         self.state = next_state
#         # print('action: ', action)
#         self.update_stats(reward, done)
#
#         self.update()
#
#     def policy(self, state, eps):
#         if self.head == 0:
#             with torch.no_grad():
#                 phi = self.rep_net(self.cfg.state_normalizer(state))
#                 q_values = self.val_net_1_1(phi)
#         else:
#             with torch.no_grad():
#                 phi = self.rep_net(self.cfg.state_normalizer(state))
#                 q_values = self.val_net_1_2(phi)
#         q_values = torch_utils.to_np(q_values).flatten()
#
#         if np.random.rand() < eps:
#             action = np.random.randint(0, len(q_values))
#         else:
#             action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
#         return action
#
#     def update(self):
#         states, actions, rewards, next_states, terminals = self.replay.sample()
#         states = self.cfg.state_normalizer(states)
#         next_states = self.cfg.state_normalizer(next_states)
#
#         actions = torch_utils.tensor(actions, self.cfg.device).long()
#
#         if not self.cfg.rep_config['train_rep']:
#             with torch.no_grad():
#                 phi = self.rep_net(states)
#         else:
#             phi = self.rep_net(states)
#
#         q1 = self.val_net_1_1(phi)[self.batch_indices, actions]
#
#         # Constructing the target
#         with torch.no_grad():
#             q_next_1 = self.targets.val_net_1_1(self.targets.rep_net(next_states))
#             q_next_1 = q_next_1.max(1)[0]
#             terminals = torch_utils.tensor(terminals, self.cfg.device)
#             rewards = torch_utils.tensor(rewards, self.cfg.device)
#             target = self.cfg.discount * q_next_1 * (1 - terminals).float()
#             target.add_(rewards.float())
#
#         loss = self.vf_loss_1(q1, target)  # (q_next - q).pow(2).mul(0.5).mean()
#         constr = self.vf_constr(q1, target, phi)
#         loss += constr
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         # grad_list = []
#         # for para in self.rep_net.parameters():
#         #     if para.grad is not None:
#         #         grad_list.append(para.grad.flatten().numpy())
#         # print(grad_list)
#         self.optimizer.step()
#
#         if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
#             self.cfg.logger.tensorboard_writer.add_scalar('dqn/loss/val_loss', loss.item(), self.total_steps)
#
#         if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
#             self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
#             self.targets.val_net_1_1.load_state_dict(self.val_net_1_1.state_dict())
#
#
#         # states, actions, rewards, next_states, terminals = self.replay.sample()
#         # states = self.cfg.state_normalizer(states)
#         # next_states = self.cfg.state_normalizer(next_states)
#         #
#         # actions = torch_utils.tensor(actions, self.cfg.device).long()
#         # rewards_flip = rewards * (-1)
#         #
#         # phi = self.rep_net(states)
#         #
#         # # Computing Loss for Value Function 1
#         # q = self.val_net_1_1(phi)[self.batch_indices, actions]
#         #
#         # with torch.no_grad():
#         #     nphi = self.targets.rep_net(next_states)
#         #     terminals = torch_utils.tensor(terminals, self.cfg.device)
#         #     rewards = torch_utils.tensor(rewards, self.cfg.device)
#         #     q_next = self.targets.val_net_1_1(nphi)
#         #     q_next = q_next.max(1)[0]
#         #     target = self.cfg.discount * q_next * (1 - terminals).float()
#         #     target.add_(rewards.float())
#         # loss = self.vf_loss_1(q, target)  # (q_next - q).pow(2).mul(0.5).mean()
#         #
#         # # Computing Loss for Value Function 2
#         # q = self.val_net_1_2(phi)[self.batch_indices, actions]
#         #
#         # with torch.no_grad():
#         #     rewards_flip = torch_utils.tensor(rewards_flip, self.cfg.device)
#         #     q_next= self.targets.val_net_1_2(nphi)
#         #     q_next = q_next.max(1)[0]
#         #     target = self.cfg.discount * q_next * (1 - terminals).float()
#         #     target.add_(rewards_flip.float())
#         # loss += self.vf_loss_2(q, target)  # (q_next - q).pow(2).mul(0.5).mean()
#         #
#         # constr = self.vf_constr(q, target, phi)
#         # loss += constr
#         #
#         # self.optimizer.zero_grad()
#         # loss.backward()
#         # self.optimizer.step()
#         #
#         # if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
#         #     self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
#         #     self.targets.val_net_1_1.load_state_dict(self.val_net_1_1.state_dict())
#         #     self.targets.val_net_1_2.load_state_dict(self.val_net_1_2.state_dict())
#
#     def log_tensorboard(self):
#         rewards = self.ep_returns_queue
#         mean, median, min, max = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)
#         self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/average_reward', mean, self.total_steps)
#         self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/median_reward', median, self.total_steps)
#         self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/min_reward', min, self.total_steps)
#         self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/max_reward', max, self.total_steps)
#
#     def log_file(self, elapsed_time=-1):
#         rewards = self.ep_returns_queue
#         total_episodes = len(self.episode_rewards)
#         mean, median, min, max = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)
#
#         log_str = 'TRAIN LOG: steps %d, episodes %3d, ' \
#                   'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'
#
#         self.cfg.logger.info(log_str % (self.total_steps, total_episodes, mean, median,
#                                         min, max, len(rewards),
#                                         elapsed_time))
#         return mean, median, min, max
#
#     def log_lipschitz(self):
#         return
#         # lips, ratio_dv_dphi, corr = compute_lipschitz(self.cfg, self.rep_net, self.val_net_1_1, self.env)
#         # lipschitz_upper = np.prod(lips)
#         # mean, median, min, max = np.mean(ratio_dv_dphi), np.median(ratio_dv_dphi), \
#         #                          np.min(ratio_dv_dphi), np.max(ratio_dv_dphi)
#         # log_str = 'total steps %d, total episodes %3d, ' \
#         #           'Lipschitz: (1) %.3f/%.5f/%.5f/%.5f/%.5f (upper/mean/median/min/max)'
#         # self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), lipschitz_upper, mean, median, min, max))
#         #
#         # lips, ratio_dv_dphi, corr = compute_lipschitz(self.cfg, self.rep_net, self.val_net_1_2, self.env)
#         # lipschitz_upper = np.prod(lips)
#         # mean, median, min, max = np.mean(ratio_dv_dphi), np.median(ratio_dv_dphi), \
#         #                          np.min(ratio_dv_dphi), np.max(ratio_dv_dphi)
#         # log_str = 'total steps %d, total episodes %3d, ' \
#         #           'Lipschitz: (2) %.3f/%.5f/%.5f/%.5f/%.5f (upper/mean/median/min/max)'
#         # self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), lipschitz_upper, mean, median, min, max))
#
#     def save(self, early=False):
#         parameters_dir = self.cfg.get_parameters_dir()
#         if early:
#             path = os.path.join(parameters_dir, "rep_net_earlystop")
#         else:
#             path = os.path.join(parameters_dir, "rep_net")
#         torch.save(self.rep_net.state_dict(), path)
#
#         if early:
#             path = os.path.join(parameters_dir, "val_net_1_earlystop")
#         else:
#             path = os.path.join(parameters_dir, "val_net_1")
#         torch.save(self.val_net_1_1.state_dict(), path)
#
#         if early:
#             path = os.path.join(parameters_dir, "val_net_1_2_earlystop")
#         else:
#             path = os.path.join(parameters_dir, "val_net_1_2")
#         torch.save(self.val_net_1_2.state_dict(), path)
#
#     def load(self, parameters_dir, early):
#         if early:
#             path = os.path.join(parameters_dir, "rep_net_earlystop")
#         else:
#             path = os.path.join(parameters_dir, "rep_net")
#         self.rep_net.load_state_dict(torch.load(path))
#
#         if early:
#             path = os.path.join(parameters_dir, "val_net_1_earlystop")
#         else:
#             path = os.path.join(parameters_dir, "val_net_1")
#         self.val_net_1_1.load_state_dict(torch.load(path))
#
#         if early:
#             path = os.path.join(parameters_dir, "val_net_1_2_earlystop")
#         else:
#             path = os.path.join(parameters_dir, "val_net_1_2")
#         self.val_net_1_2.load_state_dict(torch.load(path))
#
#         self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
#         self.targets.val_net_1_1.load_state_dict(self.val_net_1_1.state_dict())
#         self.targets.val_net_1_2.load_state_dict(self.val_net_1_2.state_dict())
