import os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import seaborn as sns
import torch

from core.agent import base
from core.utils import torch_utils
from core.utils.lipschitz import compute_lipschitz


class DQNAgent(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

        rep_net = cfg.rep_fn()
        if cfg.rep_config['load_params']:
            path = os.path.join(cfg.data_root, cfg.rep_config['path'])
            print("loading from", path)
            rep_net.load_state_dict(torch.load(path))

        val_net = cfg.val_fn()
        params = list(rep_net.parameters()) + list(val_net.parameters())
        optimizer = cfg.optimizer_fn(params)

        # Creating target networks for value, representation, and auxiliary val_net
        rep_net_target = cfg.rep_fn()
        rep_net_target.load_state_dict(rep_net.state_dict())
        val_net_target = cfg.val_fn()
        val_net_target.load_state_dict(val_net.state_dict())

        # Creating Target Networks
        TargetNets = namedtuple('TargetNets', ['rep_net', 'val_net'])
        targets = TargetNets(rep_net=rep_net_target, val_net=val_net_target)

        self.rep_net = rep_net
        self.val_net = val_net
        self.optimizer = optimizer
        self.targets = targets

        self.env = cfg.env_fn()
        self.vf_loss = cfg.vf_loss_fn()
        self.vf_constr = cfg.vf_constr_fn()
        self.replay = cfg.replay_fn()

        self.state = None
        self.action = None
        self.next_state = None

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False

        with torch.no_grad():
            phi = self.rep_net(self.cfg.state_normalizer(self.state))
            q_values = self.val_net(phi)

        q_values = torch_utils.to_np(q_values).flatten()
        if np.random.rand() < self.cfg.eps_schedule():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

        next_state, reward, done, _ = self.env.step([action])
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        self.state = next_state

        self.update_stats(reward, done)
        self.update()

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()
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
            q_next = self.targets.val_net(self.targets.rep_net(next_states))
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
        # grad_list = []
        # for para in self.rep_net.parameters():
        #     if para.grad is not None:
        #         grad_list.append(para.grad.flatten().numpy())
        # print(grad_list)
        self.optimizer.step()

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn/loss/val_loss', loss.item(), self.total_steps)

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())

    def eval_step(self, state):
        if np.random.rand() < self.cfg.eps_schedule.read_only():
            return np.random.randint(0, self.cfg.action_dim)
        else:
            q_values = self.val_net(self.rep_net(self.cfg.state_normalizer(state)))
            q_values = torch_utils.to_np(q_values).flatten()
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

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

        log_str = 'total steps %d, total episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'

        self.cfg.logger.info(log_str % (self.total_steps, total_episodes, mean, median,
                                        min, max, len(rewards),
                                        elapsed_time))

    def log_lipschitz(self):
        lips, ratio_dv_dphi, corr = compute_lipschitz(self.cfg, self.rep_net, self.val_net, self.env)
        lipschitz_upper = np.prod(lips)
        mean, median, min, max = np.mean(ratio_dv_dphi), np.median(ratio_dv_dphi), \
                                 np.min(ratio_dv_dphi), np.max(ratio_dv_dphi)
        log_str = 'total steps %d, total episodes %3d, ' \
                  'Lipschitz: %.3f/%.5f/%.5f/%.5f/%.5f (upper/mean/median/min/max)'
        self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), lipschitz_upper, mean, median, min, max))
        # log_str = 'total steps %d, total episodes %3d, ' \
        #           'Specialization: %.5f'
        # self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), corr))

    def visualize(self):
        """
        def get_visualization_segment: states, goal_states, map_coords
          states: list of states (XY, grayscale, or RGB)
          goal_states:
            one visualization plot for every goal_state
            each plot reflects the distance of states from the goal state
          map_coords: XY Coordinate of every state in states
        """
        try:
            segment = self.env.get_visualization_segment()
            states, state_coords, _, _ = segment

            states = self.cfg.state_normalizer(states)

            with torch.no_grad():
                values = self.val_net(self.rep_net(states)).max(dim=1)[0]
            
            # a plot with rows = num of goal-states and cols = 1
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            max_x = max(state_coords, key=lambda xy: xy[0])[0]
            max_y = max(state_coords, key=lambda xy: xy[1])[1]

            def compute_value_map(state_coords, values, size_x, size_y):
                value_map = np.zeros((size_x, size_y))
                for k, xy_coord in enumerate(state_coords):
                    x, y = xy_coord
                    value_map[x][y] = values[k].item()
                return value_map

            _value_map = compute_value_map(state_coords, values, max_x+1, max_y+1)
            sns.heatmap(_value_map, ax=ax)
            ax.set_title('Value Function')

            viz_dir = self.cfg.get_visualization_dir()
            viz_file = 'visualization_{}.png'.format(self.num_episodes)
            #viz_file = 'visualization.png'
            plt.savefig(os.path.join(viz_dir, viz_file))
            plt.close()

        except NotImplementedError:
            return

    def save(self):
        parameters_dir = self.cfg.get_parameters_dir()
        path = os.path.join(parameters_dir, "rep_net")
        torch.save(self.rep_net.state_dict(), path)

        path = os.path.join(parameters_dir, "val_net")
        torch.save(self.val_net.state_dict(), path)

    def load(self, parameters_dir):
        path = os.path.join(parameters_dir, "rep_net")
        self.rep_net.load_state_dict(torch.load(path))

        path = os.path.join(parameters_dir, "val_net")
        self.val_net.load_state_dict(torch.load(path))

        self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
        self.targets.val_net.load_state_dict(self.val_net.state_dict())

    # def visualize_rep(self):
    #     """
    #     def get_visualization_segment: states, goal_states, map_coords
    #       states: list of states (XY, grayscale, or RGB)
    #       goal_states:
    #         one visualization plot for every goal_state
    #         each plot reflects the distance of states from the goal state
    #       map_coords: XY Coordinate of every state in states
    #     """
    #     segment = self.env.get_visualization_segment()
    #     states, state_coords, _, _ = segment
    #     assert len(states) == len(state_coords)
    #
    #     states = self.cfg.state_normalizer(states)
    #     values = self.rep_net.forward(states).max(dim=1)[0]
    #     # a plot with rows = num of goal-states and cols = 1
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    #     max_x = max(state_coords, key=lambda xy: xy[0])[0]
    #     max_y = max(state_coords, key=lambda xy: xy[1])[1]
    #
    #     def compute_value_map(state_coords, values, size_x, size_y):
    #         value_map = np.zeros((size_x, size_y))
    #         for k, xy_coord in enumerate(state_coords):
    #             x, y = xy_coord
    #             value_map[x][y] = values[k].item()
    #         return value_map
    #
    #     _value_map = compute_value_map(state_coords, values, max_x + 1, max_y + 1)
    #     sns.heatmap(_value_map, ax=ax)
    #     ax.set_title('Value Function')
    #
    #     viz_dir = self.cfg.get_visualization_dir()
    #     # viz_file = 'visualization_{}.png'.format(self.num_episodes)
    #     viz_file = 'visualization_rep_dqn.png'
    #     plt.savefig(os.path.join(viz_dir, viz_file))
    #     plt.close()

    # def visualize_rep(self):
    #     """
    #     def get_visualization_segment: states, goal_states, map_coords
    #       states: list of states (XY, grayscale, or RGB)
    #       goal_states:
    #         one visualization plot for every goal_state
    #         each plot reflects the distance of states from the goal state
    #       map_coords: XY Coordinate of every state in states
    #     """
    #     segment = self.env.get_visualization_segment()
    #     states, state_coords, goal_states, goal_coords = segment
    #     assert len(states) == len(state_coords)
    #
    #     f_s = self.rep_net.phi(self.cfg.state_normalizer(states))
    #     f_g = self.rep_net.phi(self.cfg.state_normalizer(goal_states))
    #
    #     # a plot with rows = num of goal-states and cols = 1
    #     fig, ax = plt.subplots(nrows=len(goal_states), ncols=1, figsize=(6, 6 * len(goal_states)))
    #     max_x = max(state_coords, key=lambda xy: xy[0])[0]
    #     max_y = max(state_coords, key=lambda xy: xy[1])[1]
    #
    #     def compute_distance_map(f_states, f_goal, size_x, size_y):
    #         l2_vec = ((f_states - f_goal)**2).sum(dim=1)
    #         distance_map = np.zeros((size_x, size_y))
    #         for k, xy_coord in enumerate(state_coords):
    #             x, y = xy_coord
    #             distance_map[x][y] = l2_vec[k].item()
    #         return distance_map
    #
    #     for g_k in range(len(goal_states)):
    #         _distance_map = compute_distance_map(f_s, f_g[g_k], max_x+1, max_y+1)
    #         sns.heatmap(_distance_map, ax=ax[g_k])
    #         ax[g_k].set_title('Goal: {}'.format(goal_coords[g_k]))
    #
    #     viz_dir = self.cfg.get_visualization_dir()
    #     # viz_file = 'visualization_{}.png'.format(self.num_episodes)
    #     viz_file = 'visualization_rep_laplace.png'
    #     plt.savefig(os.path.join(viz_dir, viz_file))
    #     plt.close()
    #


class DQNRepDistance(DQNAgent):
    def __init__(self, cfg):
        super(DQNRepDistance, self).__init__(cfg)

    def visualize(self):
        """
        def get_visualization_segment: states, goal_states, map_coords
          states: list of states (XY, grayscale, or RGB)
          goal_states:
            one visualization plot for every goal_state
            each plot reflects the distance of states from the goal state
          map_coords: XY Coordinate of every state in states
        """
        segment = self.env.get_visualization_segment()
        states, state_coords, goal_states, goal_coords = segment
        assert len(states) == len(state_coords)

        with torch.no_grad():
            f_s = self.rep_net(self.cfg.state_normalizer(states))
            f_g = self.rep_net(self.cfg.state_normalizer(goal_states))

        # a plot with rows = num of goal-states and cols = 1
        fig, ax = plt.subplots(nrows=len(goal_states), ncols=1, figsize=(6, 6 * len(goal_states)))
        max_x = max(state_coords, key=lambda xy: xy[0])[0]
        max_y = max(state_coords, key=lambda xy: xy[1])[1]

        def compute_distance_map(f_states, f_goal, size_x, size_y):
            l2_vec = ((f_states - f_goal)**2).sum(dim=1)
            distance_map = np.zeros((size_x, size_y))
            for k, xy_coord in enumerate(state_coords):
                x, y = xy_coord
                distance_map[x][y] = l2_vec[k].item()
            return distance_map

        for g_k in range(len(goal_states)):
            _distance_map = compute_distance_map(f_s, f_g[g_k], max_x+1, max_y+1)
            sns.heatmap(_distance_map, ax=ax[g_k])
            ax[g_k].set_title('Goal: {}'.format(goal_coords[g_k]))

        viz_dir = self.cfg.get_visualization_dir()
        viz_file = 'visualization_rep.png'
        plt.savefig(os.path.join(viz_dir, viz_file))
        plt.close()


class DQNModelLearning(DQNAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.total_loss = np.zeros(cfg.stats_queue_size)
        self.loss_s = np.zeros(cfg.stats_queue_size)
        self.loss_r = np.zeros(cfg.stats_queue_size)
        self.loss_t = np.zeros(cfg.stats_queue_size)

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False

        phi = self.rep_net(self.cfg.state_normalizer(self.state))
        q_values, _, _, _ = self.val_net(np.expand_dims(phi, axis=0))
        q_values = torch_utils.to_np(q_values).flatten()

        if np.random.rand() < self.cfg.eps_schedule():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

        next_state, reward, done, _ = self.env.step([action])
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        self.state = next_state

        self.update_stats(reward, done)
        self.update_vf()

    def update_vf(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()
        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)

        actions = torch_utils.tensor(actions, self.cfg.device).long()
        q, pred_s, pred_r, pred_t = self.val_net(self.rep_net(states))
        q = q[self.batch_indices, actions]
        pred_r = pred_r[self.batch_indices, 0]
        pred_t = pred_t[self.batch_indices, 0]
        q_next = self.targets.val_net(self.targets.rep_net(next_states))[0] if self.cfg.use_target_network else \
                 self.val_net(self.rep_net(next_states))[0]
        q_next = q_next.detach().max(1)[0]
        next_states = torch_utils.tensor(next_states, self.cfg.device)
        terminals = torch_utils.tensor(terminals, self.cfg.device)
        rewards = torch_utils.tensor(rewards, self.cfg.device)
        target = self.cfg.discount * q_next * (1 - terminals).float()
        target.add_(rewards.float())

        vf_loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()
        s_loss = self.vf_loss(pred_s, next_states)
        r_loss = self.vf_loss(pred_r, rewards)
        t_loss = self.vf_loss(pred_t, terminals)
        loss = vf_loss + s_loss + r_loss + t_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.val_net.load_state_dict(self.val_net.state_dict())

    def eval_model(self, trajectory):
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]
        next_states = trajectory["next_states"]
        terminals = trajectory["terminals"]
        with torch.no_grad():
            states = self.cfg.state_normalizer(states)
            next_states = torch_utils.tensor(self.cfg.state_normalizer(next_states), self.cfg.device)
            # next_states = self.rep.phi(self.cfg.state_normalizer(next_states))
            rewards = torch_utils.tensor(self.cfg.reward_normalizer(rewards), self.cfg.device).view((-1, 1))
            terminals = torch_utils.tensor(terminals, self.cfg.device).view((-1, 1))
            # actions = torch_utils.tensor(self.one_hot_action(actions), self.cfg.device)
            _, pred_ns, pred_r, pred_t = self.val_net(self.rep_net(states))
            loss_s = torch.mean(torch.sum((pred_ns - next_states) ** 2, 1)).item()
            loss_r = torch.mean(torch.sum((pred_r - rewards) ** 2, 1)).item()
            loss_t = torch.mean(torch.sum((pred_t - terminals) ** 2, 1)).item()
            pred_ns, pred_r, pred_t = pred_ns.numpy(), pred_r.numpy().reshape(-1), pred_t.numpy().reshape(-1)

        log_str = 'Evaluation: %.4f/%.4f/%.4f (nextstate/reward/termination)\n'
        self.cfg.logger.info(log_str % (loss_s, loss_r, loss_t))

    def eval_step(self, state):
        if np.random.rand() < self.cfg.eps_schedule.read_only():
            return np.random.randint(0, self.cfg.action_dim)
        else:
            phi = self.rep_net(self.cfg.state_normalizer(state))
            q_values, _, _, _ = self.val_net(np.expand_dims(phi, axis=0))
            q_values = torch_utils.to_np(q_values).flatten()
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def visualize(self):
        segment = self.env.get_visualization_segment()
        states, state_coords, _, _ = segment
        assert len(states) == len(state_coords)

        states = self.cfg.state_normalizer(states)
        
        with torch.no_grad():
            values = self.val_net(self.rep_net(states))[0].max(dim=1)[0]
        # a plot with rows = num of goal-states and cols = 1
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        max_x = max(state_coords, key=lambda xy: xy[0])[0]
        max_y = max(state_coords, key=lambda xy: xy[1])[1]

        def compute_value_map(state_coords, values, size_x, size_y):
            value_map = np.zeros((size_x, size_y))
            for k, xy_coord in enumerate(state_coords):
                x, y = xy_coord
                value_map[x][y] = values[k].item()
            return value_map

        _value_map = compute_value_map(state_coords, values, max_x+1, max_y+1)
        sns.heatmap(_value_map, ax=ax)
        ax.set_title('Value Function')

        viz_dir = self.cfg.get_visualization_dir()
        # viz_file = 'visualization_{}.png'.format(self.num_episodes)
        viz_file = 'visualization.png'
        plt.savefig(os.path.join(viz_dir, viz_file))
        plt.close()