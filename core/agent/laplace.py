import os
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import seaborn as sns

from core.agent import base
from core.utils.torch_utils import tensor


class Laplace(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.replay = cfg.replay_fn()
        self.rep_net = cfg.rep_fn()

        if cfg.rep_config['load_params']:
            path = os.path.join(cfg.data_root, cfg.rep_config['path'])
            print("loading from", path)
            self.rep_net.load_state_dict(torch.load(path))

        self.optimizer = cfg.optimizer_fn(self.rep_net.parameters())
        self.env = cfg.env_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.trajectory = []
        # exp_shape = [1, self.timeout] + list(cfg.rep_config["in_dim"])
        # self.trajectory = np.zeros(exp_shape)
        # self.replay.set_exp_shape(exp_shape)

        self.tau = list(range(1, self.timeout+1))
        tau_probs = [cfg.lmbda**(x-1)-cfg.lmbda**x for x in self.tau]
        self.tau_probs = [x / np.sum(tau_probs) for x in tau_probs]
        self.sample_tau = lambda: np.random.choice(range(1, cfg.timeout+1), p=self.tau_probs, size=cfg.batch_size)
        self._tensor = lambda x: tensor(self.cfg.state_normalizer(x), self.cfg.device)

        self.total_loss = np.zeros(cfg.stats_queue_size)
        self.attractive_loss = np.zeros(cfg.stats_queue_size)
        self.repulsive_loss = np.zeros(cfg.stats_queue_size)

    def random_step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
            self.ep_steps = 0

        action = np.random.randint(0, self.cfg.action_dim)

        next_state, reward, done, _ = self.env.step([action])
        self.trajectory.append(self.state)
        self.state = next_state

        self.ep_steps += 1
        if self.ep_steps == self.timeout:
            self.replay.feed([self.trajectory])
            self.trajectory = []
            self.reset = True
        return self.replay.size()

    def update_step(self):
        self.learn_laplace()
        self.total_steps += 1

    # def step(self):
    #     if self.reset is True:
    #         self.state = self.env.reset()
    #         self.reset = False
    #
    #     action = np.random.randint(0, self.cfg.action_dim)
    #
    #     next_state, reward, done, _ = self.env.step([action])
    #     self.trajectory.append(self.state)
    #     # self.trajectory[self.ep_steps] = self.state
    #     self.state = next_state
    #
    #     if self.replay.size() > self.cfg.batch_size:
    #         self.learn_laplace()
    #
    #     self.total_steps += 1
    #     self.ep_steps += 1
    #     if self.ep_steps == self.timeout:
    #         self.replay.feed([self.trajectory])
    #         # self.replay.feed(self.trajectory, self.ep_steps)
    #         self.trajectory = []
    #         # self.trajectory.fill(0)
    #         # self.learn_laplace()
    #
    #         self.ep_steps = 0
    #         self.num_episodes += 1
    #         self.reset = True

    def learn_laplace(self):

        attractive_loss, f_u = self.get_attractive_loss()
        repulsive_loss = self.get_repulsive_loss(f_u)

        total_loss = attractive_loss + self.cfg.beta * repulsive_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.update_loss_stats(total_loss.item(), attractive_loss.item(), repulsive_loss.item())

    def get_attractive_loss(self):
        trajectories = self.replay.sample()[0]
        # trajectories, _ = self.replay.sample()
        samples = self.sample_tau()

        batch_idx = np.arange(self.cfg.batch_size)
        u = self._tensor(trajectories[batch_idx, self.cfg.timeout - 1 - samples])
        f_u = self.rep_net(u)

        v = self._tensor(trajectories[batch_idx, -1])
        f_v = self.rep_net(v)

        loss = 0.5 * torch.mean(torch.sum((f_u - f_v)**2, 1))
        return loss, f_u

    def get_repulsive_loss(self, f_u):
        trajectories_v = self.replay.sample()[0]
        # trajectories_v, _ = self.replay.sample()
        batch_idx = np.arange(self.cfg.batch_size)
        v = self._tensor(trajectories_v[batch_idx, - 1])
        f_v = self.rep_net(v)

        f_u_ = f_u.view(self.cfg.batch_size, 1, self.cfg.rep_config['out_dim'])
        f_v_ = f_v.view(self.cfg.batch_size, self.cfg.rep_config['out_dim'], 1)
        dot_product = torch.bmm(f_u_, f_v_)**2
        dot_product = dot_product.view(self.cfg.batch_size)

        norm_fu = torch.sum(f_u ** 2, 1) * self.cfg.delta
        norm_fv = torch.sum(f_v ** 2, 1) * self.cfg.delta
        d = self.cfg.rep_config['out_dim']*self.cfg.delta ** 2
        repulsive_loss = dot_product - norm_fu - norm_fv + d
        return torch.mean(repulsive_loss)

    def save(self):
        parameters_dir = self.cfg.get_parameters_dir()
        path = os.path.join(parameters_dir, "laplace")
        torch.save(self.rep_net.state_dict(), path)

    def load(self, parameters_dir):
        # parameters_dir = self.cfg.get_parameters_dir()
        # path = os.path.join(parameters_dir, "laplace")
        path = parameters_dir
        self.rep_net.load_state_dict(torch.load(path))

    def log_tensorboard(self):
        num_stats = min(self.stats_counter, self.cfg.stats_queue_size)
        total_loss = np.mean(self.total_loss[:num_stats])
        attractive_loss = np.mean(self.attractive_loss[:num_stats])
        repulsive_loss = np.mean(self.repulsive_loss[:num_stats])

        self.cfg.logger.tensorboard_writer.add_scalar('laplace/loss/total_loss', total_loss, self.num_episodes)
        self.cfg.logger.tensorboard_writer.add_scalar('laplace/loss/attractive_loss', attractive_loss, self.num_episodes)
        self.cfg.logger.tensorboard_writer.add_scalar('laplace/loss/repulsive_loss', repulsive_loss, self.num_episodes)

    def log_file(self, elapsed_time=-1):
        num_stats = min(self.stats_counter, self.cfg.stats_queue_size)
        total_loss = np.mean(self.total_loss[:num_stats])
        attractive_loss = np.mean(self.attractive_loss[:num_stats])
        repulsive_loss = np.mean(self.repulsive_loss[:num_stats])
        log_str = 'total steps %d, total episodes %3d, ' \
                  'loss %.10f/%.10f/%.10f (total/attractive/repuslive), %.2f steps/s'
        self.cfg.logger.info(log_str % (self.total_steps, self.num_episodes, total_loss, attractive_loss,
                             repulsive_loss, elapsed_time))

    def update_loss_stats(self, total_loss, attractive_loss, repulsive_loss):
        stats_idx = self.stats_counter % self.cfg.stats_queue_size
        self.total_loss[stats_idx] = total_loss
        self.attractive_loss[stats_idx] = attractive_loss
        self.repulsive_loss[stats_idx] = repulsive_loss
        self.stats_counter += 1

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

        f_s = self.rep_net(self._tensor(states))
        f_g = self.rep_net(self._tensor(goal_states))

        # a plot with rows = num of goal-states and cols = 1
        fig, ax = plt.subplots(nrows=len(goal_states), ncols=1, figsize=(6, 6 * len(goal_states)))
        if len(goal_states) == 1:
            ax = [ax]
        max_x = max(state_coords, key=lambda xy: xy[0])[0]
        max_y = max(state_coords, key=lambda xy: xy[1])[1]

        def compute_distance_map(f_states, f_goal, size_x, size_y):
            l2_vec = ((f_states - f_goal)**2).sum(dim=1)
            distance_map = np.zeros((size_x, size_y))
            for k, xy_coord in enumerate(state_coords):
                x, y = xy_coord
                distance_map[x][y] = l2_vec[k].item()
            # distance_map = distance_map.clip(0, 0.04)
            return distance_map

        for g_k in range(len(goal_states)):
            _distance_map = compute_distance_map(f_s, f_g[g_k], max_x+1, max_y+1)
            sns.heatmap(_distance_map, ax=ax[g_k])
            ax[g_k].set_title('Goal: {}'.format(goal_coords[g_k]))

        viz_dir = self.cfg.get_visualization_dir()
        # viz_file = 'visualization_{}.png'.format(self.num_episodes)
        viz_file = 'visualization.png'
        plt.savefig(os.path.join(viz_dir, viz_file))
        plt.close()

class LaplaceEvaluate(Laplace):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rep_net = cfg.rep_fn()
        if cfg.rep_config['load_params']:
            path = os.path.join(cfg.data_root, cfg.rep_config['path'])
            self.rep_net.load_state_dict(torch.load(path))

        self.replay = cfg.replay_fn()
        self.optimizer = cfg.optimizer_fn(self.rep_net.parameters())
        self.env = cfg.env_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.trajectory = []
        self.tau = list(range(1, self.timeout+1))
        tau_probs = [cfg.lmbda**(x-1)-cfg.lmbda**x for x in self.tau]
        self.tau_probs = [x / np.sum(tau_probs) for x in tau_probs]
        self.sample_tau = lambda: np.random.choice(range(1, cfg.timeout+1), p=self.tau_probs, size=cfg.batch_size)
        self._tensor = lambda x: tensor(self.cfg.state_normalizer(x), self.cfg.device)

        self.total_loss = np.zeros(cfg.stats_queue_size)
        self.attractive_loss = np.zeros(cfg.stats_queue_size)
        self.repulsive_loss = np.zeros(cfg.stats_queue_size)


class LaplaceAux(Laplace):
    def __init__(self, cfg):
        super().__init__(cfg)
        aux_tasks = [aux_fn(cfg) for aux_fn in cfg.aux_fns]
        params = list(self.rep_net.parameters())
        for aux_task in aux_tasks:
            params += list(aux_task.parameters())
        self.optimizer = cfg.optimizer_fn(params)
        self.aux_nets = aux_tasks
        self.aux_loss = np.zeros(cfg.stats_queue_size)

    def random_step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
            self.ep_steps = 0

        action = np.random.randint(0, self.cfg.action_dim)

        next_state, reward, done, _ = self.env.step([action])
        self.trajectory.append(np.concatenate([self.state.flatten(), [action], [reward]]))
        self.state = next_state

        self.ep_steps += 1
        if self.ep_steps == self.timeout:
            self.replay.feed([self.trajectory])
            self.trajectory = []
            self.reset = True
        return self.replay.size()

    def learn_laplace(self):
        attractive_loss, u, f_u, act, rwd = self.get_attractive_loss()
        repulsive_loss = self.get_repulsive_loss(f_u)
        aux_loss = self.get_aux_loss(u, f_u, act, rwd, attractive_loss)

        total_loss = attractive_loss + self.cfg.beta * repulsive_loss + aux_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.update_loss_stats(total_loss.item(), attractive_loss.item(), repulsive_loss.item(), aux_loss.item())

    def get_attractive_loss(self):
        trajectories = self.replay.sample()[0]
        # trajectories, _ = self.replay.sample()
        samples = self.sample_tau()

        batch_idx = np.arange(self.cfg.batch_size)
        u_sar = trajectories[batch_idx, self.cfg.timeout - 1 - samples]
        u_temp = self.cfg.state_normalizer(u_sar[:, :-2].reshape(([-1]+self.cfg.rep_config["in_dim"])))
        u = tensor(u_temp, self.cfg.device)
        act = u_sar[:, -2] #tensor(u_sar[:, -2], self.cfg.device)
        rwd = tensor(u_sar[:, -1], self.cfg.device)
        f_u = self.rep_net(u)

        # v = self._tensor(trajectories[batch_idx, -1][:, 0])
        v = self._tensor(trajectories[batch_idx, - 1][:, :-2].reshape(([-1]+self.cfg.rep_config["in_dim"])))
        f_v = self.rep_net(v)

        loss = 0.5 * torch.mean(torch.sum((f_u - f_v)**2, 1))
        return loss, u_temp, f_u, act, rwd

    def get_repulsive_loss(self, f_u):
        trajectories_v = self.replay.sample()[0]
        # trajectories_v, _ = self.replay.sample()
        batch_idx = np.arange(self.cfg.batch_size)
        v = self._tensor(trajectories_v[batch_idx, - 1][:, :-2].reshape(([-1]+self.cfg.rep_config["in_dim"])))
        f_v = self.rep_net(v)

        f_u_ = f_u.view(self.cfg.batch_size, 1, self.cfg.rep_config['out_dim'])
        f_v_ = f_v.view(self.cfg.batch_size, self.cfg.rep_config['out_dim'], 1)
        dot_product = torch.bmm(f_u_, f_v_)**2
        dot_product = dot_product.view(self.cfg.batch_size)

        norm_fu = torch.sum(f_u ** 2, 1) * self.cfg.delta
        norm_fv = torch.sum(f_v ** 2, 1) * self.cfg.delta
        d = self.cfg.rep_config['out_dim']*self.cfg.delta ** 2
        repulsive_loss = dot_product - norm_fu - norm_fv + d
        return torch.mean(repulsive_loss)

    def get_aux_loss(self, state, rep, action, reward, loss):
        aux_loss = torch.zeros_like(loss)
        for i, aux_net in enumerate(self.aux_nets):
            transition = (state, action, reward, None, None)
            aux_loss += aux_net.compute_loss(transition, rep, None, None)
        return aux_loss

    def log_tensorboard(self):
        num_stats = min(self.stats_counter, self.cfg.stats_queue_size)
        total_loss = np.mean(self.total_loss[:num_stats])
        attractive_loss = np.mean(self.attractive_loss[:num_stats])
        repulsive_loss = np.mean(self.repulsive_loss[:num_stats])
        reward_loss = np.mean(self.aux_loss[:num_stats])

        self.cfg.logger.tensorboard_writer.add_scalar('laplace/loss/total_loss', total_loss, self.num_episodes)
        self.cfg.logger.tensorboard_writer.add_scalar('laplace/loss/attractive_loss', attractive_loss, self.num_episodes)
        self.cfg.logger.tensorboard_writer.add_scalar('laplace/loss/repulsive_loss', repulsive_loss, self.num_episodes)
        self.cfg.logger.tensorboard_writer.add_scalar('laplace/loss/aux_loss', reward_loss, self.num_episodes)

    def log_file(self, elapsed_time=-1):
        num_stats = min(self.stats_counter, self.cfg.stats_queue_size)
        total_loss = np.mean(self.total_loss[:num_stats])
        attractive_loss = np.mean(self.attractive_loss[:num_stats])
        repulsive_loss = np.mean(self.repulsive_loss[:num_stats])
        reward_loss = np.mean(self.aux_loss[:num_stats])

        log_str = 'total steps %d, total episodes %3d, ' \
                  'loss %.10f/%.10f/%.10f/%.10f (total/attractive/repuslive/aux), %.2f steps/s'
        self.cfg.logger.info(log_str % (self.total_steps, self.num_episodes, total_loss, attractive_loss,
                             repulsive_loss, reward_loss, elapsed_time))

    def update_loss_stats(self, total_loss, attractive_loss, repulsive_loss, reward_loss):
        stats_idx = self.stats_counter % self.cfg.stats_queue_size
        self.total_loss[stats_idx] = total_loss
        self.attractive_loss[stats_idx] = attractive_loss
        self.repulsive_loss[stats_idx] = repulsive_loss
        self.aux_loss[stats_idx] = reward_loss
        self.stats_counter += 1
