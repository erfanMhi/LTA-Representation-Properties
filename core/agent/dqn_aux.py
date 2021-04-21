import torch
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
from inspect import signature

from core.agent import dqn
from core.utils import torch_utils
from core.utils.lipschitz import compute_dynamics_awareness, compute_decorrelation


class DQNAuxAgent(dqn.DQNAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        aux_tasks = [aux_fn(cfg) for aux_fn in cfg.aux_fns]
        params = list(self.rep_net.parameters()) + list(self.val_net.parameters())
        for aux_task in aux_tasks:
            params += list(aux_task.parameters())
        self.optimizer = cfg.optimizer_fn(params)

        # Creating target networks for value, representation, and auxiliary val_net
        rep_net_target = cfg.rep_fn()
        rep_net_target.load_state_dict(self.rep_net.state_dict())
        val_net_target = cfg.val_fn()
        val_net_target.load_state_dict(self.val_net.state_dict())

        aux_net_targets = [aux_fn(cfg) for aux_fn in cfg.aux_fns]
        for i, ant in enumerate(aux_net_targets):
            ant.load_state_dict(aux_tasks[i].state_dict())

        # Creating Target Networks
        TargetNets = namedtuple('TargetNets', ['rep_net', 'val_net', 'aux_nets'])
        targets = TargetNets(rep_net=rep_net_target, val_net=val_net_target, aux_nets=aux_net_targets)
        self.targets = targets
        self.aux_nets = aux_tasks

        # TODO: Remove this patchwork
        self.cfg.agent = self

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)

        # Computing Loss for Value Function
        q = self.val_net(phi)[self.batch_indices, actions]
        nphi = self.targets.rep_net(next_states)
        q_next = self.targets.val_net(nphi).detach().max(1)[0]
        terminals = torch_utils.tensor(terminals, self.cfg.device)
        rewards = torch_utils.tensor(rewards, self.cfg.device)
        target = self.cfg.discount * q_next * (1 - terminals).float()
        target.add_(rewards.float())
        loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()

        # Computing Loss for Aux Tasks
        aux_loss = torch.zeros_like(loss)
        for i, aux_net in enumerate(self.aux_nets):
            transition = (states, actions, rewards, next_states, terminals)
            aux_loss += aux_net.compute_loss(transition, phi, nphi)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/val_loss', loss.item(), self.total_steps)

        loss += aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
            for i, ant in enumerate(self.targets.aux_nets):
                ant.load_state_dict(self.aux_nets[i].state_dict())

    def get_aux_targets(self, x):
        return [aux.get_aux_target(x) for aux in self.aux_nets]

    def visualize(self):
        return


    def visualize_distance(self):
        env = self.cfg.env_fn()
        with torch.no_grad():
            goal = env.generate_state([env.goal_x, env.goal_y])
            for i in range(4):
                i = torch_utils.tensor(i, self.cfg.device).long()
                goal_aux = self.aux_nets[0].aux_predictor(self.rep_net(self.cfg.state_normalizer(goal)), i).detach().numpy()
                all_pts = np.zeros((env.max_x - env.min_x + 1, env.max_y - env.min_y + 1))
                for x in range(env.max_x - env.min_x + 1):
                    for y in range(env.max_y - env.min_y + 1):
                        if env.obstacles_map[x, y] == 0:
                            obs = env.generate_state([x, y])
                            all_pts[x, y] = np.linalg.norm(self.aux_nets[0].aux_predictor(self.rep_net(self.cfg.state_normalizer(obs)), i).detach().numpy() - goal_aux)
                plt.figure()
                plt.imshow(all_pts)
                plt.colorbar()
                viz_dir = self.cfg.get_visualization_dir()
                plt.savefig(os.path.join(viz_dir, 'aux_distance_{}.jpg'.format(i)))
                plt.close()


class DQNNasAuxAgent(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.val_loss = [0.0]
        self.aux_loss = [0.0]
        self.aux_attractive_loss = [0.0]
        self.aux_repulsive_loss = [0.0]
        self.queue_size = 1000

    def log_dynamics_awareness(self):
        dynamics_awareness = compute_dynamics_awareness(self.cfg, self.rep_net)
        log_str = 'total steps %d, total episodes %3d, Dynamics Awareness: %.5f'
        self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), dynamics_awareness))

    def log_decorrelation(self):
        decorrelation = compute_decorrelation(self.cfg, self.rep_net, self.env)
        log_str = 'total steps %d, total episodes %3d, Decorrelation: %.5f'
        self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), decorrelation))

    def log_losses(self):
        log_str = 'total steps %d, total episodes %3d, Val-loss: %.7f, Aux-loss: %.7f, Attractive-loss: %.7f,  Repulsive-loss: %.7f'
        self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), np.mean(self.val_loss),
                                        np.mean(self.aux_loss), np.mean(self.aux_attractive_loss),
                                        np.mean(self.aux_repulsive_loss)))

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)

        # Computing Loss for Value Function
        q = self.val_net(phi)[self.batch_indices, actions]
        nphi = self.targets.rep_net(next_states)
        q_next = self.targets.val_net(nphi).detach().max(1)[0]
        terminals = torch_utils.tensor(terminals, self.cfg.device)
        rewards = torch_utils.tensor(rewards, self.cfg.device)
        target = self.cfg.discount * q_next * (1 - terminals).float()
        target.add_(rewards.float())
        loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()

        # Computing Loss for Aux Tasks
        aux_loss = torch.zeros_like(loss)
        for i, aux_net in enumerate(self.aux_nets):
            transition = (states, actions, rewards, next_states, terminals)
            aux_loss += aux_net.compute_loss(transition, phi, nphi)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/val_loss', loss.item(), self.total_steps)
            self.loss = loss.item()

            ### TRACKING LOSSES
            val_loss = self.loss
            aux_loss, attractive_loss, repulsive_loss = self.aux_nets[0].aux_loss, self.aux_nets[0].attractive_loss, \
                                                        self.aux_nets[0].repulsive_loss

            if len(self.val_loss) < self.queue_size:
                self.val_loss.append(val_loss)
                self.aux_loss.append(aux_loss)
                self.aux_attractive_loss.append(attractive_loss)
                self.aux_repulsive_loss.append(repulsive_loss)
            else:
                idx = self.total_steps % self.queue_size
                self.val_loss[idx] = val_loss
                self.aux_loss[idx] = aux_loss
                self.aux_attractive_loss[idx] = attractive_loss
                self.aux_repulsive_loss[idx] = repulsive_loss

        loss += aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
            for i, ant in enumerate(self.targets.aux_nets):
                ant.load_state_dict(self.aux_nets[i].state_dict())


class DQNAuxAgentGeneral(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)

        # Computing Loss for Value Function
        q = self.val_net(phi)[self.batch_indices, actions]
        nphi = self.targets.rep_net(next_states)
        q_next, action_next = self.targets.val_net(nphi).detach().max(1)
        terminals = torch_utils.tensor(terminals, self.cfg.device)
        rewards = torch_utils.tensor(rewards, self.cfg.device)
        target = self.cfg.discount * q_next * (1 - terminals).float()
        target.add_(rewards.float())
        loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()

        # Computing Loss for Aux Tasks
        aux_loss = torch.zeros_like(loss)
        for i, aux_net in enumerate(self.aux_nets):
            transition = (states, actions, rewards, next_states, terminals)
            sig = signature(aux_net.compute_loss)
            if len(sig.parameters) == 4:
                aux_loss += aux_net.compute_loss(transition, phi, nphi, action_next)
            else:
                aux_loss += aux_net.compute_loss(transition, phi, nphi)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/val_loss', loss.item(), self.total_steps)

        loss += aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
            for i, ant in enumerate(self.targets.aux_nets):
                ant.load_state_dict(self.aux_nets[i].state_dict())


class DQNSwitchHeadAgent(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.head = None
        self.policy_head = self.val_net

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
            self.head = self.env.info("use_head")
            if self.head==0:
                self.policy_head = self.val_net
            else:
                self.policy_head = self.aux_nets[self.head-1]

        action = self.policy(self.state, self.cfg.eps_schedule())

        next_state, reward, done, _ = self.env.step([action])
        if self.head > 0:
            reward = reward * (-1)
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        self.state = next_state
        # print('action: ', action)
        self.update_stats(reward, done)
        self.update()

    def policy(self, state, eps):
        with torch.no_grad():
            phi = self.rep_net(self.cfg.state_normalizer(state))
            q_values = self.policy_head.forward(phi)
        q_values = torch_utils.to_np(q_values).flatten()

        if np.random.rand() < eps:
            action = np.random.randint(0, len(q_values))
        else:
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        return action

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)

        # Computing Loss for Value Function
        q = self.val_net(phi)[self.batch_indices, actions]
        nphi = self.targets.rep_net(next_states)
        q_next, action_next = self.targets.val_net(nphi).detach().max(1)
        terminals = torch_utils.tensor(terminals, self.cfg.device)
        rewards = torch_utils.tensor(rewards, self.cfg.device)
        target = self.cfg.discount * q_next * (1 - terminals).float()
        target.add_(rewards.float())
        loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()

        # Computing Loss for Aux Tasks
        aux_loss = torch.zeros_like(loss)
        for i, aux_net in enumerate(self.aux_nets):
            transition = (states, actions, rewards, next_states, terminals)
            aux_loss += aux_net.compute_loss(transition, phi, nphi, action_next)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/val_loss', loss.item(), self.total_steps)

        loss += aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
            for i, ant in enumerate(self.targets.aux_nets):
                ant.load_state_dict(self.aux_nets[i].state_dict())

        if self.cfg.visualize_aux_distance and (self.total_steps-1) % self.cfg.eval_interval == 0:
            self.visualize_distance()



class DQNAuxSuccessorAgent(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)

        # Computing Loss for Value Function
        q = self.val_net(phi)[self.batch_indices, actions]
        nphi = self.targets.rep_net(next_states)
        q_next, action_next = self.targets.val_net(nphi).detach().max(1)
        terminals = torch_utils.tensor(terminals, self.cfg.device)
        rewards = torch_utils.tensor(rewards, self.cfg.device)
        target = self.cfg.discount * q_next * (1 - terminals).float()
        target.add_(rewards.float())
        loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()

        # Computing Loss for Aux Tasks
        aux_loss = torch.zeros_like(loss)
        for i, aux_net in enumerate(self.aux_nets):
            transition = (states, actions, rewards, next_states, terminals)
            aux_loss += aux_net.compute_loss(transition, phi, nphi, action_next)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/val_loss', loss.item(), self.total_steps)

        loss += aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
            for i, ant in enumerate(self.targets.aux_nets):
                ant.load_state_dict(self.aux_nets[i].state_dict())

        if self.cfg.visualize_aux_distance and (self.total_steps-1) % self.cfg.eval_interval == 0:
            self.visualize_distance()

    def visualize_distance(self):
        segment = self.env.get_visualization_segment()
        states, state_coords, goal_states, goal_coords = segment
        assert len(states) == len(state_coords)

        states = self.cfg.state_normalizer(torch_utils.tensor((states), self.cfg.device))
        goal_states = self.cfg.state_normalizer(torch_utils.tensor((goal_states), self.cfg.device))

        # a plot with rows = num of goal-states and cols = 1
        fig, ax = plt.subplots(nrows=len(goal_states), ncols=4, figsize=(6*4, 6 * len(goal_states)))
        max_x = max(state_coords, key=lambda xy: xy[0])[0]
        max_y = max(state_coords, key=lambda xy: xy[1])[1]

        for a in range(4):
            action = torch_utils.tensor(a, self.cfg.device).long()
            f_s = self.aux_nets[0].aux_predictor(self.rep_net(states), action)
            f_g = self.aux_nets[0].aux_predictor(self.rep_net(goal_states), action)

            def compute_distance_map(f_states, f_goal, size_x, size_y):
                l2_vec = ((f_states - f_goal)**2).sum(dim=1)
                distance_map = np.zeros((size_x, size_y))
                for k, xy_coord in enumerate(state_coords):
                    x, y = xy_coord
                    distance_map[x][y] = l2_vec[k].item()
                return distance_map

            for g_k in range(len(goal_states)):
                _distance_map = compute_distance_map(f_s, f_g[g_k], max_x+1, max_y+1)
                sns.heatmap(_distance_map, ax=ax[g_k][a])
                ax[g_k][a].set_title('Goal: {} | Action: {}'.format(goal_coords[g_k], a))

        viz_dir = self.cfg.get_visualization_dir()
        # viz_file = 'visualization_{}.png'.format(self.num_episodes)
        viz_file = 'aux_visualization_{}.png'.format(self.total_steps)
        plt.savefig(os.path.join(viz_dir, viz_file))
        plt.close()



class DQNAuxAgentKnowUsefulArea(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.useful_area = cfg.useful_area

    def get_aux_targets(self, x):
        s, t1, t2, sp, t3 = x
        s = s[:, :self.useful_area[0], :self.useful_area[1], :self.useful_area[2]]
        sp = sp[:, :self.useful_area[0], :self.useful_area[1], :self.useful_area[2]]
        x = [s, t1, t2, sp, t3]
        return [aux.get_aux_target(x) for aux in self.aux_nets]


class DQNAuxAgentLplcTrue(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trajectory = []
        self.traj_len = cfg.aux_traj_len
        self.traj_idx = 0
        self.sr = None

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

        self.update_traj(self.state, action, next_state, reward, done)
        self.feed_successor()

        self.state = next_state
        self.update_stats(reward, done)
        if self.traj_idx >= self.traj_len:
            self.update()

    def update_traj(self, state, action, next_state, reward, done):
        if self.traj_idx < self.traj_len:
            self.trajectory.append([state, action, next_state, reward, done])
            # self.a_traj.append(action)
        else:
            i = self.traj_idx % self.traj_len
            self.trajectory[i] = [state, action, next_state, reward, done]
            # self.a_traj[i] = action
        self.traj_idx += 1

    def feed_successor(self):
        if self.traj_idx > self.traj_len:
            add_to_bf = self.traj_idx % self.traj_len
            # update_sf = (self.traj_idx - 1) % self.traj_len
            # print(self.traj_idx%self.traj_len,
            #       self.sr.sum(),
            #       ((self.sr - self.cfg.state_normalizer(self.trajectory[add_to_bf][0]).flatten())/ self.cfg.aux_lmbda).sum(),
            #       (self.cfg.state_normalizer(self.trajectory[update_sf][0]).flatten()).sum())
            # self.sr = (self.sr - self.cfg.state_normalizer(self.trajectory[add_to_bf][0]).flatten()) / self.cfg.aux_lmbda + \
            #           self.cfg.aux_lmbda ** (self.traj_len - 2) * self.cfg.state_normalizer(self.trajectory[update_sf][0]).flatten()
            self.sr = self.cfg.state_normalizer(self.trajectory[(add_to_bf+1)%self.traj_len][0])#.flatten()
            for i in range(self.traj_len-1):
                self.sr += self.cfg.aux_lmbda**(add_to_bf+i) * self.cfg.state_normalizer(self.trajectory[(add_to_bf+1+i)%self.traj_len][0])#.flatten()

            state, action, next_state, reward, done = self.trajectory[add_to_bf]
            self.replay.feed([state, action, reward, next_state, int(done), self.sr * (1-self.cfg.aux_lmbda)])
        elif self.traj_idx == 1:
            # self.sr = np.zeros(np.prod(self.state.shape))
            self.sr = np.zeros(self.state.shape)
        elif 1 < self.traj_idx < self.traj_len:
            if self.traj_idx != 0:
                self.sr += self.cfg.aux_lmbda ** (self.traj_idx - 2) * self.cfg.state_normalizer(self.trajectory[self.traj_idx - 1][0])#.flatten()
        elif self.traj_idx == self.traj_len:
            self.sr += self.cfg.aux_lmbda ** (self.traj_idx - 2) * self.cfg.state_normalizer(self.trajectory[self.traj_idx - 1][0])#.flatten()
            state, action, next_state, reward, done = self.trajectory[0]
            self.replay.feed([state, action, reward, next_state, int(done), self.sr])

    def update(self):
        states, actions, rewards, next_states, terminals, srs = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)

        # Computing Loss for Value Function
        q = self.val_net(phi)[self.batch_indices, actions]
        nphi = self.targets.rep_net(next_states)
        q_next = self.targets.val_net(nphi).detach().max(1)[0]
        terminals = torch_utils.tensor(terminals, self.cfg.device)
        rewards = torch_utils.tensor(rewards, self.cfg.device)
        target = self.cfg.discount * q_next * (1 - terminals).float()
        target.add_(rewards.float())
        loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()

        # Computing Loss for Aux Tasks
        aux_loss = torch.zeros_like(loss)
        for i, aux_net in enumerate(self.aux_nets):
            transition = (states, actions, rewards, next_states, terminals, srs)
            aux_loss += aux_net.compute_loss(transition, phi, nphi)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/val_loss', loss.item(), self.total_steps)

        loss += aux_loss
        # loss = aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
            for i, ant in enumerate(self.targets.aux_nets):
                ant.load_state_dict(self.aux_nets[i].state_dict())


class DecoderRandomBaseline(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)
        # Computing Loss for Aux Tasks
        aux_net = self.aux_nets[0]
        transition = (states, actions, rewards, next_states, terminals)
        aux_loss = aux_net.compute_loss(transition, phi, None)

        loss = aux_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SuccessorRandomBaseline(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)
        with torch.no_grad():
            nphi = self.targets.rep_net(next_states)

        random_actions = torch.LongTensor(self.cfg.batch_size).random_(0, 4)


        transition = (states, actions, rewards, next_states, terminals)

        aux_loss = self.aux_nets[0].compute_loss(transition, phi, nphi, random_actions)
        aux_loss += self.aux_nets[1].compute_loss(transition, phi, nphi, random_actions)

        loss = aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())


class DQNAuxColorPredictorAgent(DQNAuxAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()

        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)
        actions = torch_utils.tensor(actions, self.cfg.device).long()

        phi = self.rep_net(states)

        # Computing Loss for Value Function
        q = self.val_net(phi)[self.batch_indices, actions]
        nphi = self.targets.rep_net(next_states)
        q_next = self.targets.val_net(nphi).detach().max(1)[0]
        terminals = torch_utils.tensor(terminals, self.cfg.device)
        rewards = torch_utils.tensor(rewards, self.cfg.device)
        target = self.cfg.discount * q_next * (1 - terminals).float()
        target.add_(rewards.float())
        loss = self.vf_loss(q, target)  # (q_next - q).pow(2).mul(0.5).mean()

        transition = (states, actions, rewards, next_states, terminals)
        # Computing Loss for Aux Tasks
        aux_loss = torch.zeros_like(loss)
        xy_aux, color_auxes = self.aux_nets[0], self.aux_nets[1:]

        aux_loss += xy_aux.compute_loss(transition, phi, nphi)

        aux_select = np.random.choice(np.arange(12), 1).item()
        aux_loss += color_auxes[aux_select].compute_loss(transition, phi, nphi)

        for aux in color_auxes:
            aux.total_steps += 1

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/val_loss', loss.item(), self.total_steps)

        loss += aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
            for i, ant in enumerate(self.targets.aux_nets):
                ant.load_state_dict(self.aux_nets[i].state_dict())

        if self.cfg.visualize_aux_distance and (self.total_steps - 1) % self.cfg.eval_interval == 0:
            self.visualize_distance()

    def visualize_distance(self):
        segment = self.env.get_visualization_segment()
        states, state_coords, goal_states, goal_coords = segment
        assert len(states) == len(state_coords)

        states = self.cfg.state_normalizer(torch_utils.tensor((states), self.cfg.device))
        goal_states = self.cfg.state_normalizer(torch_utils.tensor((goal_states), self.cfg.device))

        # a plot with rows = num of goal-states and cols = 1
        fig, ax = plt.subplots(nrows=len(goal_states), ncols=4, figsize=(6 * 4, 6 * len(goal_states)))
        max_x = max(state_coords, key=lambda xy: xy[0])[0]
        max_y = max(state_coords, key=lambda xy: xy[1])[1]

        for a in range(4):
            action = torch_utils.tensor(a, self.cfg.device).long()
            f_s = self.aux_nets[0].aux_predictor(self.rep_net(states), action)
            f_g = self.aux_nets[0].aux_predictor(self.rep_net(goal_states), action)

            def compute_distance_map(f_states, f_goal, size_x, size_y):
                l2_vec = ((f_states - f_goal) ** 2).sum(dim=1)
                distance_map = np.zeros((size_x, size_y))
                for k, xy_coord in enumerate(state_coords):
                    x, y = xy_coord
                    distance_map[x][y] = l2_vec[k].item()
                return distance_map

            for g_k in range(len(goal_states)):
                _distance_map = compute_distance_map(f_s, f_g[g_k], max_x + 1, max_y + 1)
                sns.heatmap(_distance_map, ax=ax[g_k][a])
                ax[g_k][a].set_title('Goal: {} | Action: {}'.format(goal_coords[g_k], a))

        viz_dir = self.cfg.get_visualization_dir()
        # viz_file = 'visualization_{}.png'.format(self.num_episodes)
        viz_file = 'aux_visualization_{}.png'.format(self.total_steps)
        plt.savefig(os.path.join(viz_dir, viz_file))
        plt.close()

