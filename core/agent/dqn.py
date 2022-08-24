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
from core.utils.data_augs import random_shift


class DQNAgent(base.Agent):
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

        self.vf_loss = cfg.vf_loss_fn()
        self.vf_constr = cfg.vf_constr_fn()
        self.replay = cfg.replay_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.random_shift_prob = cfg.random_shift_prob
        self.random_shift_pad = cfg.random_shift_pad

        self.ortho_loss_weight = cfg.ortho_loss_weight

        if self.cfg.evaluate_interference:
            self.ac_last_sample = None
            self.ac_last_td2 = None
            self.update_interfs = []
            self.itera_interfs = []

        self.device = cfg.device

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            
            if self.record_video:
                self.image_array.append(self.state)

            self.reset = False
         
        # with torch.no_grad():
        #     phi = self.rep_net(self.cfg.state_normalizer(self.state))
        #     q_values = self.val_net(phi)
        #
        # q_values = torch_utils.to_np(q_values).flatten()
        # if np.random.rand() < self.cfg.eps_schedule():
        #     action = np.random.randint(0, len(q_values))
        # else:
        #     action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

        action = self.policy(self.state, self.cfg.eps_schedule())

        next_state, reward, done, _ = self.env.step([action])
#        print('state: ', next_state.shape)
#        print('action: ', action)
#        raise ValueError('Here')
#         print(self.ep_steps)
#         print(self.timeout)
#         if done or self.ep_steps+1 == self.timeout:
#             print(self.ep_steps)
#             print(done)
        self.replay.feed([self.state, action, reward, next_state, int(done)], (bool(done) or self.ep_steps+1 == self.timeout))
        self.state = next_state
        # print('action: ', action)
        self.update_stats(reward, done)
        if self.cfg.update_network:
            self.update()
        return self.env.prev_state, action, done

    def policy(self, state, eps):
        
        with torch.no_grad():
            # print(np.array(state).shape)
            # state = torch_utils.tensor(state, self.cfg.device)
            phi = self.rep_net(self.cfg.state_normalizer(state))
            q_values = self.val_net(phi)

        q_values = torch_utils.to_np(q_values).flatten()

        if np.random.rand() < eps:
            action = np.random.randint(0, len(q_values))
        else:
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        return action

    def update(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()
#        print('states shape: ', states.shape)
#        raise ValueError('No')
        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)

        actions = torch_utils.tensor(actions, self.cfg.device).long()
        if self.random_shift_prob > 0.:

            states = random_shift(
                imgs=states,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )

            next_states = random_shift(
                imgs=next_states,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )
        
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

        ######### orthognality loss ##########
        if self.ortho_loss_weight > 0:
        #     s = states
        #     reps = phi
        #     # with torch.no_grad():
        #     #     reps = agent.rep_net(s)
        #     # Vincent's thesis
        #     # reps = reps.detach().numpy()
        #     reps_duplicate = reps.clone().detach()
        #     dot_prod = torch.mm(reps, reps_duplicate.T)
        #     norm = torch.linalg.norm(reps_duplicate, dim=1).reshape((-1, 1))
        #     norm_prod = torch.mm(norm, norm.T)
        #     if len(torch.where(norm_prod==0)[0]) != 0:
        #         norm_prod[torch.where(norm_prod==0)] += 1e-05

        #     normalized = torch.abs(torch.div(dot_prod, norm_prod))
        #     normalized[torch.arange(normalized.shape[0]), torch.arange(normalized.shape[1])] = 0
        #     rho = normalized.sum() / (normalized.shape[0] * (normalized.shape[0]-1))

        #     loss += self.ortho_loss_weight*rho
        # elif self.ortho_loss_weight < 0:
            s = states
            reps = phi
            # with torch.no_grad():
            #     reps = agent.rep_net(s)
            # Vincent's thesis
            # reps = reps.detach().numpy()
            #reps_duplicate = reps.clone().detach()
            dot_prod = torch.mm(reps, reps.T).pow(2)
            dot_prod[torch.arange(dot_prod.shape[0]), torch.arange(dot_prod.shape[1])] = torch.sqrt(torch.diag(dot_prod)) * -1
 
            loss += self.ortho_loss_weight * dot_prod.sum()
        
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

    def log_tensorboard(self):
        rewards = self.ep_returns_queue
        mean, median, min, max = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)
        self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/average_reward', mean, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/median_reward', median, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/min_reward', min, self.total_steps)
        self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/max_reward', max, self.total_steps)


    def log_values(self):
        
        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        
        total_episodes = len(self.episode_rewards)
        for fruit_color in ['red', 'green']:
            for i in range(self.env.fruit_num):
                remove_fruit_list = powerset(list(range(self.env.fruit_num)))
                for rm_fruits in remove_fruit_list:
                    if (i in rm_fruits and self.env.rewarding_color == fruit_color):
                        continue
                    state_action_list = self.env.get_fruit_nearby_states(i, fruit_color, rm_fruits)
                    state_vals = 0
                    to_fruit_vals = 0
                    other_dir_vals_mean = 0
                    other_dir_vals_max = 0
                    other_dir_vals_min = 0
                    for state, fruit_action in state_action_list:
                        with torch.no_grad():
                            phi = self.rep_net(self.cfg.state_normalizer(state))
                            q_values = self.val_net(phi)
                        q_values = torch_utils.to_np(q_values).flatten()
                        policy = 0.1/(len(self.env.actions)) * np.ones(len(self.env.actions))
                        policy[q_values.argmax()] += 0.9
                        state_vals += policy@q_values
                        to_fruit_vals += q_values[fruit_action]
                        other_dir_vals = q_values[np.arange(len(self.env.actions))!=fruit_action]
                        other_dir_vals_mean += np.mean(other_dir_vals)
                        other_dir_vals_max +=  np.max(other_dir_vals)
                        other_dir_vals_min +=  np.min(other_dir_vals)

                    log_str = 'Fruit-directed action-values (%s, %d) removed_fruits %s LOG: steps %d, episodes %3d, ' \
                                'values %.10f (mean)'

                    self.cfg.logger.info(log_str % (fruit_color, i, str(rm_fruits), self.total_steps, total_episodes, to_fruit_vals/len(self.env.actions)))

                    log_str = 'Fruit-undirected action-values (%s, %d) removed_fruits %s LOG: steps %d, episodes %3d, ' \
                                'values %.10f/%.10f/%.10f (mean/min/max)'
                    
                    self.cfg.logger.info(log_str % (fruit_color, i, str(rm_fruits), self.total_steps, total_episodes, other_dir_vals_mean/len(self.env.actions),
                                                other_dir_vals_min/len(self.env.actions), other_dir_vals_max/len(self.env.actions)))
                    
                    log_str = 'Fruit state-values (%s, %d) removed_fruits %s LOG: steps %d, episodes %3d, ' \
                                'values %.10f (mean)'

                    self.cfg.logger.info(log_str % (fruit_color, i, str(rm_fruits), self.total_steps, total_episodes, state_vals/len(self.env.actions)))

                
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

    def update_interference(self, calc_accuracy_change=True):
        def td_square(rep_net, val_net, rep_target, val_target, samples):
            states, actions, next_states, rewards, terminals = samples
            with torch.no_grad():
                q = val_net(rep_net(states))[np.array(range(len(actions))), actions[:, 0]]
                q_next = val_target(rep_target(next_states))
                q_next = q_next.max(1)[0]
                terminals = torch_utils.tensor(terminals, self.cfg.device)
                rewards = torch_utils.tensor(rewards, self.cfg.device)
                target = self.cfg.discount * q_next * (1 - terminals).float()
                target.add_(rewards.float())

                q = torch_utils.to_np(q)
                target = torch_utils.to_np(target)
            return (target - q)**2

        def accuracy_change():
            delta2 = td_square(self.rep_net,
                                  self.val_net,
                                  self.targets.rep_net,
                                  self.targets.val_net,
                                  self.ac_last_sample)
            
            # print('delta2: ', delta2)
            # print('self.ac_last_td2: ', self.ac_last_td2)

            return delta2 - self.ac_last_td2

        # print('calc_accuracy_change: ', calc_accuracy_change)
        # print(self.ac_last_sample is not None and calc_accuracy_change)
        if self.ac_last_sample is not None and calc_accuracy_change:
            ac = accuracy_change()
            ui = np.clip(ac.mean(), 0, np.inf) # average over samples
            self.update_interfs.append(ui)
 
        states, actions, next_states, rewards, terminals = self.cfg.eval_dataset.sample()
        states = self.cfg.state_normalizer(states)
        next_s = self.cfg.state_normalizer(next_states)
        actions = actions.reshape([-1, 1])
        self.ac_last_sample = states, actions, next_s, rewards, terminals
        self.ac_last_td2 = td_square(self.rep_net,
                                    self.val_net,
                                    self.targets.rep_net,
                                    self.targets.val_net,
                                    self.ac_last_sample)
        # if not calc_accuracy_change:
        #     print(self.ac_last_td2)

        # print('self.ac_last_td2: ', self.ac_last_td2)

    def iteration_interference(self):
        if len(self.update_interfs) > 0:
            self.itera_interfs.append(np.array(self.update_interfs).mean())
        self.update_interfs = []

    def log_interference(self, label=None, empty_interf=True):
        if len(self.itera_interfs) > 0:
            itera_interfs = np.array(self.itera_interfs)
            pct = np.percentile(itera_interfs, 90)
            target_idx = np.where(itera_interfs >= pct)[0]
            itf = np.mean(itera_interfs[target_idx])
        else:
            itf = np.nan
        if empty_interf:
            self.itera_interfs = []

        if label is None:
            log_str = 'total steps %d, total episodes %3d, ' \
                      'Interference: %.8f/'
            self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), itf))
        else:
            log_str = 'total steps %d, total episodes %3d, ' \
                      '%s Interference: %.8f/'
            self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), label, itf))

        return itf

    def visualize_vf(self):
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
            # viz_file = 'visualization_{}.png'.format(self.total_steps)
            #viz_file = 'visualization.png'
            plt.savefig(os.path.join(viz_dir, viz_file))
            plt.close()

        except NotImplementedError:
            return

    def visualize(self):
        # a plot with rows = num of goal-states and cols = 1
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

        frame = self.state.astype(np.uint8)
        figure, ax = plt.subplots()
        ax.imshow(frame)
        plt.axis('off')

        viz_dir = self.cfg.get_visualization_dir()
        # viz_file = 'visualization_{}.png'.format(self.num_episodes)
        viz_file = 'visualization_{}.png'.format(self.total_steps)
        # viz_file = 'visualization.png'
        plt.savefig(os.path.join(viz_dir, viz_file))
        plt.close()

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

    # def eval_step(self, state):
    #     # if np.random.rand() < self.cfg.eps_schedule.read_only():
    #     #     return np.random.randint(0, self.cfg.action_dim)
    #     # else:
    #     #     phi = self.rep_net(self.cfg.state_normalizer(state))
    #     #     q_values, _, _, _ = self.val_net(np.expand_dims(phi, axis=0))
    #     #     q_values = torch_utils.to_np(q_values).flatten()
    #     #     return np.random.choice(np.flatnonzero(q_values == q_values.max()))
    #     phi = self.rep_net(self.cfg.state_normalizer(state))
    #     q_values, _, _, _ = self.val_net(np.expand_dims(phi, axis=0))
    #     q_values = torch_utils.to_np(q_values).flatten()
    #     return np.random.choice(np.flatnonzero(q_values == q_values.max()))

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
