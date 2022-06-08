import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import seaborn as sns

from core.network import network_architectures
from core.utils import torch_utils


class AuxTask(nn.Module):
    def __init__(self, aux_predictor, cfg):
        super().__init__()
        self.aux_predictor = aux_predictor
        self.total_steps = 0
        self.cfg = cfg


class Rgb2Xy(AuxTask):
    def __init__(self, aux_predictor, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.aux_predictor(x)

    def compute_loss(self, transition, phi, nphi, place_holder=None):
        prediction = self.forward(phi)

        state, _, _, _, _ = transition
        # Un-normalizing and finding the xy coordinate of the agent
        input_state = (255.0/2.0) * (state + 1)
        # input_state = (np.argwhere(input_state[:, :, :, 2] == 255.0)[:, 1:])
        input_state = np.argwhere(np.logical_and(np.logical_and(input_state[:, :, :, 2] == 255.0,
                                                                input_state[:, :, :, 1] == 0.0),
                                                 input_state[:, :, :, 0] == 0.0))[:, 1:]
        input_state = 2 * (1. / 14.0) * input_state - 1

        target = torch_utils.tensor(input_state, phi.device)
        loss = self.loss(prediction, target)

        self.total_steps += 1
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/xy_loss', loss.item(), self.total_steps)

        return loss


class InputDecoder(Rgb2Xy):
    def __init__(self, aux_predictor, cfg):
        super().__init__(aux_predictor, cfg)

    def compute_loss(self, transition, phi, _):
        prediction = self.forward(phi)

        input_state, _, _, _, _ = transition
        target = torch_utils.tensor(input_state, phi.device)
        loss = self.loss(prediction, target)
        self.total_steps += 1
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/decoder_loss', loss.item(), self.total_steps)
            if self.cfg.visualize and self.total_steps % self.cfg.eval_interval == 0:
                self.visualize_input_decoder(prediction, target)

        return loss

    def visualize_input_decoder(self, prediction, target, num_samples=5):
        perm = torch.randperm(prediction.size(0))
        idx = perm[:num_samples]
        prediction, target = prediction[idx], target[idx]

        def invert(x):
            x = (x.detach().cpu().numpy() + 1) * 255 / 2
            # x = x.astype(np.uint8)
            return np.clip(x/255.0, 0.0, 1.0)

        fig, axs = plt.subplots(num_samples, 2)
        for k in range(num_samples):
            targ_ax = axs[k, 0]
            targ_ax.imshow(invert(target[k]))
            targ_ax.axis('off')
            pred_ax = axs[k, 1]
            pred_ax.imshow(invert(prediction[k]))
            pred_ax.axis('off')
            if k == 0:
                targ_ax.set_title('Target'.format(k))
                pred_ax.set_title('Prediction'.format(k))

        viz_dir = self.cfg.get_visualization_dir()
        plt.savefig(os.path.join(viz_dir, 'decoder.jpg'.format(self.total_steps)))
        plt.close()


class NASv1(AuxTask):
    def __init__(self, aux_predictor, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        s, a = x
        return self.aux_predictor(s, a)

    def compute_target(self, _, nphi):
        return nphi.detach()

    def compute_loss(self, transition, phi, nphi):
        _, action, _, _, _ = transition
        prediction = self.forward((phi, action))
        target = self.compute_target(phi, nphi)
        loss = self.loss(prediction, target)

        self.total_steps += 1
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/aux_loss', loss.item(), self.total_steps)

        return loss


class NASv1Delta(NASv1):
    def __init__(self, aux_predictor, cfg):
        super().__init__(aux_predictor, cfg)

    def compute_target(self, phi, nphi):
        return (nphi - phi).detach()


class NASv2(AuxTask):
    def __init__(self, aux_predictor, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        s, a = x
        return self.aux_predictor(s, a)

    def compute_target(self, phi, nphi):
        return nphi

    def compute_loss(self, transition, phi, _):
        _, action, _, next_states, _ = transition
        prediction = self.forward((phi, action))
        target = self.compute_target(phi, self.cfg.agent.rep_net(next_states))
        shift = len(prediction) // 2
        attractive = self.loss(prediction - target, torch.zeros_like(prediction))

        target = torch.cat([target[shift:], target[:shift]]) # TODO: I should look into this.
        distance = torch.sum((prediction - target) ** 2, dim=1)
        repulsive = torch.max(torch.zeros_like(distance), 1 - distance).mean()

        loss = attractive + repulsive

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/aux_loss', loss.item(), self.total_steps)
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/attractive_loss', attractive.item(), self.total_steps)
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/repuslive_loss', repulsive.item(), self.total_steps)

            self.aux_loss = loss.item()
            self.attractive_loss = attractive.item()
            self.repulsive_loss = repulsive.item()

        self.total_steps += 1

        return attractive + repulsive


class NASv2Delta(NASv2):
    def __init__(self, aux_predictor, cfg):
        super().__init__(aux_predictor, cfg)

    def compute_target(self, phi, nphi):
        return nphi - phi


class SuccessorFeaturesAS(AuxTask):
    def __init__(self, aux_predictor, aux_target_predictor, lmbda, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.MSELoss()
        self.aux_target_predictor = aux_target_predictor
        self.lmbda = lmbda

    def compute_loss(self, transition, phi, nphi, action_next):
        _, action, _, _, _ = transition

        prediction = self.aux_predictor(phi, action)
        with torch.no_grad():
            target = (1 - self.lmbda) * nphi + self.lmbda * self.aux_target_predictor(nphi, action_next) #TODO: Related works 

        loss = self.loss(prediction, target)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/successor_as_loss', loss.item(), self.total_steps)

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.aux_target_predictor.load_state_dict(self.aux_predictor.state_dict())

        self.total_steps += 1

        return loss


class SuccessorFeaturesOS(SuccessorFeaturesAS):
    def __init__(self, aux_predictor, aux_target_predictor, lmbda, cfg):
        super().__init__(aux_predictor, aux_target_predictor, lmbda, cfg)

    def compute_loss(self, transition, phi, nphi, action_next):
        _, action, _, next_state, _ = transition
        next_state = torch_utils.tensor(self.cfg.state_normalizer(next_state), self.cfg.device).flatten(start_dim=1)
        prediction = self.aux_predictor(phi, action)
        with torch.no_grad():
            target = next_state + self.lmbda * self.aux_target_predictor(nphi, action_next)

        loss = self.loss(prediction, target)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/successor_as_loss', loss.item(), self.total_steps)


class RewardPredictor(AuxTask):
    def __init__(self, aux_predictor, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.MSELoss()
        self.device = cfg.device

    def forward(self, x):
        s, a = x
        return self.aux_predictor(s, a)

    def compute_loss(self, transition, phi, nphi, actions_next=None):
        _, action, rewards, _, _ = transition
        prediction = self.forward((phi, action))
        rewards = torch_utils.tensor(torch.unsqueeze(rewards, 1), self.device)
        loss = self.loss(prediction, rewards)

        self.total_steps += 1
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/reward_loss', loss.item(), self.total_steps)

        return loss


class AuxControl(AuxTask):
    def __init__(self, aux_predictor, aux_target_predictor, goal_id, discount, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.MSELoss()
        self.aux_target_predictor = aux_target_predictor
        self.batch_indices = torch.arange(self.cfg.batch_size).long().to(cfg.device)
        self.goal_id = goal_id
        self.env = cfg.env_fn()
        self.goals = self.env.goals
        self.discount = discount
        # self.goals = [[9, 9], [0, 0], [0, 14], [14, 0], [14, 14], [7, 7]]
        self.goal = np.array(self.goals[self.goal_id])

    def compute_rewards_dones(self, state):
        # Un-normalizing and finding the xy coordinate of the agent
        state = (255.0/2.0) * (state + 1)
        state = (np.argwhere(state[:, :, :, 2] == 255.0)[:, 1:])

        terminals = np.all(state == self.goal, axis=1)

        rewards = terminals.astype(np.float64)
        terminals = terminals.astype(np.int64)
        return rewards, terminals

    def compute_loss(self, transition, phi, nphi, action_next):
        states, actions, _, next_states, _ = transition

        aux_rewards, aux_terminals = self.compute_rewards_dones(next_states)
        # Computing Loss for Aux Value Function
        with torch.no_grad():
            q_next = self.aux_target_predictor(nphi).detach().max(1)[0]
            aux_terminals = torch_utils.tensor(aux_terminals, self.cfg.device)
            aux_rewards = torch_utils.tensor(aux_rewards, self.cfg.device)
            target = self.discount * q_next * (1 - aux_terminals).float()
            target.add_(aux_rewards.float())
        q = self.aux_predictor(phi)[self.batch_indices, actions]
        loss = self.loss(q, target)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/aux_control_{}_loss'.format(self.goal_id),
                                                          loss.item(), self.total_steps)

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.aux_target_predictor.load_state_dict(self.aux_predictor.state_dict())

        if self.cfg.visualize and (self.total_steps+1) % self.cfg.eval_interval == 0:
            self.visualize()

        self.total_steps += 1

        return loss

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
            assert len(states) == len(state_coords)

            states = self.cfg.state_normalizer(states)
            values = self.aux_predictor(self.cfg.agent.rep_net(states)).max(dim=1)[0]
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
            viz_file = 'visualization_aux_{}_{}.png'.format(self.goal_id, self.total_steps)
            plt.savefig(os.path.join(viz_dir, viz_file))
            plt.close()
        except NotImplementedError:
            return


class AuxControlCollect(AuxTask):
    def __init__(self, aux_predictor, aux_target_predictor, flip_reward, discount, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.MSELoss()
        self.aux_target_predictor = aux_target_predictor
        self.batch_indices = torch.arange(self.cfg.batch_size).long().to(cfg.device)
        self.flip_reward = flip_reward
        self.env = cfg.env_fn()
        self.discount = discount

        def flip(x):
            if x >= 0.999: return -1.001
            elif x <= -1.001: return 0.999
            else: return x

        self.flip = np.vectorize(flip)

    # def forward(self, phi):
    #     q = self.aux_predictor(phi)
    #     return q

    def compute_loss(self, transition, phi, nphi, next_action):
        states, actions, rewards, next_states, terminals = transition

        if self.flip_reward:
            aux_rewards = self.flip(rewards.cpu())
        else:
            aux_rewards = rewards
        aux_terminals = terminals

        # Computing Loss for Aux Value Function
        with torch.no_grad():
            q_next = self.aux_target_predictor(nphi).detach().max(1)[0]
            aux_terminals = torch_utils.tensor(aux_terminals, self.cfg.device)
            aux_rewards = torch_utils.tensor(aux_rewards, self.cfg.device)
            target = self.discount * q_next * (1 - aux_terminals).float()
            target.add_(aux_rewards.float())
        q = self.aux_predictor(phi)[self.batch_indices, actions]
        loss = self.loss(q, target)

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/aux_control_loss',
                                                          loss.item(), self.total_steps)

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.aux_target_predictor.load_state_dict(self.aux_predictor.state_dict())

        self.total_steps += 1

        return loss


class ColorPredictor(AuxTask):
    def __init__(self, aux_predictor, coord, id, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.CrossEntropyLoss()
        self.coord = coord
        self.target_color = np.zeros((cfg.batch_size,)) # 4 colors: red, green, blue, gray
        self.red, self.green, self.blue, self.gray = np.array([255., 0., 0.]), np.array([0., 255., 0.]), \
                                                     np.array([0., 0., 255.]), np.array([128., 128., 128.])
        self.id = id


    def compute_loss(self, transition, phi, nphi, place_holder=None):
        prediction = self.aux_predictor(phi)
        state, _, _, _, _ = transition
        x, y = self.coord
        input_state = (255.0/2.0) * (state + 1)
        color = input_state[:, x, y]
        self.target_color[np.all(color == self.red, axis=1)] = 0
        self.target_color[np.all(color == self.green, axis=1)] = 1
        self.target_color[np.all(color == self.blue, axis=1)] = 2
        self.target_color[np.all(color == self.gray, axis=1)] = 3

        target = torch_utils.tensor(self.target_color, phi.device).long()
        ls = self.loss(prediction, target)

        # self.total_steps += 1 # moved total_steps += 1 in the aux_agent code
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/color_predictor_{}'.format(self.id), ls.item(), self.total_steps)

        self.total_steps += 1

        return ls


class CountBlocks(AuxTask):
    def __init__(self, aux_predictor, color, cfg):
        super().__init__(aux_predictor, cfg)
        self.loss = torch.nn.MSELoss()
        self.target_color = np.zeros((cfg.batch_size,)) # 4 colors: red, green, blue, gray
        red, green = np.array([255., 0., 0.]), np.array([0., 255., 0.])
        if color == "red":
            self.color_coord = red
        elif color == "green":
            self.color_coord = green
        else:
            raise NotImplementedError

        self.color = color

    def compute_loss(self, transition, phi, nphi, place_holder=None):
        prediction = self.aux_predictor(phi)
        state, _, _, _, _ = transition

        input_state = (255.0/2.0) * (state + 1)

        count_color = np.zeros(self.cfg.batch_size)
        r, g, b = self.color_coord
        where_color = np.argwhere(np.logical_and(np.logical_and(input_state[:, :, :, 0] == r, input_state[:, :, :, 1] == g), input_state[:, :, :, 2] == b))
        for k in range(len(where_color)):
            count_color[where_color[k][0]] += 1

        target = torch_utils.tensor(count_color, phi.device)
        loss = self.loss(prediction.squeeze(1), target)

        self.total_steps += 1
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/count_blocks_{}'.format(self.color), loss.item(), self.total_steps)

        return loss


class Orthogonal(AuxTask):
    def __init__(self, aux_predictor, cfg, weight):
        super().__init__(aux_predictor, cfg)
        self.loss = self.orthogonal_loss
        self.batch_indices = torch.arange(self.cfg.batch_size).long().to(cfg.device)
        self.env = cfg.env_fn()
        self.weight = weight

    def orthogonal_loss(self, p1, p2, weight):
        l = (p1 * p2).sum(dim=1).mean() - torch.norm(p1, dim=1).mean() - torch.norm(p2, dim=1).mean()
        l = weight * l
        return l

    def compute_loss(self, transition, phi, nphi, action):
        random_idx = torch.randperm(phi.size()[0]).to(phi.device)
        phi2 = phi[random_idx]
        loss = self.loss(phi, phi2, self.weight)
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/aux/orthogonal', loss.item(), self.total_steps)
        return loss










class AuxFactory:
    @classmethod
    def get_aux_task(cls, cfg):
        aux_tasks = []
        aux_weights = []
        for aux_config in cfg.aux_config:
            if "aux_weight" not in aux_config.keys():
                aux_weights.append(1)
            else:
                aux_weights.append(aux_config["aux_weight"])

            # Creating aux_predictor (a network to predict aux targets)
            if aux_config['aux_task'] in ['nas_v1', 'nas_v1_delta', 'nas_v2', 'nas_v2_delta', 'reward_predictor']:
                # Aux tasks where the prediction depends on state and action
                if aux_config['aux_fn_type'] == 'linear':
                    aux_predictor = network_architectures.LinearActionNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                    aux_config['aux_out_dim'], cfg.action_dim)
                elif aux_config['aux_fn_type'] == 'fc':
                    aux_predictor = network_architectures.FCActionNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                aux_config['hidden_units'], aux_config['aux_out_dim'],
                                                                cfg.action_dim)
                else:
                    raise NotImplementedError
            elif aux_config['aux_task'] in ['successor_as', 'successor_os']:
                # Aux tasks where the prediction depends on state and action
                if aux_config['aux_fn_type'] == 'linear':
                    aux_predictor = network_architectures.LinearActionNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                    aux_config['aux_out_dim'], cfg.action_dim)
                    aux_target_predictor = network_architectures.LinearActionNetwork(cfg.device,
                                                                              np.prod(cfg.rep_fn().output_dim),
                                                                              aux_config['aux_out_dim'], cfg.action_dim)
                elif aux_config['aux_fn_type'] == 'fc':
                    aux_predictor = network_architectures.FCActionNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                aux_config['hidden_units'], aux_config['aux_out_dim'],
                                                                cfg.action_dim)
                    aux_target_predictor = network_architectures.FCActionNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                aux_config['hidden_units'], aux_config['aux_out_dim'],
                                                                cfg.action_dim)
                else:
                    raise NotImplementedError
            elif aux_config['aux_task'] in ['aux_control', 'aux_control_collect']:
                if aux_config['aux_fn_type'] == 'linear':
                    aux_predictor = network_architectures.LinearNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                    aux_config['aux_out_dim'])
                    aux_target_predictor = network_architectures.LinearNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                    aux_config['aux_out_dim'])
                elif aux_config['aux_fn_type'] == 'fc':
                    aux_predictor = network_architectures.FCNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                aux_config['hidden_units'], aux_config['aux_out_dim'])
                    aux_target_predictor = network_architectures.FCNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                aux_config['hidden_units'], aux_config['aux_out_dim'])
                else:
                    raise NotImplementedError
            elif aux_config['aux_task'] in ['color_predictor']:
                if aux_config['aux_fn_type'] == 'linear':
                    raise NotImplementedError
                elif aux_config['aux_fn_type'] == 'fc':
                    aux_predictor = network_architectures.FCNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                    aux_config['hidden_units'], aux_config['aux_out_dim'],
                                                                    head_activation=nn.Softmax(1).to(cfg.device))
                else:
                    raise NotImplementedError
            else:
                # Aux tasks where the prediction depends only on state (and not on action)
                if aux_config['aux_fn_type'] == 'linear':
                    aux_predictor = network_architectures.LinearNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                        aux_config['aux_out_dim'])
                elif aux_config['aux_fn_type'] == 'fc':
                    aux_predictor = network_architectures.FCNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                                aux_config['hidden_units'], aux_config['aux_out_dim'])
                elif aux_config['aux_fn_type'] == 'deconv':
                    aux_predictor = network_architectures.DeConvNetwork(cfg.device, aux_config['aux_in_dim'],
                                                                        aux_config['aux_out_dim'],
                                                                        aux_config['deconv_architecture'],
                                                                        cfg.rep_config['in_dim'],  # state-dim
                                                                        cfg.rep_config["conv_architecture"])
                elif aux_config['aux_fn_type'] == 'none':
                    aux_predictor = None
                else:
                    raise NotImplementedError

            # Creating the auxiliary task
            if aux_config['aux_task'] == 'nas_v1':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor: NASv1(aux_predictor=aux_predictor, cfg=cfg))
            elif aux_config['aux_task'] == 'nas_v1_delta':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor: NASv1Delta(aux_predictor=aux_predictor, cfg=cfg))
            elif aux_config['aux_task'] == 'nas_v2':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor: NASv2(aux_predictor=aux_predictor, cfg=cfg))
            elif aux_config['aux_task'] == 'nas_v2_delta':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor: NASv2Delta(aux_predictor=aux_predictor, cfg=cfg))
            elif aux_config['aux_task'] == 'rgb2xy':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor: Rgb2Xy(aux_predictor=aux_predictor, cfg=cfg))
            elif aux_config['aux_task'] == 'input_decoder':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor: InputDecoder(aux_predictor=aux_predictor, cfg=cfg))
            elif aux_config['aux_task'] == 'successor_as':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor, aux_target_predictor=aux_target_predictor,
                                        aux_config=aux_config: SuccessorFeaturesAS(aux_predictor=aux_predictor,
                                                                                   aux_target_predictor=aux_target_predictor,
                                                                                   lmbda=aux_config['successor_lmbda'], cfg=cfg))
            elif aux_config['aux_task'] == 'successor_os':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor, aux_target_predictor=aux_target_predictor,
                                        aux_config=aux_config: SuccessorFeaturesAS(aux_predictor=aux_predictor,
                                                                                   aux_target_predictor=aux_target_predictor,
                                                                                   lmbda=aux_config['successor_lmbda'], cfg=cfg))
                # aux_tasks.append(lambda cfg: SuccessorFeaturesOS(aux_predictor=aux_predictor,
                #                                                  aux_target_predictor=aux_target_predictor,
                #                                                  lmbda=aux_config['successor_lmbda'], cfg=cfg))
            elif aux_config['aux_task'] == 'reward_predictor':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor: RewardPredictor(aux_predictor=aux_predictor, cfg=cfg))
            elif aux_config['aux_task'] == 'aux_control':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor, aux_target_predictor=aux_target_predictor,
                                        aux_config=aux_config: AuxControl(aux_predictor=aux_predictor,
                                                               aux_target_predictor=aux_target_predictor,
                                                               goal_id=aux_config['goal_id'],
                                                               discount=aux_config['discount'], cfg=cfg))
            elif aux_config['aux_task'] == 'aux_control_collect':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor, aux_target_predictor=aux_target_predictor,
                                        aux_config=aux_config: AuxControlCollect(aux_predictor=aux_predictor,
                                                               aux_target_predictor=aux_target_predictor,
                                                               flip_reward=aux_config['flip_reward'],
                                                               discount=aux_config['discount'], cfg=cfg))
            elif aux_config['aux_task'] == 'color_predictor':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor, aux_config=aux_config:
                                 ColorPredictor(aux_predictor=aux_predictor, coord=aux_config["coord"],
                                                id=aux_config["id"], cfg=cfg))
            elif aux_config['aux_task'] == 'count_blocks':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor, aux_config=aux_config:
                                 CountBlocks(aux_predictor=aux_predictor,
                                                color=aux_config["color"], cfg=cfg))
            elif aux_config['aux_task'] == 'orthogonal':
                aux_tasks.append(lambda cfg, aux_predictor=aux_predictor, aux_config=aux_config:
                                 Orthogonal(aux_predictor=aux_predictor, cfg=cfg, weight=aux_config["aux_weight"]))
            else:
                raise NotImplementedError

        return aux_tasks, aux_weights
