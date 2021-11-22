import torch
import copy

import numpy as np

from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
from inspect import signature

from core.agent import dqn
from core.utils import torch_utils
from core.utils.lipschitz import compute_dynamics_awareness, compute_decorrelation
from core.utils.data_augs import random_shift
from core.utils.torch_utils import valid_from_done, update_state_dict 
from core.network.ul_networks import UlEncoderModel
from core.network.ul_networks import ContrastModel

IGNORE_INDEX = -100
class DQNAuxCLAgent(dqn.DQNAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ul_encoder = UlEncoderModel(
            conv=self.rep_net.net.conv_body, # is using conv body for the behaviour network or the target network?
            latent_size=self.cfg.ul_latent_size,
            conv_out_size=self.rep_net.net.conv_body.feature_dim,
            device=cfg.device
        )
        self.ul_target_encoder = copy.deepcopy(self.ul_encoder)
        self.ul_contrast = ContrastModel(
            latent_size=cfg.ul_latent_size,
            anchor_hidden_sizes=cfg.ul_anchor_hidden_sizes,
            device=cfg.device
        )

        params = list(self.ul_encoder.parameters()) + list(self.ul_contrast.parameters())
        self.ul_optimizer = self.cfg.ul_optimizer_fn(params)
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
       
       # Creating Target Networks
        TargetNets = namedtuple('TargetNets', ['rep_net', 'val_net', 'ul_net'])
        targets = TargetNets(rep_net=self.targets.rep_net, val_net=self.targets.val_net, ul_net=self.ul_target_encoder)
        self.targets = targets

        self.ul_delta_T = cfg.ul_delta_T
        self.ul_random_shift_pad = cfg.ul_random_shift_pad
        self.ul_random_shift_prob = cfg.ul_random_shift_prob
        
        # TODO: Remove this patchwork
        self.cfg.agent = self
        self.ul_weight = cfg.ul_weight
        self.ul_clip_grad_norm  = None
            
        self.ul_target_update_interval = cfg.ul_target_update_interval
        self.ul_target_update_tau = cfg.ul_target_update_tau
 
    def update(self):
        #print('Calculating RL loss')
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
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())

        ########### Computing the contrastive loss ###########
        # Currently it is implemented only with a single environment in mind and one repetition
        if self.replay.num_in_buffer > self.replay.replay_T + 2:
            #print('Calculating CL loss')
            self.ul_optimizer.zero_grad()
            states, actions, rewards, next_states, terminals = self.replay.sample_ul()
            #print(terminals)
            states = self.cfg.state_normalizer(states)
            anchor = states[:-self.ul_delta_T]
            positive = states[self.ul_delta_T:]

            #print('anchor: ', anchor.shape)        
            #print('positive: ', positive.shape)
            #print(self.ul_random_shift_prob)
            #print(self.ul_random_shift_pad)
            if self.ul_random_shift_prob > 0.:

                anchor = random_shift(
                    imgs=anchor,
                    pad=self.ul_random_shift_pad,
                    prob=self.ul_random_shift_prob,
                )

                positive = random_shift(
                    imgs=positive,
                    pad=self.ul_random_shift_pad,
                    prob=self.ul_random_shift_prob,
                )
        #anchor, positive = buffer_to((anchor, positive),
        #    device=self.agent.device)
            with torch.no_grad():
                c_positive = self.ul_target_encoder(positive)
            c_anchor = self.ul_encoder(anchor)
            logits = self.ul_contrast(c_anchor, c_positive)  # anchor mlp in here.

            labels = torch.arange(c_anchor.shape[0],
                dtype=torch.long, device=self.device)
            terminals = torch_utils.tensor(terminals, self.device)
            valid = valid_from_done(terminals).type(torch.bool)  # use all
            valid = valid[self.ul_delta_T:].reshape(-1)  # at positions of positive
            labels[~valid] = IGNORE_INDEX

            ul_loss = self.ul_weight * self.c_e_loss(logits, labels)
            ul_loss.backward()
            if self.ul_clip_grad_norm is None:
                grad_norm = 0.
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.ul_parameters(), self.ul_clip_grad_norm)
            self.ul_optimizer.step()

            correct = torch.argmax(logits.detach(), dim=1) == labels
            accuracy = torch.mean(correct[valid].float())
            
            if self.total_steps % self.ul_target_update_interval == 0:
                update_state_dict(self.ul_target_encoder, self.ul_encoder.state_dict(),
                    self.ul_target_update_tau)
        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn_aux/loss/val_loss', loss.item(), self.total_steps)

#        return ul_loss, accuracy, grad_norm

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

