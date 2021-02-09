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


class TrainedDQNAgent(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

        rep_net = cfg.rep_fn()
        val_net = cfg.val_fn()

        if cfg.rep_config['load_params']:
            path = os.path.join(cfg.data_root, cfg.rep_config['path'])
            rep_net.load_state_dict(torch.load(path))

            path = os.path.join(cfg.data_root, cfg.val_fn_config['path'])
            val_net.load_state_dict(torch.load(path))

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
        self.replay = cfg.replay_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.trajectory = []
        self.actions = []
        self.rewards = []
        self.termins = []

        self.sampled_states = []
        self.sampled_next_states = []
        self.sampled_actions = []
        self.sampled_rewards = []
        self.sampled_termins = []
        self.sampled_different = []

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False

            self.sample_states()

        with torch.no_grad():
            phi = self.rep_net(self.cfg.state_normalizer(self.state))
            q_values = self.val_net(phi)

        q_values = torch_utils.to_np(q_values).flatten()
        if np.random.rand() < self.cfg.eps_schedule():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

        if self.trajectory != []:
            self.actions.append(action)
            self.rewards.append(self.r)
            self.termins.append(self.g)

        next_state, reward, done, _ = self.env.step([action])
        self.r = reward
        self.g = done

        self.trajectory.append(next_state)

        self.replay.feed([self.state, action, reward, next_state, int(done)])
        self.state = next_state

        self.update_stats(reward, done)
        self.update()

    def sample_states(self):
        length = len(self.trajectory)
        if not length:
            return

        # chunk_len = length//3
        # chunks = [self.trajectory[:chunk_len], self.trajectory[chunk_len:chunk_len*2], self.trajectory[chunk_len*2:]]
        # chunk = chunks[np.random.choice(range(len(chunks)))]
        # state = chunk[np.random.choice(range(len(chunk)))]
        # self.sampled_states.append(state)
        # self.trajectory = []
        chunk_len = length//3
        if length > 2:
            chunks = [self.trajectory[:chunk_len], self.trajectory[chunk_len:chunk_len*2], self.trajectory[chunk_len*2:]]
            ci = np.random.choice(range(len(chunks)))
            si = np.random.choice(range(len(chunks[ci])))
            si = min(ci * chunk_len + si, length-2)
            state = self.trajectory[si]
            action = self.actions[si]
            reward = self.rewards[si]
            termin = self.termins[si]
            next_s = self.trajectory[si+1]
            self.sampled_states.append(state)
            self.sampled_next_states.append(next_s)
            self.sampled_different.append(self.trajectory[np.random.randint(len(self.trajectory) - 1)]) # remove last state
            self.sampled_actions.append(action)
            self.sampled_rewards.append(reward)
            self.sampled_termins.append(termin)
            self.trajectory = []
            self.actions = []
            self.rewards = []
            self.termins = []
        # elif length == 2:
        #     state = self.trajectory[0]
        #     action = self.actions[0]
        #     reward = self.rewards[0]
        #     termin = self.termins[0]
        #     next_s = self.trajectory[1]
        #     self.sampled_states.append(state)
        #     self.sampled_next_states.append(next_s)
        #     self.sampled_actions.append(action)
        #     self.sampled_rewards.append(reward)
        #     self.sampled_termins.append(termin)
        #     self.trajectory = []
        #     self.actions = []
        #     self.rewards = []
        #     self.termins = []
        else:
            self.trajectory = []
            self.actions = []
            self.rewards = []
            self.termins = []


    def update(self):
        return

    def eval_step(self, state):
        if np.random.rand() < self.cfg.eps_schedule.read_only():
            return np.random.randint(0, self.cfg.action_dim)
        else:
            q_values = self.val_net(self.rep_net(self.cfg.state_normalizer(state)))
            q_values = torch_utils.to_np(q_values).flatten()
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

    def log_tensorboard(self):
        pass

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
        pass

    def visualize(self):
        pass

    def save(self):
        pass

    def load(self, parameters_dir):
        pass


class RandomAgent(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env = cfg.env_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.trajectory = []
        self.actions = []
        self.rewards = []
        self.termins = []

        self.sampled_states = []
        self.sampled_next_states = []
        self.sampled_actions = []
        self.sampled_rewards = []
        self.sampled_termins = []

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False

            self.sample_states()

        action = np.random.randint(0, self.env.action_dim)
        next_state, reward, done, _ = self.env.step([action])

        if self.trajectory != []:
            self.actions.append(action)
            self.rewards.append(reward)
            self.termins.append(done)
        self.trajectory.append(next_state)
        self.state = next_state

        self.update_stats(reward, done)
        self.update()

    def sample_states(self):
        length = len(self.trajectory)
        if not length:
            return

        chunk_len = length//3
        if length > 2:
            chunks = [self.trajectory[:chunk_len], self.trajectory[chunk_len:chunk_len*2], self.trajectory[chunk_len*2:]]
            ci = np.random.choice(range(len(chunks)))
            chunk = chunks[ci]
            si = np.random.choice(range(len(chunk)))
            si = min(ci * chunk_len + si, length-2)
            state = self.trajectory[si]
            action = self.actions[si]
            reward = self.rewards[si]
            termin = self.termins[si]
            next_s = self.trajectory[si+1]

            self.sampled_states.append(state)
            self.sampled_next_states.append(next_s)
            self.sampled_actions.append(action)
            self.sampled_rewards.append(reward)
            self.sampled_termins.append(termin)
            self.trajectory = []
            self.actions = []
            self.rewards = []
            self.termins = []

        elif length == 2:
            state = self.trajectory[0]
            action = self.actions[0]
            reward = self.rewards[0]
            termin = self.termins[0]
            next_s = self.trajectory[1]
            self.sampled_states.append(state)
            self.sampled_next_states.append(next_s)
            self.sampled_actions.append(action)
            self.sampled_rewards.append(reward)
            self.sampled_termins.append(termin)
            self.trajectory = []
            self.actions = []
            self.rewards = []
            self.termins = []
        else:
            self.trajectory = []
            self.actions = []
            self.rewards = []
            self.termins = []

    def update(self):
        return

    def eval_step(self, state):
        return np.random.randint(0, self.cfg.action_dim)

    def log_tensorboard(self):
        pass

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
        pass

    def visualize(self):
        pass

    def save(self):
        pass

    def load(self, parameters_dir):
        pass
