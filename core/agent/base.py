import numpy as np
import torch


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.reset = True
        self.ep_steps = 0
        self.num_episodes = 0
        self.timeout = cfg.timeout
        self.stats_counter = 0
        self.batch_indices = torch.arange(self.cfg.batch_size).long().to(cfg.device)
        self.eval_env = cfg.env_fn()
        self.ep_returns_queue = np.zeros(cfg.stats_queue_size)

    def update_stats(self, reward, done):
        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.num_episodes += 1
            if self.cfg.evaluation_criteria == "return":
                self.add_episode_return(self.episode_reward)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_episode_return(self.ep_steps)
            else:
                raise NotImplementedError
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

    def add_episode_return(self, ep_return):
        self.ep_returns_queue[self.stats_counter] = ep_return
        self.stats_counter += 1
        self.stats_counter = self.stats_counter % self.cfg.stats_queue_size

    def populate_returns(self):
        for ep in range(self.cfg.stats_queue_size):
            ep_return, steps = self.eval_episode()
            if self.cfg.evaluation_criteria == "return":
                self.add_episode_return(ep_return)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_episode_return(steps)
            else:
                raise NotImplementedError

    def eval_episode(self):
        state = self.eval_env.reset()
        total_rewards = 0
        ep_steps = 0
        while True:
            action = self.eval_step(state)
            state, reward, done, _ = self.eval_env.step([action])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.cfg.timeout:
                break
        return total_rewards, ep_steps

    def eval_episodes(self):
        return

    def eval_step(self, state):
        raise NotImplementedError

    # def save(self, filename):
    #     raise NotImplementedError
    #
    # def load(self, filename):
    #     raise NotImplementedError

    def one_hot_action(self, actions):
        one_hot = np.zeros((len(actions), self.cfg.action_dim))
        np.put_along_axis(one_hot, actions.reshape((-1, 1)), 1, axis=1)
        return one_hot
