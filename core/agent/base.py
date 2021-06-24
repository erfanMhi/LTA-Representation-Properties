import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import os

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode_reward = 0
        self.episode_rewards = []
        self.image_array = [] # This one is needed for creating the videos
        self.record_video = self.cfg.record_video
        self.total_steps = 0
        self.reset = True
        self.ep_steps = 0
        self.num_episodes = 0
        self.timeout = cfg.timeout
        self.stats_counter = 0
        self.batch_indices = torch.arange(self.cfg.batch_size).long().to(cfg.device)
        self.ep_returns_queue = np.zeros(cfg.stats_queue_size)
        self.env = cfg.env_fn()
        self.eval_env = cfg.env_fn()

        self.step_reward = np.zeros(self.timeout)

    def update_stats(self, reward, done):
        
        self.step_reward[self.ep_steps] = reward

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        if self.record_video:
            self.image_array.append(self.state)
        
        if done or self.ep_steps == self.timeout:
            if self.ep_steps < self.timeout:
                self.step_reward[self.ep_steps:] = -3
                self.step_reward[self.ep_steps] = -2

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

            if self.record_video:
                self.cfg.logger.log_video(self.image_array, len(self.episode_rewards), 10, (150, 150, 3))
                del self.image_array
                self.image_array = []
#             if (self.num_episodes <= 20  or self.total_steps >= self.cfg.max_steps*(1-0.01)) and \
                # self.num_episodes % 10 == 0:
                # # cmap = cm.get_cmap('viridis', 5)
                # cmap = (mpl.colors.ListedColormap(["white", 'red', 'cyan', 'grey', 'yellow', ]))
                # plt.figure()
                # temp = self.step_reward.reshape((25, 20))
                # plt.imshow(temp, cmap=cmap, vmax=1.5, vmin=-3.5)
                # plt.colorbar()
                # viz_dir = self.cfg.get_visualization_dir()
                # viz_file = 'reward_ep{}.png'.format(self.num_episodes)
                # plt.savefig(os.path.join(viz_dir, viz_file))
                # plt.close()
                # plt.clf()
                # print("Save reward plot in {}".format(viz_file))
            self.step_reward = np.zeros(self.timeout)

    def add_episode_return(self, ep_return):
        self.ep_returns_queue[self.stats_counter] = ep_return
        self.stats_counter += 1
        self.stats_counter = self.stats_counter % self.cfg.stats_queue_size

    def populate_returns(self):
        for ep in range(self.cfg.stats_queue_size):
            ep_return, steps = self.populate_episode()
            if self.cfg.evaluation_criteria == "return":
                self.add_episode_return(ep_return)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_episode_return(steps)
            else:
                raise NotImplementedError

    def populate_episode(self):
        state = self.env.reset()
        total_rewards = 0
        ep_steps = 0
        while True:
            action = self.policy(state, self.cfg.eps_schedule.read_only())
            state, reward, done, _ = self.env.step([action])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.cfg.timeout:
                break
        return total_rewards, ep_steps

    def eval_episodes(self, elapsed_time=None):
        eval_res = []
        
        for ep in range(self.cfg.eval_episodes):
            ep_return, steps = self.eval_episode()
            if self.cfg.evaluation_criteria == "return":
                eval_res.append(ep_return)
            elif self.cfg.evaluation_criteria == "steps":
                eval_res.append(steps)
            else:
                raise NotImplementedError

        eval_res = np.array(eval_res)
        mean, median, min, max = np.mean(eval_res), np.median(eval_res), np.min(eval_res), np.max(eval_res)

        total_episodes = len(self.episode_rewards)
        log_str = 'EVAL: total steps %d, total episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'

        self.cfg.logger.info(log_str % (self.total_steps, total_episodes, mean, median,
                                        min, max, len(eval_res),
                                        elapsed_time))
        
        self.episode_reward = 0
        self.ep_steps = 0
        self.reset = True
        return

    def eval_episode(self):
        state = self.env.reset()
        total_rewards = 0
        ep_steps = 0
        
        while True:
            action = self.eval_step(state)
            state, reward, done, _ = self.env.step([action])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.cfg.timeout:
                break
        
        return total_rewards, ep_steps

    def policy(self, state, eps):
        raise NotImplementedError

    def eval_step(self, state):
        action = self.policy(state, 0)
        return action

    # def save(self, filename):
    #     raise NotImplementedError
    #
    # def load(self, filename):
    #     raise NotImplementedError

    def one_hot_action(self, actions):
        one_hot = np.zeros((len(actions), self.cfg.action_dim))
        np.put_along_axis(one_hot, actions.reshape((-1, 1)), 1, axis=1)
        return one_hot
