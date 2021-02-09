import os
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import seaborn as sns

from core.agent import base
from core.utils.torch_utils import tensor


class Baseline(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # self.rep_net = cfg.rep_fn()
        self.env = cfg.env_fn()

        self.state = None
        self.action = None
        self.next_state = None

        # self._tensor = lambda x: tensor(self.cfg.state_normalizer(x), self.cfg.device)

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

        # f_s = self.rep_net(self._tensor(states))
        # f_g = self.rep_net(self._tensor(goal_states))
        f_s = self.cfg.state_normalizer(states)
        f_g = self.cfg.state_normalizer(goal_states)
        f_s = f_s.reshape((len(f_s), -1))
        f_g = f_g.reshape((len(f_g), -1))

        # a plot with rows = num of goal-states and cols = 1
        fig, ax = plt.subplots(nrows=len(goal_states), ncols=1, figsize=(6, 6 * len(goal_states)))
        max_x = max(state_coords, key=lambda xy: xy[0])[0]
        max_y = max(state_coords, key=lambda xy: xy[1])[1]

        def compute_distance_map(f_states, f_goal, size_x, size_y):
            l2_vec = ((f_states - f_goal)**2).sum(axis=1)#.sum(axis=1).sum(axis=1)
            distance_map = np.zeros((size_x, size_y))
            for k, xy_coord in enumerate(state_coords):
                x, y = xy_coord
                distance_map[x][y] = l2_vec[k]
            return distance_map

        for g_k in range(len(goal_states)):
            _distance_map = compute_distance_map(f_s, f_g[g_k], max_x+1, max_y+1)
            sns.heatmap(_distance_map, ax=ax[g_k])
            ax[g_k].set_title('Goal: {}'.format(goal_coords[g_k]))

        viz_dir = self.cfg.get_visualization_dir()
        viz_file = 'visualization.png'
        plt.savefig(os.path.join(viz_dir, viz_file))
        plt.close()
