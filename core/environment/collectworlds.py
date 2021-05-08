import numpy as np
import matplotlib.pyplot as plt

from core.utils.torch_utils import random_seed


class CollectXY:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)

        self.object_coords = [(7, 2), (2, 7), (8, 6), (6, 8),
                              (8, 0), (0, 8), (14, 0), (0, 14),
                              (6, 14), (14, 6), (7, 11), (11, 7)]

        # one indiciate the object is available to be picked up
        self.object_status = np.ones(12)
        self.action_dim = 4

        self.obstacles_map = self.get_obstacles_map()
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.agent_loc = (0, 0)
        self.object_status = np.ones(len(self.object_coords))

        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14
        self.goal_x, self.goal_y = 14, 14

        self.blues, self.greens, self.reds = None, None, None
        self.rewarding_color = 'red'
        self.rewarding_blocks = None

    def info(self, key):
        return

    def generate_state(self, agent_loc, object_status, reds, greens, blues):
        reds = np.array(reds).flatten()
        greens = np.array(greens).flatten()
        blues = np.array(blues).flatten()
        return np.concatenate([np.array(agent_loc), object_status * 14, reds, greens, blues])

    def reset(self):
        obj_ids = np.arange(len(self.object_coords))
        obj_ids = np.random.permutation(obj_ids)

        red_ids, green_ids, blue_ids = obj_ids[:4], obj_ids[4:8], obj_ids[8:]

        self.reds = [self.object_coords[k] for k in red_ids]
        self.greens = [self.object_coords[k] for k in green_ids]
        self.blues = [self.object_coords[k] for k in blue_ids]
        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'blue':
            self.rewarding_blocks = self.blues
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.reds, self.greens, self.blues)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.agent_loc

        nx = x + dx
        ny = y + dy

        # Ensuring the next position is within bounds
        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)
        if not self.obstacles_map[nx][ny]:
            x, y = nx, ny

        reward = 0.0
        if (x, y) in self.object_coords:
            object_idx = self.object_coords.index((x, y))
            if self.object_status[object_idx]:
                # the object is available for picking
                self.object_status[object_idx] = 0.0

                if (x, y) in self.rewarding_blocks:
                    reward += 1.0
                else:
                    reward += -0.5

        self.agent_loc = x, y

        reward -= 0.001

        done = np.asarray(True) if x == self.goal_x and y == self.goal_y else np.asarray(False)
        state = self.generate_state(self.agent_loc, self.object_status, self.reds, self.greens, self.blues)
        return state, np.asarray(reward), done, ""

    def get_visualization_segment(self):
        raise NotImplementedError
        # state_coords = [[x, y] for x in range(15)
        #                for y in range(15) if not int(self.obstacles_map[x][y])]
        # states = [self.generate_state(coord, self.object_status, [], [], []) for coord in state_coords]
        # goal_coords = [[14, 14]]
        # goal_states = [self.generate_state(coord, self.object_status, [], [], []) for coord in goal_coords]
        # return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[7, 0:2] = 1.0
        _map[7, 4:11] = 1.0
        _map[7, 13:] = 1.0

        _map[0:2, 7] = 1.0
        _map[4:11, 7] = 1.0
        _map[13:, 7] = 1.0

        return _map


class CollectRGB(CollectXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)
        self.main_template = np.zeros(self.state_dim)
        for x in range(d):
            for y in range(d):
                if self.obstacles_map[x][y]:
                    self.main_template[x][y] = np.array([0, 0, 0])
                else:
                    self.main_template[x][y] = np.array([128, 128, 128])
        self.main_template[self.goal_x][self.goal_y] = np.array([255, 255, 255])

        self.episode_template = None

    def get_episode_template(self, reds, greens, blues):
        episode_template = np.copy(self.main_template)

        for rx, ry in reds:
            episode_template[rx][ry] = np.array([255, 0, 0])

        for gx, gy in greens:
            episode_template[gx][gy] = np.array([0, 255, 0])

        for bx, by in blues:
            episode_template[bx][by] = np.array([0, 0, 255])

        return episode_template

    def reset(self):
        obj_ids = np.arange(len(self.object_coords))
        obj_ids = np.random.permutation(obj_ids)

        red_ids, green_ids, blue_ids = obj_ids[:4], obj_ids[4:8], obj_ids[8:]

        self.reds = [self.object_coords[k] for k in red_ids]
        self.greens = [self.object_coords[k] for k in green_ids]
        self.blues = [self.object_coords[k] for k in blue_ids]

        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'blue':
            self.rewarding_blocks = self.blues
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        self.episode_template = self.get_episode_template(self.reds, self.greens, self.blues)
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.reds, self.greens, self.blues)

    def generate_state(self, agent_loc, object_status, reds, greens, blues):
        state = np.copy(self.episode_template)
        x, y = agent_loc

        for object_idx, coord in enumerate(self.object_coords):
            if not self.object_status[object_idx]:
                ox, oy = coord
                state[ox][oy] = np.array([128, 128, 128])
        state[x][y] = np.array([255, 255, 0])
        return state

    def get_useful(self, img=None):
        raise NotImplementedError
        # if img is None:
        #     img = self.generate_state(self.agent_loc, self.object_status, self.reds, self.greens, self.blues)
        #
        # red, green, blue, gray, yellow = np.array([255., 0., 0.]), np.array([0., 255., 0.]), \
        #                          np.array([0., 0., 255.]), np.array([128., 128., 128.]), np.array([255, 255, 0])
        # target = np.zeros((len(self.object_coords)))
        #
        # agent_loc = np.argwhere(np.logical_and(np.logical_and(img[:, :, 2] == 255.0,
        #                                                         img[:, :, 1] == 0.0),
        #                                          img[:, :, 0] == 0.0))[:, 1:]
        #
        # object_coords = np.array(self.object_coords)
        # color = img[object_coords[:, 0], object_coords[:, 1]]
        # target[np.all(color == red, axis=1)] = 0
        # target[np.all(color == green, axis=1)] = 1
        # target[np.all(color == blue, axis=1)] = 2
        # target[np.all(color == gray, axis=1)] = 3
        # target[np.all(color == yellow, axis=1)] = 4
        # # target = target / 4 * self.max_x  # fit the data range used in the normalizer
        # # target = target * self.max_x  # fit the data range used in the normalizer
        # return np.concatenate([agent_loc, target])

    def get_visualization_segment(self):
        if self.episode_template is not None:
            obj_ids = np.arange(len(self.object_coords))
            obj_ids = np.random.permutation(obj_ids)
            red_ids, green_ids, blue_ids = obj_ids[:4], obj_ids[4:8], obj_ids[8:]
            self.reds = [self.object_coords[k] for k in red_ids]
            self.greens = [self.object_coords[k] for k in green_ids]
            self.blues = [self.object_coords[k] for k in blue_ids]

            state_coords = [[x, y] for x in range(15)
                           for y in range(15) if not int(self.obstacles_map[x][y])]
            states = [self.generate_state(coord, self.object_status, self.reds, self.greens) for coord in state_coords]
            goal_coords = [[self.goal_x, self.goal_y]]
            goal_states = [self.generate_state(coord, self.object_status, self.reds, self.greens) for coord in goal_coords]
            return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)
        else:
            raise NotImplementedError


class CollectColor(CollectRGB):
    def __init__(self, rewarding_color, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.rewarding_color = rewarding_color


class CollectTwoColorXY:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)

        self.object_coords = [(7, 2), (2, 7), (8, 6), (6, 8),
                              (8, 0), (0, 8), (14, 0), (0, 14),
                              (6, 14), (14, 6), (7, 11), (11, 7)]

        # one indiciate the object is available to be picked up
        self.object_status = np.ones(12)
        self.action_dim = 4

        self.obstacles_map = self.get_obstacles_map()
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.agent_loc = (0, 0)
        self.object_status = np.ones(len(self.object_coords))

        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14
        self.goal_x, self.goal_y = 14, 14

        self.greens, self.reds = None, None
        self.rewarding_color = 'green'
        self.rewarding_blocks = None

    def info(self, key):
        return

    def generate_state(self, agent_loc, object_status, reds, greens):
        greens = np.array(greens).flatten()
        reds = np.array(reds).flatten()
        return np.concatenate([np.array(agent_loc), object_status * 14, greens, reds])

    def reset(self):
        obj_ids = np.arange(len(self.object_coords))
        obj_ids = np.random.permutation(obj_ids)

        green_ids, red_ids = obj_ids[:6], obj_ids[6:]

        self.greens = [self.object_coords[k] for k in green_ids]
        self.reds = [self.object_coords[k] for k in red_ids]
        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.agent_loc

        nx = x + dx
        ny = y + dy

        # Ensuring the next position is within bounds
        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)
        if not self.obstacles_map[nx][ny]:
            x, y = nx, ny

        reward = 0.0
        if (x, y) in self.object_coords:
            object_idx = self.object_coords.index((x, y))
            if self.object_status[object_idx]:
                # the object is available for picking
                self.object_status[object_idx] = 0.0

                if (x, y) in self.rewarding_blocks:
                    reward += 1.0
                else:
                    reward += -1.0

        self.agent_loc = x, y

        reward -= 0.001

        done = np.asarray(True) if x == self.goal_x and y == self.goal_y else np.asarray(False)
        state = self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)
        return state, np.asarray(reward), done, ""

    def get_visualization_segment(self):
        raise NotImplementedError

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[7, 0:2] = 1.0
        _map[7, 4:11] = 1.0
        _map[7, 13:] = 1.0

        _map[0:2, 7] = 1.0
        _map[4:11, 7] = 1.0
        _map[13:, 7] = 1.0

        return _map


class CollectTwoColorXYEarlyTermin:
    def __init__(self, seed=np.random.randint(int(1e5)), fruit_num=6):
        random_seed(seed)

        self.object_coords = [(7, 2), (2, 7), (8, 6), (6, 8),
                              (8, 0), (0, 8), (14, 0), (0, 14),
                              (6, 14), (14, 6), (7, 11), (11, 7)]

        self.fruit_num = fruit_num
        
        # Reducing number of the fruits to fruit_num
        self.object_coords = self.object_coords[:self.fruit_num] + self.object_coords[-self.fruit_num:]
        
        # one indiciate the object is available to be picked up
        self.object_status = np.ones(self.fruit_num*2)
        self.action_dim = 4

        self.obstacles_map = self.get_obstacles_map()
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.agent_loc = (0, 0)
        self.object_status = np.ones(len(self.object_coords))

        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14

        self.greens, self.reds = None, None
        self.rewarding_color = 'green'
        self.rewarding_blocks = None
        self.correct_collect = 0
        self.wrong_collect = 0

    def info(self, key):
        return

    def generate_state(self, agent_loc, object_status, reds, greens):
        greens = np.array(greens).flatten()
        reds = np.array(reds).flatten()
        return np.concatenate([np.array(agent_loc), object_status * 14, greens, reds])

    def reset(self):
        obj_ids = np.arange(len(self.object_coords))
        obj_ids = np.random.permutation(obj_ids)

        green_ids, red_ids = obj_ids[:self.fruit_num], obj_ids[self.fruit_num:]

        self.greens = [self.object_coords[k] for k in green_ids]
        self.reds = [self.object_coords[k] for k in red_ids]
        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        self.correct_collect = 0
        self.wrong_collect = 0

        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.agent_loc

        nx = x + dx
        ny = y + dy

        # Ensuring the next position is within bounds
        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)
        if not self.obstacles_map[nx][ny]:
            x, y = nx, ny

        reward = 0.0
        if (x, y) in self.object_coords:
            object_idx = self.object_coords.index((x, y))
            if self.object_status[object_idx]:
                # the object is available for picking
                self.object_status[object_idx] = 0.0

                if (x, y) in self.rewarding_blocks:
                    reward += 1.0
                    self.correct_collect += 1
                    # print(self.correct_collect, self.rewarding_color, reward)
                else:
                    reward += -1.0
                    self.wrong_collect += 1
                    # print(self.correct_collect, self.rewarding_color, reward)

        self.agent_loc = x, y

        reward -= 0.001

        done = np.asarray(True) if self.correct_collect==self.fruit_num else np.asarray(False)
        # if done:
        #     print("done", self.correct_collect, reward, self.rewarding_color)
        state = self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)
        return state, np.asarray(reward), done, ""

    def get_visualization_segment(self):
        raise NotImplementedError

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[7, 0:2] = 1.0
        _map[7, 4:11] = 1.0
        _map[7, 13:] = 1.0

        _map[0:2, 7] = 1.0
        _map[4:11, 7] = 1.0
        _map[13:, 7] = 1.0

        return _map


class CollectTwoColorRGB(CollectTwoColorXYEarlyTermin):
    def __init__(self, seed=np.random.randint(int(1e5)), fruit_num=6):
        super().__init__(seed, fruit_num)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)
        self.main_template = np.zeros(self.state_dim)
        for x in range(d):
            for y in range(d):
                if self.obstacles_map[x][y]:
                    self.main_template[x][y] = np.array([0, 0, 0])
                else:
                    self.main_template[x][y] = np.array([128, 128, 128])

        # self.main_template[self.goal_x][self.goal_y] = np.array([255, 255, 255])

        self.episode_template = None

    def get_episode_template(self, greens, reds):
        episode_template = np.copy(self.main_template)

        for rx, ry in reds:
            episode_template[rx][ry] = np.array([255, 0, 0])

        for gx, gy in greens:
            episode_template[gx][gy] = np.array([0, 255, 0])

        return episode_template

    def reset(self):
        obj_ids = np.arange(len(self.object_coords))
        obj_ids = np.random.permutation(obj_ids)

        green_ids, red_ids = obj_ids[:self.fruit_num], obj_ids[self.fruit_num:]

        self.reds = [self.object_coords[k] for k in red_ids]
        self.greens = [self.object_coords[k] for k in green_ids]

        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        self.correct_collect = 0
        self.wrong_collect = 0

        self.episode_template = self.get_episode_template(self.greens, self.reds)
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            # if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
            if not int(self.obstacles_map[rx][ry]) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)

    def generate_state(self, agent_loc, object_status, greens, reds):
        state = np.copy(self.episode_template)
        x, y = agent_loc

        for object_idx, coord in enumerate(self.object_coords):
            if not self.object_status[object_idx]:
                ox, oy = coord
                state[ox][oy] = np.array([128, 128, 128])
        state[x][y] = np.array([0, 0, 255])
        return state

    def get_useful(self, img=None):
        if img is None:
            img = self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)
        red, green, blue, gray = np.array([255., 0., 0.]), np.array([0., 255., 0.]), \
                                 np.array([0., 0., 255.]), np.array([128., 128., 128.])

        agent_loc = np.argwhere(np.logical_and(np.logical_and(img[:, :, 2] == 255.0,
                                                                img[:, :, 1] == 0.0),
                                                 img[:, :, 0] == 0.0))[0]
        target = np.zeros((len(self.object_coords)))

        object_coords = np.array(self.object_coords)
        color = img[object_coords[:, 0], object_coords[:, 1]]
        reds = np.all(color == red, axis=1)
        target[reds] = 0
        greens = np.all(color == green, axis=1)
        target[greens] = 1
        target[np.all(color == blue, axis=1)] = 2
        target[np.all(color == gray, axis=1)] = 3
        count = np.array([len(np.where(reds==True)[0]), len(np.where(greens==True)[0])])
        # count = count / len(self.object_coords) * 2 - 1
        # target = target / 3 * self.max_x  # fit the data range used in the normalizer
        # target = target * self.max_x  # fit the data range used in the normalizer
        return np.concatenate([np.array(agent_loc), target, count])

        # # reds: [255, 0, 0]
        # # greens: [0, 255, 0]
        # color = np.zeros((len(self.object_coords)))
        # for i in range(len(self.object_coords)):
        #     coord = self.object_coords[i]
        #     idx = np.where(self.episode_template[coord] == 255)
        #     color[i] = idx * self.max_x # fit the data range used in the normalizer
        # return self.agent_loc, color

    def get_visualization_segment(self):
        if self.episode_template is not None:
            obj_ids = np.arange(len(self.object_coords))
            obj_ids = np.random.permutation(obj_ids)
            red_ids, green_ids, blue_ids = obj_ids[:4], obj_ids[4:8], obj_ids[8:]
            self.reds = [self.object_coords[k] for k in red_ids]
            self.greens = [self.object_coords[k] for k in green_ids]
            self.blues = [self.object_coords[k] for k in blue_ids]

            state_coords = [[x, y] for x in range(15)
                           for y in range(15) if not int(self.obstacles_map[x][y])]
            states = [self.generate_state(coord, self.object_status, self.greens, self.reds) for coord in state_coords]
            goal_coords = [[14, 14]]
            goal_states = [self.generate_state(coord, self.object_status, self.greens, self.reds) for coord in goal_coords]
            return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)
        else:
            raise NotImplementedError
        # raise NotImplementedError

class CollectRandColorRGB(CollectTwoColorRGB):
    def __init__(self, seed=np.random.randint(int(1e5)), fruit_num=6):
        super().__init__(seed, fruit_num=fruit_num)
        self.all_rewarding = ['green', 'red']

    def reset(self):
        obj_ids = np.arange(len(self.object_coords))
        obj_ids = np.random.permutation(obj_ids)
        green_ids, red_ids = obj_ids[:self.fruit_num], obj_ids[self.fruit_num:]

        self.reds = [self.object_coords[k] for k in red_ids]
        self.greens = [self.object_coords[k] for k in green_ids]

        self.rewarding_color = self.all_rewarding[np.random.randint(len(self.all_rewarding))]

        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        self.correct_collect = 0
        self.wrong_collect = 0
        # print("reset", self.rewarding_color, self.correct_collect)

        self.episode_template = self.get_episode_template(self.greens, self.reds)
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            # if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
            if not int(self.obstacles_map[rx][ry]) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)

    def info(self, key):
        if key == "use_head":
            # return self.all_rewarding.index(self.rewarding_color)
            head = 0 if self.rewarding_color == 'green' else 1
            return head
        elif key == "left_fruit":
            # return green fruit number first, then red fruit
            if self.rewarding_color == 'green':
                return self.fruit_num-self.correct_collect, self.fruit_num-self.wrong_collect
            elif self.rewarding_color == 'red':
                return self.fruit_num - self.wrong_collect, self.fruit_num - self.correct_collect
            else:
                raise NotImplementedError
        return

class CollectRandColorRGBTest(CollectRandColorRGB):
    def __init__(self, seed=np.random.randint(int(1e5)), fruit_num=6, color='green'):
        super().__init__(seed, fruit_num=fruit_num)
        self.test_color = color
    def reset(self):
        obj_ids = np.arange(len(self.object_coords))
        obj_ids = np.random.permutation(obj_ids)
        green_ids, red_ids = obj_ids[:self.fruit_num], obj_ids[self.fruit_num:]

        self.reds = [self.object_coords[k] for k in red_ids]
        self.greens = [self.object_coords[k] for k in green_ids]

        self.rewarding_color = self.test_color
        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        self.correct_collect = 0
        self.wrong_collect = 0
        # print("reset", self.rewarding_color, self.correct_collect)

        self.episode_template = self.get_episode_template(self.greens, self.reds)
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            # if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
            if not int(self.obstacles_map[rx][ry]) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)

class CollectTwoColor(CollectTwoColorRGB):
    def __init__(self, rewarding_color, seed=np.random.randint(int(1e5)), fruit_num=6):
        super().__init__(seed, fruit_num=fruit_num)
        self.rewarding_color = rewarding_color


class CollectTwoColorLip(CollectTwoColor):
    def __init__(self, lip_sampled_states_path, rewarding_color, seed=np.random.randint(int(1e5)), fruit_num=6):
        super().__init__(rewarding_color, seed, fruit_num=fruit_num)

        self.lip_sampled_states = np.load(lip_sampled_states_path)

    def get_visualization_segment(self):
        return self.lip_sampled_states, None, None, None


class CollectTwoColorRGBFix(CollectTwoColorRGB):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

    def reset(self):
        obj_ids = np.arange(len(self.object_coords))

        green_ids, red_ids = obj_ids[:6], obj_ids[6:]

        self.reds = [self.object_coords[k] for k in red_ids]
        self.greens = [self.object_coords[k] for k in green_ids]

        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        self.episode_template = self.get_episode_template(self.greens, self.reds)
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)


class CollectTwoColorRGBSimple(CollectTwoColorRGB):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

    def reset(self):
        obj_ids = np.arange(len(self.object_coords))
        a, b = obj_ids[:6], obj_ids[6:]
        if np.random.random() < 0.5:
            green_ids, red_ids = a, b
        else:
            green_ids, red_ids = b, a

        self.reds = [self.object_coords[k] for k in red_ids]
        self.greens = [self.object_coords[k] for k in green_ids]

        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
        elif self.rewarding_color == 'green':
            self.rewarding_blocks = self.greens
        else:
            raise NotImplementedError

        self.episode_template = self.get_episode_template(self.greens, self.reds)
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                self.object_status = np.ones(len(self.object_coords))
                return self.generate_state(self.agent_loc, self.object_status, self.greens, self.reds)


def draw(state):
    frame = state.astype(np.uint8)
    figure, ax = plt.subplots()
    ax.imshow(frame)
    plt.axis('off')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # env = CollectXY()
    # state = env.reset()
    # print('State: ', state)
    # done = False
    # while not done:
    #     action = int(input('input_action: '))
    #     state, reward, done, _ = env.step([action])
    #     print(state, ' ', reward, ' ', done)



    # env = CollectColor('green')
    env = CollectRandColorRGB(1, 3)
    state = env.reset()
    draw(state)
    done = False
    while not done:
        try:
            action = int(input('input_action: '))
        except ValueError:
            action = 0
        state, reward, done, _ = env.step([action])
        print(reward, ' ', done, env.rewarding_blocks, env.correct_collect)
        draw(state)

