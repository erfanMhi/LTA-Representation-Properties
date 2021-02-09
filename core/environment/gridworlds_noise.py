from core.environment.gridworlds import *
import core.environment.gridworlds

class NoisyGridHardXY(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5)), ns_cl=1):
        super().__init__(seed)
        self.state_dim = 2 + ns_cl
        self.ns_cl = ns_cl

    def generate_state(self, coords):
        return np.concatenate([np.array(coords), np.random.randint(self.min_x, self.max_x+1, size=self.ns_cl)], axis=0)

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(15)
                       for y in range(15) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        goal_coords = [[9, 9], [0, 0], [14, 0], [7, 14]]
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)


class NoisyGridHardGS(GridHardGS):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d+d//2, d+d//2, 1)
        self.d = d

        # """
        # # Gray-scale image
        #     walls are white: 255.0
        #     agent is gray:   128.0
        #     open space is black: 0.0
        # """
        # self.gray_template = np.ones(self.state_dim) - 1
        # for x in range(d):
        #     for y in range(d):
        #         if self.obstacles_map[x][y]:
        #             self.gray_template[x][y] = 255.0
        """
        # Gray-scale image
            walls are black: 0.0
            agent is gray:   128.0
            open space is white: 255.0
        """
        self.gray_template = np.ones(self.state_dim) * 255.0
        for x in range(d):
            for y in range(d):
                if self.obstacles_map[x][y]:
                    self.gray_template[x][y] = 0.0

    def generate_state(self, coords):
        state = np.copy(self.gray_template)
        x, y = coords
        state[x][y] = 128.0
        state = self._generate_noise(state)
        return state

    def get_features(self, state):
        raise NotImplementedError

    def _generate_noise(self, state):
        state[np.random.randint(self.d, self.d+self.d//2), np.random.randint(0, self.d)] = 128.0
        return state

def grid_fake_agent_simple(parent, seed):
    parent = getattr(core.environment.gridworlds, parent)
    class GridFakeAgentSimple(parent):
        def __init__(self, seed=np.random.randint(int(1e5))):
            super().__init__(seed)

            d = len(self.obstacles_map)
            self.state_dim = (d+d//2, d+d//2, 3)
            self.d = d
            """
            # RGB image
                Walls are Red
                Open spaces are Green
                Agent is Blue
            """
            self.rgb_template = np.zeros(self.state_dim)
            for x in range(self.d):
                for y in range(self.d):
                    if self.obstacles_map[x][y]:
                        self.rgb_template[x][y][0] = 255.0
                    else:
                        self.rgb_template[x][y][1] = 255.0

        def generate_state(self, coords):
            state = np.copy(self.rgb_template)
            x, y = coords
            assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0, print(state[x][y])

            state[x][y][1] = 0.0    # setting the green color off
            state[x][y][2] = 255.0  # turning the blue color on
            state = self._generate_noise(state)
            return state

        def _generate_noise(self, state):
            x, y = np.random.randint(self.d, self.d+self.d//2), np.random.randint(0, self.d)
            state[x, y, 1] = 0.0
            state[x, y, 2] = 255.0
            return state
    return GridFakeAgentSimple(seed)

def grid_fake_agent(parent, seed):
    parent = getattr(core.environment.gridworlds, parent)
    class GridFakeAgent(parent):
        def __init__(self, seed=np.random.randint(int(1e5))):
            super().__init__(seed)

            d = len(self.obstacles_map)
            self.obstacles_map = np.concatenate((self.obstacles_map, np.zeros((d//2, d))), axis=0)
            self.obstacles_map = np.concatenate((self.obstacles_map, np.zeros((d+d//2, d//2))), axis=1)
            self.state_dim = (d+d//2, d+d//2, 3)
            self.d = d
            """
            # RGB image
                Walls are Red
                Open spaces are Green
                Agent is Blue
            """
            self.rgb_template = np.zeros(self.state_dim)
            for x in range(self.state_dim[0]):
                for y in range(self.state_dim[1]):
                    if self.obstacles_map[x][y]:
                        self.rgb_template[x][y][0] = 255.0
                    else:
                        self.rgb_template[x][y][1] = 255.0

        def generate_state(self, coords):
            state = np.copy(self.rgb_template)
            x, y = coords
            assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0, print(state[x][y])

            state[x][y][1] = 0.0    # setting the green color off
            state[x][y][2] = 255.0  # turning the blue color on
            state = self._generate_noise(state)
            return state

        def _generate_noise(self, state):
            if np.random.random() < (self.d / (2*self.d+self.d//2)):
                x, y = np.random.randint(self.d, self.d+self.d//2), np.random.randint(0, self.d)
            else:
                x, y = np.random.randint(0, self.d+self.d//2), np.random.randint(self.d, self.d+self.d//2)
            state[x, y, 1] = 0.0
            state[x, y, 2] = 255.0
            return state
    return GridFakeAgent(seed)

def grid_number(parent, seed):
    parent = getattr(core.environment.gridworlds, parent)
    class GridNumber(parent):
        def __init__(self, seed=np.random.randint(int(1e5))):
            super().__init__(seed)

            d = len(self.obstacles_map)
            self.d = d
            self.n = self.d//2
            self.state_dim = (self.d+self.n, self.d+self.n, 3)
            """
            # RGB image
                Walls are Red
                Open spaces are Green
                Agent is Blue
            """
            self.number_template = self._get_number_template()
            self.rgb_template = np.zeros(self.state_dim)
            for x in range(d):
                for y in range(d):
                    if self.obstacles_map[x][y]:
                        self.rgb_template[x][y][0] = 255.0
                    else:
                        self.rgb_template[x][y][1] = 255.0

        def generate_state(self, coords):
            state = np.copy(self.rgb_template)
            x, y = coords
            assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0, print(state[x][y])

            state[x][y][1] = 0.0    # setting the green color on
            state[x][y][2] = 255.0  # turning the blue color on
            state = self._generate_noise(state)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(state)
            # plt.show()
            return state

        def _generate_noise(self, state):
            numbers = np.random.randint(0, 10, size=3)
            state[self.d: self.d+self.n, 0: self.n] = self.number_template[numbers[0]]
            state[self.d: self.d+self.n, self.n: self.n*2] = self.number_template[numbers[1]]
            state[self.d: self.d+self.n, self.n*2: self.n*3] = self.number_template[numbers[2]]
            return state

        def _get_number_template(self):
            numbers = np.zeros((10, 7, 7, 3))
            # 0
            numbers[0, 1, 2:6, 0] = 255
            numbers[0, 5, 2:6, 0] = 255
            numbers[0, 2:5, 2, 0] = 255
            numbers[0, 2:5, 5, 0] = 255

            # 1
            numbers[1, 1:6, 5, 0] = 255

            # 2
            numbers[2, 1, 2:6, 0] = 255
            numbers[2, 2, 5, 0] = 255
            numbers[2, 3, 2:6, 0] = 255
            numbers[2, 4, 2, 0] = 255
            numbers[2, 5, 2:6, 0] = 255

            # 3
            numbers[3, 1, 2:6, 0] = 255
            numbers[3, 2, 5, 0] = 255
            numbers[3, 3, 2:6, 0] = 255
            numbers[3, 4, 5, 0] = 255
            numbers[3, 5, 2:6, 0] = 255

            # 4
            numbers[4, 1:6, 5, 0] = 255
            numbers[4, 1:4, 2, 0] = 255
            numbers[4, 3, 3:5, 0] = 255

            # 5
            numbers[5, 1, 2:6, 0] = 255
            numbers[5, 2, 2, 0] = 255
            numbers[5, 3, 2:6, 0] = 255
            numbers[5, 4, 5, 0] = 255
            numbers[5, 5, 2:6, 0] = 255

            # 6
            numbers[6, 1, 2:6, 0] = 255
            numbers[6, 2:5, 2, 0] = 255
            numbers[6, 3, 2:6, 0] = 255
            numbers[6, 4, 5, 0] = 255
            numbers[6, 5, 2:6, 0] = 255

            # 7
            numbers[7, 1:6, 5, 0] = 255
            numbers[7, 1, 2:5, 0] = 255

            # 8
            numbers[8, 1, 2:6, 0] = 255
            numbers[8, 2:5, 2, 0] = 255
            numbers[8, 3, 2:6, 0] = 255
            numbers[8, 2:5, 5, 0] = 255
            numbers[8, 5, 2:6, 0] = 255

            # 9
            numbers[9, 1, 2:6, 0] = 255
            numbers[9, 2, 2, 0] = 255
            numbers[9, 3, 2:6, 0] = 255
            numbers[9, 2:5, 5, 0] = 255
            numbers[9, 5, 2:6, 0] = 255

            return numbers

        def get_features(self, state):
            raise NotImplementedError
    return GridNumber(seed)


def grid_background(parent, seed, change_pxl):
    # parent = importlib.import_module("core.environment.gridworlds."+parent)
    # env = importlib.import_module(parent)
    parent = getattr(core.environment.gridworlds, parent)

    # parent = importlib.import_module(parent)
    class GridBackground(parent):
        def __init__(self, seed=np.random.randint(int(1e5)), change_pxl=0.01):
            super().__init__(seed)
            self.change_percent = change_pxl
            self.change_size = int((self.state_dim[0] * self.state_dim[1] -
                                    len(np.where(self.obstacles_map==1)[0]) - 2) *
                                   self.change_percent)
            self.open_map = self.get_open_map()
            self.open_size = len(np.where(self.open_map==1)[0]) - 1

        def generate_state(self, coords):
            state = np.copy(self.rgb_template)
            x, y = coords
            assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0, print(state[x][y])
            state[x][y][1] = 0.0    # setting the green color off
            state[x][y][2] = 255.0  # turning the blue color on
            state = self._generate_noise(state, coords)
            return state

        def _generate_noise(self, state, coords):
            _map = np.copy(self.open_map)
            _map[coords[0], coords[1]] = 0
            change_idx = np.random.randint(0, self.open_size, size=self.change_size)
            open_x, open_y = np.where(_map==1)
            change_x = open_x[change_idx]
            change_y = open_y[change_idx]
            state[change_x, change_y, 0] = 255
            # print(state[change_x, change_y])
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(state)
            # plt.show()
            return state

        def get_open_map(self):
            _map = np.copy(self.obstacles_map)
            _map = -1 * _map + 1
            _map[self.goal_x, self.goal_y] = 0
            return _map

    return GridBackground(seed, change_pxl)

