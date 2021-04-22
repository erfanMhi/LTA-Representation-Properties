import numpy as np

from core.utils.torch_utils import random_seed


class GridHardXY:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.state_dim = (2,)
        self.action_dim = 4
        self.obstacles_map = self.get_obstacles_map()
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14
        self.goal_x, self.goal_y = 9, 9
        self.current_state = None

    def generate_state(self, coords):
        return np.array(coords)

    def info(self, key):
        return

    def reset(self):
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y):
                self.current_state = rand_state[0], rand_state[1]
                return self.generate_state(self.current_state)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not self.obstacles_map[nx][ny]:
            x, y = nx, ny

        self.current_state = x, y
        if x == self.goal_x and y == self.goal_y:
            return self.generate_state([x, y]), np.asarray(1.0), np.asarray(True), ""
        else:
            return self.generate_state([x, y]), np.asarray(0.0), np.asarray(False), ""

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(15)
                       for y in range(15) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        goal_coords = [[9, 9], [0, 0], [14, 0], [7, 14]]
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[2, 0:6] = 1.0
        _map[2, 8:] = 1.0
        _map[3, 5] = 1.0
        _map[4, 5] = 1.0
        _map[5, 2:7] = 1.0
        _map[5, 9:] = 1.0
        _map[8, 2] = 1.0
        _map[8, 5] = 1.0
        _map[8, 8:] = 1.0
        _map[9, 2] = 1.0
        _map[9, 5] = 1.0
        _map[9, 8] = 1.0
        _map[10, 2] = 1.0
        _map[10, 5] = 1.0
        _map[10, 8] = 1.0
        _map[11, 2:6] = 1.0
        _map[11, 8:12] = 1.0
        _map[12, 5] = 1.0
        _map[13, 5] = 1.0
        _map[14, 5] = 1.0

        return _map

    def get_useful(self, state=None):
        if state:
            return state
        else:
            return self.current_state


class GridHardGS(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 1)

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
        return state

    def get_features(self, state):
        raise NotImplementedError


class GridHardRGB(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)

        """
        # Gray-scale image
            Walls are Red
            Open spaces are Green
            Agent is Blue
        """
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
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color on
        state[x][y][2] = 255.0  # turning the blue color on
        return state

    def get_features(self, state):
        raise NotImplementedError

    def get_useful(self, state=None):
        blue = np.array([0., 0., 255.])
        if state is None:
            state = self.generate_state(self.current_state)
        idx = np.where(np.all(state==blue, axis=2) == True)
        coord = np.array([idx[0][0], idx[1][0]])
        return coord


class GridTwoRoomXY(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goal_x, self.goal_y = 8, 14

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(15)
                       for y in range(15) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        goal_coords = [[8, 14], [14, 0], [7, 7]]
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[7, :7] = 1.0
        _map[7, 9:] = 1.0
        return _map


class GridTwoRoomRGB(GridTwoRoomXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)

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
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color on
        state[x][y][2] = 255.0  # turning the blue color on
        return state

    def get_useful(self, state=None):
        blue = np.array([0., 0., 255.])
        if state is None:
            state = self.generate_state(self.current_state)
        idx = np.where(np.all(state==blue, axis=2) == True)
        coord = np.array([idx[0][0], idx[1][0]])
        return coord


class GridOneRoomXY(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goal_x, self.goal_y = 14, 14

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(15)
                       for y in range(15) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        goal_coords = [[8, 14], [14, 0], [7, 7]]
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        return _map


class GridOneRoomRGB(GridOneRoomXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)

        self.rgb_template = np.zeros(self.state_dim)
        for x in range(d):
            for y in range(d):
                self.rgb_template[x][y][1] = 255.0

    def generate_state(self, coords):
        state = np.copy(self.rgb_template)
        x, y = coords
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color on
        state[x][y][2] = 255.0  # turning the blue color on
        return state

    def get_useful(self, state=None):
        blue = np.array([0., 0., 255.])
        if state is None:
            state = self.generate_state(self.current_state)
        idx = np.where(np.all(state==blue, axis=2) == True)
        coord = np.array([idx[0][0], idx[1][0]])
        return coord

