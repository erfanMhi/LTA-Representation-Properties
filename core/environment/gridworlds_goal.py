import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../../")
import matplotlib.pyplot as plt

import numpy as np

from core.environment.gridworlds import GridHardXY, GridHardRGB, GridTwoRoomRGB, GridOneRoomRGB


class GridHardXYGoal(GridHardXY):
    def __init__(self, goal_id, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goals = [[9, 9], [0, 0], [0, 14], [14, 0], [14, 14], [7, 7]]
        self.goal_x, self.goal_y = self.goals[goal_id]

class GridHardRGBGoal(GridHardRGB):
    def __init__(self, goal_id, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goals = [[9, 9], [0, 0], [0, 14], [14, 0], [14, 14], [7, 7]]
        self.goal_x, self.goal_y = self.goals[goal_id]


class GridHardRGBGoalAll(GridHardRGB):
    def __init__(self, goal_id, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        # self.nos = (self.state_dim[0] * self.state_dim[1]) - int(np.sum(self.obstacles_map))
        self.goals = [[i, j] for i in range(self.state_dim[0]) \
                              for j in range(self.state_dim[1]) if not self.obstacles_map[i, j]]
        self.goal_x, self.goal_y = self.goals[goal_id]
        self.goal_state_idx = goal_id

    def get_goal(self):
        return self.goal_state_idx, [self.goal_x, self.goal_y]

    def get_goals_list(self):
        return self.goals

    def visualize_goal_id(self):
        ids = np.zeros((self.state_dim[0], self.state_dim[1]))
        for idx, xy in enumerate(self.goals):
            ids[xy[0], xy[1]] = idx
        plt.figure()
        plt.imshow(ids, interpolation='nearest', cmap="Blues")
        for k in range(self.state_dim[0]):
            for j in range(self.state_dim[1]):
                if ids[k, j] != 0:
                    plt.text(j, k, "{:1.0f}".format(ids[k, j]),
                             ha="center", va="center", color="orange")
        plt.show()


class GridHardRGBGoalAllRandom(GridHardRGBGoalAll):
    def __init__(self, goal_ids, seed=np.random.randint(int(1e5))):
        super().__init__(goal_ids[0], seed)
        self.random_starts = []
        self.random_starts_idx = goal_ids
        for goal_id in goal_ids:
            self.random_starts.append(self.goals[goal_id])
        self.choose_goal()

    def choose_goal(self):
        rnd_g = np.random.choice(self.random_starts_idx)
        self.goal_x, self.goal_y = self.goals[rnd_g]
        self.goal_state_idx = rnd_g

    def get_goals(self):
        return self.goal_state_idx, [self.goal_x, self.goal_y], self.random_starts_idx, self.random_starts

    def reset(self):
        self.choose_goal()
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y):
                self.current_state = rand_state[0], rand_state[1]
                return self.generate_state(self.current_state)

    def visualize_goal_id(self):
        ids = np.zeros((self.state_dim[0], self.state_dim[1]))
        for idx, xy in enumerate(self.goals):
            ids[xy[0], xy[1]] = idx
        plt.figure()
        plt.imshow(ids, interpolation='nearest', cmap="Blues")
        for k in range(self.state_dim[0]):
            for j in range(self.state_dim[1]):
                if ids[k, j] != 0 and ids[k, j] in self.random_starts_idx:
                    plt.text(j, k, "{:1.0f}".format(ids[k, j]),
                             ha="center", va="center", color="orange")
        plt.show()

    def generate_state(self, coords):
        state = np.copy(self.rgb_template)
        x, y = coords
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color on
        state[x][y][2] = 255.0  # turning the blue color on
        state[self.goal_x][self.goal_y][:] = 0  # turning the black color on
        return state


class GridTwoRoomRGBGoal(GridTwoRoomRGB):
    def __init__(self, goal_id, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goals = [[8, 14], [14, 0], [7, 7]]
        self.goal_x, self.goal_y = self.goals[goal_id]

class GridOneRoomRGBGoal(GridOneRoomRGB):
    def __init__(self, goal_id, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goals = [[8, 14], [14, 0], [7, 7]]
        self.goal_x, self.goal_y = self.goals[goal_id]

class GridHardRGBMultiGoal(GridHardRGB):
    def __init__(self, task_id, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goals = np.array([[9, 9], [0, 0], [0, 14], [14, 0], [14, 14], [7, 7], [4, 4], [4, 14], [7, 14], [10, 4]])
        self.tasks = [self.goals[[2, 8, 4, 9, 1]], self.goals[[0, 3, 5, 6, 7]]]
        self.goal_sets = self.tasks[task_id]
        self.episode_goal()

    def episode_goal(self):
        idx = np.random.randint(0, len(self.goal_sets))
        # print(idx, self.goal_sets)
        self.goal_x, self.goal_y = self.goal_sets[idx]

    def reset(self):
        self.episode_goal()
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y):
                self.current_state = rand_state[0], rand_state[1]
                return self.generate_state(self.current_state)

    def generate_state(self, coords):
        state = np.copy(self.rgb_template)
        x, y = coords
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color on
        state[x][y][2] = 255.0  # turning the blue color on
        state[self.goal_x][self.goal_y][:] = 0  # turning the black color on
        return state

    # Only used once.
    # Leave the function here to record the random seed
    def generate_source_goal(self):
        rds = np.random.RandomState(seed=0)
        idx = rds.choice(list(range(len(self.goals))), size=len(self.goals)//2, replace=False)
        print(idx) # [2 8 4 9 1]

class GridHardRGBMultiGoalVI(GridHardRGBMultiGoal):
    def __init__(self, task_id, seed=np.random.randint(int(1e5))):
        super().__init__(task_id, seed)
        task_0 = [[0, 11], [1, 0], [2, 8], [4, 2], [3, 12], [5, 9],
                  [7, 5], [7, 11], [9, 9], [10, 3], [11, 0], [11, 7],
                  [11, 12], [14, 4], [14, 11]]
        task_1 = [[7, 0], [0, 4], [4, 14], [9, 12], [13, 13]]
        tasks = [task_0, task_1]
        self.goal_sets = tasks[task_id]
        
        self.episode_goal()

class GridHardRGBMultiGoalSelect(GridHardRGBMultiGoal):
    def __init__(self, goals_file, seed=np.random.randint(int(1e5))):
        super().__init__(0, seed)
        self.goal_sets = np.load(goals_file) 
        
        self.episode_goal()


def draw(state):
    frame = state.astype(np.uint8)
    figure, ax = plt.subplots()
    ax.imshow(frame)
    plt.axis('off')
    plt.show()
    plt.close()

def test_GridHardRGBMultiGoal():
    env = GridHardRGBMultiGoal(task_id=1)
    # env.generate_source_goal()
    done = False
    reset = True
    while not done:
        if reset:
            state = env.reset()
            draw(state)
        action = np.random.randint(4)
        state, reward, reset, _ = env.step([action])
        if reset:
            draw(state)
            print(reward)

def test_GridHardRGBGoalAll():
    env = GridHardRGBGoalAll(goal_id=0)
    env.visualize_goal_id()
    goals = env.get_goals_list()
    print(list(range(len(goals))))

    env = GridHardRGBGoalAll(goal_id=106) # 9,9
    print(env.get_goal())
    env = GridHardRGBGoalAll(goal_id=0) # 0,0
    print(env.get_goal())
    env = GridHardRGBGoalAll(goal_id=172) # 14,14
    print(env.get_goal())
    env = GridHardRGBGoalAll(goal_id=159) # 14,0
    print(env.get_goal())
    env = GridHardRGBGoalAll(goal_id=100) # 9,0
    print(env.get_goal())

    env = GridHardRGBGoalAll(goal_id=106) # 9,9
    done = False
    reset = True
    while not done:
        if reset:
            count = 0
            state = env.reset()
            draw(state)
        action = np.random.randint(4)
        state, reward, reset, _ = env.step([action])
        count += 1
        if reset:
            draw(state)
            print(reward, count, env.goal_x, env.goal_y)

def test_GridHardRGBGoalAllRandom():
    env = GridHardRGBGoalAllRandom(goal_ids=[106, 108, 0])
    # env.visualize_goal_id()
    for _ in range(5):
        s = env.reset()
        print(env.get_goal())
        plt.figure()
        plt.imshow(s)
        plt.show()

if __name__ == '__main__':
    test_GridHardRGBGoalAllRandom()


