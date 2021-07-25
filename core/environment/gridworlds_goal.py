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

if __name__ == '__main__':

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

