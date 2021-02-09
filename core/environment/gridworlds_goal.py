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
