import numpy as np
import matplotlib.pyplot as plt

from core.utils.torch_utils import random_seed
from core.environment.collectworlds import CollectXY, CollectRGB

class CollectRandomXY(CollectXY):
    "Block positions are randomized"
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        random_seed(seed)

    def reset(self):
        for k in range(len(self.object_coords)):
            while True:
                rx, ry = np.random.randint(low=0, high=15, size=2)
                if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
                        not (rx, ry) in self.object_coords:
                    self.object_coords[k] = (rx, ry)
                    break
        return super().reset()


class CollectRandomRGB(CollectRGB):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

    def reset(self):
        for k in range(len(self.object_coords)):
            while True:
                rx, ry = np.random.randint(low=0, high=15, size=2)
                if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y) and \
                        not (rx, ry) in self.object_coords:
                    self.object_coords[k] = (rx, ry)
                    break

        return super().reset()

    def get_features(self, state):
        raise NotImplementedError

    def get_visualization_segment(self):
        raise NotImplementedError


if __name__ == '__main__':
    # env = CollectXY()
    # state = env.reset()
    # print('State: ', state)
    # done = False
    # while not done:
    #     action = int(input('input_action: '))
    #     state, reward, done, _ = env.step([action])
    #     print(state, ' ', reward, ' ', done)


    def draw(state):
        frame = state.astype(np.uint8)
        figure, ax = plt.subplots()
        ax.imshow(frame)
        plt.axis('off')
        plt.show()
        plt.close()

    env = CollectRandomRGB()

    state = env.reset()
    draw(state)
    done = False
    while not done:
        try:
            action = int(input('input_action: '))
        except ValueError:
            action = 0
        state, reward, done, _ = env.step([action])
        print(reward, ' ', done)
        draw(state)

