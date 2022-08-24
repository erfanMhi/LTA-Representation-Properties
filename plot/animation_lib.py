import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patches
# from matplotlib.image import AxesImage, Image

from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage


layer_color = '#bdc3c7'
cnn_layer_color = '#bdc3c7'
active_neuron = '#34495e'
deactive_neuron = 'w'

class Neuron(patches.Circle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def activate(self):
        self.set_color(active_neuron)

    def deactivate(self):
        self.set_color(deactive_neuron)


class Layer:
    def __init__(self, xy, num_neurons, radius=0.025, margin=0.005):
        self.num_neurons = num_neurons
        diameter = radius*2
        self.radius = radius
        self.height = diameter * num_neurons + margin*(num_neurons+1)
        self.width =  diameter + margin*2
        self.rect = patches.Rectangle(xy, self.width, self.height, color=layer_color, zorder=1)
        x, y = xy
        self.neurons = []
        for n in range(num_neurons):
            neuron = Neuron((x+margin+radius, y + (2*n+1)*radius + (n+1)*margin), radius,
                                color=deactive_neuron, ec='k', zorder=4)
            self.neurons.append(neuron)
    
    def add_patches(self, ax):
        ax.add_patch(self.rect)
        for neuron in self.neurons:
            ax.add_patch(neuron)
    
    def add_arrow(self, ax):
        assert self.num_neurons == 4
        radius = self.radius
        xs = [0, - 0.6*radius, 0, 0.6*radius]
        ys = [- 0.6*radius, 0, 0.6*radius, 0]
        dxs = [0, 0.4*radius*2, 0, -0.4*radius*2]
        dys = [0.4*radius*2, 0, -0.4*radius*2, 0]
        for i, neuron in enumerate(self.neurons[::-1]):
            # print(neuron.get_center())
            x, y = neuron.get_center()
            radius = neuron.get_radius()

            ax.arrow(x+ xs[i], y+ys[i], dxs[i], dys[i], head_width=0.01, zorder=5)

    def get_neurons_cord(self):
        cords = []
        for neuron in self.neurons:
            # print(neuron.get_center())
            cords.append(neuron.get_center())

        return cords

    def activate_neuron(self, num):
        for neuron in self.neurons:
            # print(neuron.get_center())
            neuron.deactivate()
        self.neurons[num].activate()


class Connections:
    def __init__(self, out_layer, in_layer, weights=None):
        if weights is None:
            weights = np.random.rand(out_layer.num_neurons, in_layer.num_neurons)
        # Edges
        self.lines = []
        in_layer_cords = in_layer.get_neurons_cord()
        out_layer_cords = out_layer.get_neurons_cord()
        for i, out_neuron_cord in enumerate(out_layer_cords):
            for j, in_neuron_cord in enumerate(in_layer_cords):
                x_cords, y_cords = list(zip(*[out_neuron_cord, in_neuron_cord]))
                line = plt.Line2D(x_cords, y_cords, c='k', alpha=weights[i, j], zorder=2)
                self.lines.append(line)

        # for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        #     layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        #     layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        #     for m in range(layer_size_a):
        #         for o in range(layer_size_b):
        #             line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
        #                             [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
        #             lines.append(line)
        #             ax.add_artist(line)
    
    def add_artists(self, ax):
        for line in self.lines:
            ax.add_artist(line)

class TextBox:
    def __init__(self, xy, width, height, text):
        self.rect = patches.Rectangle(xy, width, height, color=cnn_layer_color, alpha=0.9, edgecolor="#2980b9", facecolor='none', capstyle='round')

        self.left = xy[0]
        self.right = xy[0] + width

        self.bottom = xy[1]
        self.top = xy[1] + height
        
        self.rect.set_clip_on(False)
        self.text = text
        
    def get_corners(self):
        return (self.left, self.bottom), (self.left, self.top), (self.right, self.top), (self.right, self.bottom) # clock-wise bottom-left, top-left, top-right, bottom-right 
    
    def add_textbox(self, ax):
        self.rect.set_transform(ax.transAxes)
        ax.add_patch(self.rect)
        ax.text(0.5 * (self.left + self.right), 0.5 * (self.bottom + self.top), self.text,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes, fontsize=20)


class Maze:
    def __init__(self, ax, xy, width, height, agent_position=(6, 6)):
        maze_row_size, maze_column_size = 15., 15.
        self.ax = ax
        self.width, self.height = width, height
        self.agent_position = agent_position
        self.agent_width=width/maze_row_size
        self.agent_height=height/maze_row_size
        x, y = xy
        self.x, self.y = x, y
        maze_img = Image.open('plot/figs/maze.png')
        self.maze_img_container = AxesImage(ax)
        self.maze_img_container.set(data=maze_img, extent=(x, x+width, y, y+height))

        agent_img = Image.open('plot/figs/agent.png')
        self.agent_img_container = AxesImage(ax, zorder=5)

        self.upper_xy = (x, y+height-self.agent_height)
        self.agent_img_container.set(data=agent_img)
        self.agent_position_update(self.agent_position)
        # self.agent_x, self.agent_y = upper_xy[0]+(self.agent_width*agent_position[1]), upper_xy[1]-(self.agent_height*agent_position[0])
        # self.agent_img_container.set(data=agent_img, extent=(self.agent_x, self.agent_x+self.agent_width, self.agent_y, self.agent_y+self.agent_height))

    def agent_position_update(self, position):
        self.agent_x, self.agent_y = self.upper_xy[0]+(self.agent_width*position[1]), self.upper_xy[1]-(self.agent_height*position[0])
        self.agent_img_container.set(extent=(self.agent_x, self.agent_x+self.agent_width, self.agent_y, self.agent_y+self.agent_height))

    def add_to_axis(self):
        self.ax.add_artist(self.maze_img_container)
        self.ax.add_artist(self.agent_img_container)

    def get_corners(self):
        return (self.x, self.y), (self.x, self.y + self.height), (self.x + self.width, self.y + self.height), (self.x + self.width, self.y) # clock-wise bottom-left, top-left, top-right, bottom-right 


    # def step(self, action):
    #     assert 0 <= action <= 3
        # if -1 < action < 15
