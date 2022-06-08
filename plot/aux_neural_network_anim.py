import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patches
# from matplotlib.image import AxesImage, Image
from animation_lib import *
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage


# layer_color = '#bdc3c7'
# cnn_layer_color = '#bdc3c7'
# active_neuron = '#34495e'
# deactive_neuron = 'w'

# class Neuron(patches.Circle):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def activate(self):
#         self.set_color(active_neuron)

#     def deactivate(self):
#         self.set_color(deactive_neuron)


# class Layer:
#     def __init__(self, xy, num_neurons, radius=0.025, margin=0.005):
#         self.num_neurons = num_neurons
#         diameter = radius*2
#         self.radius = radius
#         self.height = diameter * num_neurons + margin*(num_neurons+1)
#         self.width =  diameter + margin*2
#         self.rect = patches.Rectangle(xy, self.width, self.height, color=layer_color, zorder=1)
#         x, y = xy
#         self.neurons = []
#         for n in range(num_neurons):
#             neuron = Neuron((x+margin+radius, y + (2*n+1)*radius + (n+1)*margin), radius,
#                                 color=deactive_neuron, ec='k', zorder=4)
#             self.neurons.append(neuron)
    
#     def add_patches(self, ax):
#         ax.add_patch(self.rect)
#         for neuron in self.neurons:
#             ax.add_patch(neuron)
    
#     def add_arrow(self, ax):
#         assert self.num_neurons == 4
#         radius = self.radius
#         xs = [0, - 0.6*radius, 0, 0.6*radius]
#         ys = [- 0.6*radius, 0, 0.6*radius, 0]
#         dxs = [0, 0.4*radius*2, 0, -0.4*radius*2]
#         dys = [0.4*radius*2, 0, -0.4*radius*2, 0]
#         for i, neuron in enumerate(self.neurons[::-1]):
#             # print(neuron.get_center())
#             x, y = neuron.get_center()
#             radius = neuron.get_radius()

#             ax.arrow(x+ xs[i], y+ys[i], dxs[i], dys[i], head_width=0.01, zorder=5)

#     def get_neurons_cord(self):
#         cords = []
#         for neuron in self.neurons:
#             # print(neuron.get_center())
#             cords.append(neuron.get_center())

#         return cords

#     def activate_neuron(self, num):
#         for neuron in self.neurons:
#             # print(neuron.get_center())
#             neuron.deactivate()
#         self.neurons[num].activate()


# class Connections:
#     def __init__(self, out_layer, in_layer, weights=None):
#         if weights is None:
#             weights = np.random.rand(out_layer.num_neurons, in_layer.num_neurons)
#         # Edges
#         self.lines = []
#         in_layer_cords = in_layer.get_neurons_cord()
#         out_layer_cords = out_layer.get_neurons_cord()
#         for i, out_neuron_cord in enumerate(out_layer_cords):
#             for j, in_neuron_cord in enumerate(in_layer_cords):
#                 x_cords, y_cords = list(zip(*[out_neuron_cord, in_neuron_cord]))
#                 line = plt.Line2D(x_cords, y_cords, c='k', alpha=weights[i, j], zorder=2)
#                 self.lines.append(line)

#         # for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
#         #     layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
#         #     layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
#         #     for m in range(layer_size_a):
#         #         for o in range(layer_size_b):
#         #             line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
#         #                             [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
#         #             lines.append(line)
#         #             ax.add_artist(line)
    
#     def add_artists(self, ax):
#         for line in self.lines:
#             ax.add_artist(line)

# class TextBox:
#     def __init__(self, xy, width, height, text):
#         self.rect = patches.Rectangle(xy, width, height, color=cnn_layer_color, alpha=0.9, edgecolor="#2980b9", facecolor='none', capstyle='round')

#         self.left = xy[0]
#         self.right = xy[0] + width

#         self.bottom = xy[1]
#         self.top = xy[1] + height
        
#         self.rect.set_clip_on(False)
#         self.text = text
        

    
#     def add_textbox(self, ax):
#         self.rect.set_transform(ax.transAxes)
#         ax.add_patch(self.rect)
#         ax.text(0.5 * (self.left + self.right), 0.5 * (self.bottom + self.top), self.text,
#         horizontalalignment='center',
#         verticalalignment='center',
#         transform=ax.transAxes, fontsize=20)


# class Maze:
#     def __init__(self, ax, xy, width, height, agent_position=(6, 6)):
#         maze_row_size, maze_column_size = 15., 15.
#         self.ax = ax
#         self.agent_position = agent_position
#         self.agent_width=width/maze_row_size
#         self.agent_height=height/maze_row_size
#         x, y = xy
#         self.x, self.y = x, y
#         maze_img = Image.open('figs/maze.png')
#         self.maze_img_container = AxesImage(ax)
#         self.maze_img_container.set(data=maze_img, extent=(x, x+width, y, y+height))

#         agent_img = Image.open('figs/agent.png')
#         self.agent_img_container = AxesImage(ax, zorder=5)

#         self.upper_xy = (x, y+height-self.agent_height)
#         self.agent_img_container.set(data=agent_img)
#         self.agent_position_update(self.agent_position)
#         # self.agent_x, self.agent_y = upper_xy[0]+(self.agent_width*agent_position[1]), upper_xy[1]-(self.agent_height*agent_position[0])
#         # self.agent_img_container.set(data=agent_img, extent=(self.agent_x, self.agent_x+self.agent_width, self.agent_y, self.agent_y+self.agent_height))

#     def agent_position_update(self, position):
#         self.agent_x, self.agent_y = self.upper_xy[0]+(self.agent_width*position[1]), self.upper_xy[1]-(self.agent_height*position[0])
#         self.agent_img_container.set(extent=(self.agent_x, self.agent_x+self.agent_width, self.agent_y, self.agent_y+self.agent_height))

#     def add_to_axis(self):
#         self.ax.add_artist(self.maze_img_container)
#         self.ax.add_artist(self.agent_img_container)

#     # def step(self, action):
#     #     assert 0 <= action <= 3
#         # if -1 < action < 15


# def draw_neural_net:
def main(left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.gca()
    fig, ax = plt.subplots(figsize=(18, 18), nrows=1, ncols=1)
    rec_left, rec_width = .2, .25
    rec_bottom, rec_height = .37, .30
    rec_right = rec_left + rec_width
    rec_top = rec_bottom + rec_height

    CNN_xy = (0.3, 0.33)
    CNN_width= 0.2
    CNN_text = 'Convolutional\nLayers'
    NN_distance = 0.05

    # Adding the fallten layer on top of the convolutional layer
    flatten_layer_xy = (CNN_xy[0] + CNN_width, CNN_xy[1])
    flatten_layer = Layer(flatten_layer_xy, 6)
    flatten_layer.add_patches(ax)

    # Adding Convolutional Layer
    CNN_box = TextBox(CNN_xy, CNN_width, flatten_layer.height, CNN_text)
    CNN_box.add_textbox(ax)
    print(flatten_layer.height)


    # Adding representation layer
    
    representation_layer_xy = (flatten_layer_xy[0] + flatten_layer.width + NN_distance, flatten_layer_xy[1])
    representation_layer = Layer(representation_layer_xy, 6)
    representation_layer.add_patches(ax)

    # l_x = (representation_layer_xy[0]+ representation_layer.width/2, representation_layer_xy[0]+ representation_layer.width/2) 
    # l_y = (representation_layer_xy[1]-0.002, representation_layer_xy[1]-0.14)
    # line = plt.Line2D(l_x, l_y, c='k', linewidth=5, zorder=2)
    # ax.add_artist(line) 

    # ax.arrow(l_x[1], l_y[1], -0.06, 0, head_width=0.01, linewidth=5, zorder=5, color='black')

    # Flat to Rep connection
    flat_rep_connections= Connections(flatten_layer, representation_layer)
    flat_rep_connections.add_artists(ax)


    ######## Value Network #########


    val_layer_xy = (representation_layer_xy[0] + representation_layer.width + NN_distance*2, 0.55)
    val_layer= Layer(val_layer_xy, 5)
    val_layer.add_patches(ax)

    # rep value connection
    rep_value_connections = Connections(representation_layer, val_layer)
    rep_value_connections.add_artists(ax)

    val_output_xy = (val_layer_xy[0] + val_layer.width + NN_distance, 0.55)
    val_output = Layer(val_output_xy, 4)
    val_output_xy = (val_layer_xy[0] + val_layer.width + NN_distance, 0.55+(val_layer.height - val_output.height)/2)
    val_output = Layer(val_output_xy, 4)
    val_output.add_patches(ax)

    val_output.add_arrow(ax)

    val_output_connections = Connections(val_layer, val_output)
    val_output_connections.add_artists(ax)

    ###### Aux Network #######



    
    aux_layer_xy = (representation_layer_xy[0] + representation_layer.width + NN_distance*2, 0.45 - val_layer.height)
    aux_layer = Layer(aux_layer_xy, 5)
    aux_layer.add_patches(ax)


    rep_aux_connections = Connections(representation_layer, aux_layer)
    rep_aux_connections.add_artists(ax)


    aux_output_xy = (aux_layer_xy[0] + aux_layer.width + NN_distance, 0.55)
    aux_output = Layer(aux_output_xy, 1)
    aux_output_xy = (aux_layer_xy[0] + aux_layer.width + NN_distance, aux_layer_xy[1] + (aux_layer.height - aux_output.height)/2)
    aux_output = Layer(aux_output_xy, 1)
    aux_output.add_patches(ax)


    aux_output_connections = Connections(aux_layer, aux_output)
    aux_output_connections.add_artists(ax)


    ######## Maze #######
    maze_size = 0.25
    text_x = (CNN_xy[0]-0.275)+maze_size/2
    text_y = 0.5+maze_size/2+0.02
    step_text = ax.text(text_x, text_y, 'Step: 0',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes, fontsize=25)

    maze_xy = (CNN_xy[0]-0.275, 0.5-maze_size/2)
    maze = Maze(ax, maze_xy, maze_size, maze_size)
    maze.add_to_axis()
    
    ######## Header Episode Text ########
    episode_text = ax.text(0.5, 0.95, 'Episode: 0',
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax.transAxes, fontsize=30)
    # agent_size=maze_size/15.
    # maze_img = Image.open('figs/maze.png')
    # # maze_img = np.array(maze_img).astype(np.float)
    # print(maze_img)
    # maze_img_container = AxesImage(ax)
    # maze_img_container.set(data=maze_img, extent=(CNN_xy[0]-0.275, CNN_xy[0]-0.275+maze_size, 0.5-maze_size/2, 0.5+maze_size/2))
    # ax.add_artist(maze_img_container)
    # agent_img = Image.open('figs/agent.png')
    # agent_img_container = AxesImage(ax, zorder=5)
    # agent_img_container.set(data=agent_img, extent=(CNN_xy[0]-0.275, CNN_xy[0]-0.275+agent_size, 0.5-agent_size/2, 0.5+agent_size/2))
    # ax.add_artist(agent_img_container)

    # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    # AxesImage(ax)
    # ax.add_artist( #ax can be added image as artist.
    #     AnnotationBbox(
    #         OffsetImage(img, zoom=0.3)
    #         , (0.1, 0.1)
    #         , frameon=False
    #     ) 
    # )
    # newax = fig.add_axes([CNN_xy[0]-0.275, 0.7-maze_size, maze_size, maze_size], anchor='NE', zorder=10)

    # newax.axis('off')
    # newax.imshow(img)
    # plt.imshow(img)

    # rect = patches.Rectangle((rec_left, rec_bottom), rec_width, rec_height, color="#3498db", edgecolor="#2980b9", facecolor='none', capstyle='round', linewidth=5)
    # rect.set_transform(ax.transAxes)
    # rect.set_clip_on(False)
    # ax.add_patch(rect)
    # ax.text(0.5 * (rec_left + rec_right), 0.5 * (rec_bottom + rec_top), 'middle',
    #         horizontalalignment='center',
    #         verticalalignment='center',
    #         transform=ax.transAxes)

    # layer_1 = Layer((0.1, 0.1), 6)
    # layer_1.add_patches(ax)

    # layer_2 = Layer((0.2, 0.1), 4)
    # layer_2.add_patches(ax)
    # x, y = list(zip(*layer_1.get_neurons_cord()))
    # plt.scatter(x, y, zorder=10)
    # x, y = list(zip(*layer_2.get_neurons_cord()))
    # plt.scatter(x, y, zorder=10)

    # connections= Connections(layer_1, layer_2)
    # connections.add_artists(ax)


    # n_layers = len(layer_sizes)
    # v_spacing = (top - bottom)/float(max(layer_sizes))
    # h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # circles = []
    # print('v_spacing: ', v_spacing)
    # # Nodes
    # for n, layer_size in enumerate(layer_sizes):
    #     layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
    #     for m in range(layer_size):
    #         circle = Neuron((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
    #                             color='w', ec='k', zorder=4)
    #         circles.append(circle)
    #         ax.add_artist(circle)

    # # Edges
    # lines = []
    # for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    #     layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
    #     layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
    #     for m in range(layer_size_a):
    #         for o in range(layer_size_b):
    #             line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
    #                             [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
    #             lines.append(line)
    #             ax.add_artist(line)

    lines = flat_rep_connections.lines + rep_value_connections.lines + val_output_connections.lines + rep_aux_connections.lines + aux_output_connections.lines

    action_list = np.load('anim_data/agent_action_list.npy')
    agent_pos_list = np.load('anim_data/agent_pos_list.npy')
    steps = np.load('anim_data/step_num_list.npy')
    episodes = np.load('anim_data/episode_num.npy')

    item_list = [agent_pos_list, steps, episodes, action_list]
    items = [[] for i in range(len(item_list))]
    for i in range(len(episodes)):
        if episodes[i] % 100 == 0 and episodes[i] < 1200:
            for j in range(len(item_list)):
                items[j].append(item_list[j][i])

    agent_pos_list, steps, episodes, action_list = items
    action_map = [2, 0, 1, 3]

    def init():
        return lines + [maze.agent_img_container] + val_output.neurons

    def animate(i):
        for line in lines:
            line.set(alpha=np.random.rand())
        maze.agent_position_update(agent_pos_list[i])
        val_output.activate_neuron(action_map[action_list[i]])
        episode_text.set_text('Episode: {}'.format(episodes[i]))
        step_text.set_text('Step: {}'.format(steps[i]))
        return lines + [maze.agent_img_container] + val_output.neurons + [episode_text, step_text]
        
    print('Frames num: ', len(action_list))
    anim = FuncAnimation(fig, animate, init_func = init,
                    frames = len(action_list), interval = 20, blit = True)
    # ax.axis('off')
    #ax.margins(x=0)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    anim.save('here.mp4', writer = 'ffmpeg', fps = 10)

# print(type(draw_neural_net))
main(.6, .9, .1, .9, [4, 7, 2])
# fig.savefig('nn.png')