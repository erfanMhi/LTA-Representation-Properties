#Tool imports
import numpy as np
import pandas as pd
import networkx as nx

#Graphic imports
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
#Format of weightlist 
weights = [[0.8, 0.7, 0.56], [0.5, -0.2, 1.4]] #each epoch must have a 1d array of weight values starting from w1....wn 

#print(weights)
layers = [2, 3, 3, 1] #Layers in our neural network

'''
Add the neural network in a dictionary and specifu it's x and y position in graph. If not specified nx will assign it's own position 
in random order
'''
nodes = {'ip1': (10, 40), 'ip2':(10, 60), 'b1': (10, 5), 'a1': (20, 32), 'a2': (20, 52), 'a3': (20, 72),
         'b2':(20, 5), 'a4': (30, 32), 'a5': (30, 52), 'a6': (30, 72), 'b3': (30, 5), 'op': (35, 52)}

G = nx.MultiGraph() #Create an empty graph with no nodes and edges

G.add_nodes_from(nodes.keys(), color='r') #Add the nodes from the dictionary

#Add the positions of the nn from the graph
for n, p in nodes.items():
    G.nodes[n]['pos'] = p

#Connect the egdes with thier respective nodes manually
G.add_edge('ip1', 'a1', color='r', weight = 2)
G.add_edge('ip1', 'a2', color='r',  weight = 2)
G.add_edge('ip1', 'a3', color='r',  weight = 4)
G.add_edge('ip2', 'a1', color='r',  weight = 4)
G.add_edge('ip2', 'a2', color='r',  weight = 6)
G.add_edge('ip2', 'a3', color='r',  weight = 6)
G.add_edge('b1', 'a1', color='r', weight = 2)
G.add_edge('b1', 'a2', color='r',  weight = 2)
G.add_edge('b1', 'a3', color='r',  weight = 4)
G.add_edge('a1', 'a4', color='r',  weight = 4)
G.add_edge('a1', 'a5', color='r',  weight = 6)
G.add_edge('a1', 'a6', color='r',  weight = 6)
G.add_edge('a2', 'a4', color='r', weight = 2)
G.add_edge('a2', 'a5', color='r',  weight = 2)
G.add_edge('a2', 'a6', color='r',  weight = 4)
G.add_edge('a3', 'a4', color='r',  weight = 4)
G.add_edge('a3', 'a5', color='r',  weight = 6)
G.add_edge('a3', 'a6', color='r',  weight = 6)
G.add_edge('b2', 'a4', color='r', weight = 2)
G.add_edge('b2', 'a5', color='r',  weight = 2)
G.add_edge('b2', 'a6', color='r',  weight = 4)
G.add_edge('a4', 'op', color='r',  weight = 4)
G.add_edge('a5', 'op', color='r',  weight = 6)
G.add_edge('a6', 'op', color='r',  weight = 6)
G.add_edge('b3', 'op', color='r',  weight = 6)


def color_mapper_node(x):
 '''
 Args: Take node values as inputs and assigns colour to it
 '''
 c_map = []
 for i in x:
  if i == 'b1' or  i == 'b2' or i == 'b3':
   c_map.append('red')
  elif i == 'ip1' or  i == 'ip2':
   c_map.append('pink')
  elif i == 'op':
   c_map.append('yellow')
  else:
   c_map.append('blue')
 return c_map


def color_mapper_edge(x):
 '''
 Takes edges as inputs and assigns colour of 
 different intensity based on the weight value
 '''
 edge_map = []
 for i in x:
  i = np.abs(i)
  if i >= 0.70:
   edge_map.append('#00ff11')
  elif i >= 0.5 and i < 0.75:
   edge_map.append('#7cfc84')
  elif i >= 0.25 and i < 0.5:
   edge_map.append('#b0ffb5')
  else:
   edge_map.append('#d2fad4')
 return edge_map


def status_edge():
 '''
 Generator to yield colour value for each epoch
 during animation
 '''
 for w in weights:
  yield color_mapper_edge(w)

edge_map = status_edge() 


def draw_next_status(n):
 '''
 Function that draws based on colour value it 
 gets from the colour mapper function
 '''
 plt.cla()
 e_map = next(edge_map)
 nx.draw(G, nodes, node_color=color_mapper_node(nodes.keys()), edge_color=e_map, width=4, with_labels=True)


# Matplotlib animation function


ani = animation.FuncAnimation(plt.gcf(), draw_next_status, interval=1000, frames=20, repeat= False)
plt.show()


'''
Before you save your animation change
matplotlib.use('TKAgg') to matplotlib.use('Agg')
'''

#writer = animation.PillowWriter(fps = 2)
#ani.save('Path/neuralnetwork_animate.gif', writer=writer