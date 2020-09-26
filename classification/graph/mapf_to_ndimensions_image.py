import glob
import os

# with open("data/from-azure/kiva/Instances/kiva_0-35-0", "r") as f:
# with open("data/from-azure/big/Instances/Instance-60-30-70-0", "r") as f:
from mapf_graph import MapfGraph

# This script is used to transform a mapf problem to a numpy file of the shape [N+1,H,W]
# where N is the number of agents, H and W are the grid height and width
# This should be used as the input for a CNN later.
# For each agent channel, all cells on its shortest path have the value of the time it reaches there.
# The N+1 channel is a binary image representing the map without agents.

mapf_dir = 'AllData'
# output_dir = '../edgelists/' + mapf_dir + '/'
output_dir = 'nd-images/'
# for file in glob.glob('../data/from-azure/' + mapf_dir + '/*'):
for file in glob.glob('instances/*'):
    if 'current' in file:
        continue
    filename = file.split('\\')[-1]
    if os.path.isfile(output_dir + filename +'.npz'):
        print(filename, "Already generated, moving on...")
        continue
    print("Working on ", filename)
    graph = MapfGraph(file)
    graph.create_graph()
    graph.draw_nd_image_to(output_dir + filename)
    # graph.draw_graph_to(output_dir + filename + ".png")
    # graph.save_gexf_graph_to(output_dir + filename + ".gexf")
