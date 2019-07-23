import glob
import os

# with open("data/from-azure/kiva/Instances/kiva_0-35-0", "r") as f:
# with open("data/from-azure/big/Instances/Instance-60-30-70-0", "r") as f:
from mapf_graph import MapfGraph

mapf_dir = 'AllData'

for file in glob.glob('../data/from-azure/' + mapf_dir + '/*'):
    if 'current' in file:
        continue

    filename = file.split('\\')[-1]
    if os.path.isfile('../edgelists/' + mapf_dir + '/' + filename + ".png"):
        continue  # NOT REDO INSTANCES ALREADY EVALUATED

    print("Working on ", filename)
    graph = MapfGraph(file)
    graph.create_graph()
    graph.draw_graph_to('../edgelists/' + mapf_dir + '/' + filename + ".png")
    graph.save_graph_to('../edgelists/' + mapf_dir + '/' + filename + ".edgelists")
