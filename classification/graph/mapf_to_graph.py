import glob
import os

# with open("data/from-azure/kiva/Instances/kiva_0-35-0", "r") as f:
# with open("data/from-azure/big/Instances/Instance-60-30-70-0", "r") as f:
from mapf_graph import MapfGraph

mapf_dir = 'AllData'

for file in glob.glob('../data/from-azure/' + mapf_dir + '/*'):
    if 'current' in file:
        continue

    if 'Instance-110-10-80-0' not in file:
        continue
    filename = file.split('\\')[-1]
    # if os.path.isfile('../edgelists/' + mapf_dir + '/' + filename + ".png"):
    #     continue  # NOT REDO INSTANCES ALREADY EVALUATED

    # if os.path.isfile('../edgelists/' + mapf_dir + '/' + filename + ".gexf"):
    #     continue  # NOT REDO INSTANCES ALREADY EVALUATED

    if 'brc' in file or 'ost' in file or 'den' in file:
        continue

<<<<<<< HEAD
=======

>>>>>>> 4d8f07cd55d17bda7fb9aa0442a39a9056248d28
    print("Working on ", filename)
    graph = MapfGraph(file)
    graph.create_graph()
    # graph.draw_graph_to('../edgelists/' + mapf_dir + '/' + filename + ".png")
    graph.save_gexf_graph_to('../edgelists/' + mapf_dir + '/' + filename + ".gexf")
