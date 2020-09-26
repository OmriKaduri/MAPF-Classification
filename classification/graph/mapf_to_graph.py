import glob
import os

# with open("data/from-azure/kiva/Instances/kiva_0-35-0", "r") as f:
# with open("data/from-azure/big/Instances/Instance-60-30-70-0", "r") as f:
from mapf_graph import MapfGraph

mapf_dir = 'AllData'
# output_dir = '../edgelists/' + mapf_dir + '/'
output_dir = 'nathan-images/'
# for file in glob.glob('../data/from-azure/' + mapf_dir + '/*'):
for file in glob.glob('instances/*'):
    if 'current' in file:
        continue

    filename = file.split('\\')[-1]
    # if os.path.isfile('../edgelists/' + mapf_dir + '/' + filename + ".png"):
    #     continue  # NOT REDO INSTANCES ALREADY EVALUATED

    # if os.path.isfile('../edgelists/' + mapf_dir + '/' + filename + ".gexf"):
    #     continue  # NOT REDO INSTANCES ALREADY EVALUATED

    # if 'brc' in file or 'ost' in file or 'den' in file:
    #     continue

    print("Working on ", filename)
    graph = MapfGraph(file)
    graph.create_graph()
    graph.draw_graph_to(output_dir + filename + ".png")
    # graph.save_gexf_graph_to(output_dir + filename + ".gexf")
