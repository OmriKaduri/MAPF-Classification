import glob
import os
import linecache
import pandas as pd

from mapf_picat import MapfPicat


def mapname_from_file(mapfile):
    return mapfile.split('.map')[0].split('/')[-1]


def agent_points_from(agent_row, grid_width):
    agent_data = agent_row.split()
    start_y = int(agent_data[4])
    start_x = int(agent_data[5])
    goal_y = int(agent_data[6])
    goal_x = int(agent_data[7])

    return start_x * grid_width + start_y, goal_x * grid_width + goal_y


def generate_instance_from_scen(mapfile, scenfile, num_agents, base_file, grid_size, mapf_graph):
    instance_id = scenfile.split('.scen')[0].split('-')[-1]
    map_name = mapname_from_file(mapfile)
    if not os.path.isfile(base_file):
        print("ERROR! Can't find base file while trying to create instance!")
        return

    picat_file = 'picats/' + '-'.join([str(x) for x in [map_name, num_agents, instance_id]]) + '.pi'

    with open(picat_file, 'w+', newline="") as picat, open(base_file, 'r') as base:
        # for line in base:
        #     picat.write(line)
        # elif index > self.grid_size[0] + 4:
        # agent_data = line.split(',')
        # start, goal = agent_points_from(agent_data, self.grid_size[1])
        # self.agents.append((start + 1, goal + 1))
        # instance.write(str(num_agents) + "\n")
        agents = []

        for i in range(2, num_agents + 2):
            agent = agent_points_from(linecache.getline(scenfile, i), grid_size[1])
            orig_start, orig_end = agent[0], agent[1]
            start, end = mapf_graph.G.nodes[orig_start]['free_index'], mapf_graph.G.nodes[orig_end]['free_index']
            agents.append((start, end))

        picat.write(f'    As = {[agent for agent in agents]},')
        picat.write('\n    Avoid = new_array(0, 0),')
        picat.write('\n    Makespan = -1,')
        picat.write('\n    SumOfCosts = -1.\n')

    # print("Creating instance",instance_file)


def basepicat_from_map(mapfile):
    base_instance_file = 'base-picats/' + mapname_from_file(mapfile) + '-base'

    with open(base_instance_file, 'w+') as picat_file:
        graph = MapfPicat(mapfile)
        graph.create_graph()
        picat_file.write('ins(Graph, As, Avoid, Makespan, SumOfCosts) =>\n')
        picat_file.write('    Graph = [')
        graph.write_neibs_to_picat(picat_file)  # No longer than 200ms operation (on 256x256 graph)
        grid_size = graph.grid_size
    return base_instance_file, grid_size, graph


# def generate_instances_from_scen(map_file, scen_file):
#     base_file, grid_size = basepicat_from_map(map_file)
#     with open(scen_file) as scen:
#         for index, line in enumerate(scen):
#             if index == 0:
#                 assert ('version 1' in line)
#             else:
#                 generate_instance_from_scen(map_file, scen_file, index, base_file, grid_size)


df = pd.read_csv('../data/from-vpn/AllData-labelled.csv')

maps_dir = '../data/from-vpn/maps/'
scen_dir = '../data/from-vpn/scen/scen-even/'


def generate_picat_from_mapf(mapf_problem):
    map_name = maps_dir + mapf_problem.GridName + '.map'
    scen_name = ''.join([str(x) for x in [
        scen_dir,
        mapf_problem.GridName,
        '-even-',
        mapf_problem.InstanceId,
        '.scen']])  # Berlin_1_256-even-1.scen

    basepicat, grid_size, mapf_graph = basepicat_from_map(map_name)
    generate_instance_from_scen(map_name, scen_name, mapf_problem.NumOfAgents, basepicat, grid_size, mapf_graph)
    # print(map_name, scen_name)


df.apply(generate_picat_from_mapf, axis=1)
# for scen in glob.glob('../src/data/nathan/scen/used-scen/*'):
#     for map in glob.glob('../src/data/nathan/maps/*'):
#         map_name = mapname_from_file(map)
#         if map_name in scen:
#             generate_instances_from_scen(map, scen)
