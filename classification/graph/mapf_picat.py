from graph_utils import mark_cell_as_free, mark_cell_as_obstacle, agent_points_from, \
    rotate_positions_90_clockwise

import networkx as nx


class MapfPicat:
    def __init__(self, mapf_instance_filename):
        self.filename = mapf_instance_filename
        self.instance = -1
        self.grid_size = [-1, -1]
        self.num_agents = -1
        self.color_map = []
        self.node_size = []
        self.weights = []
        # self.labels = {}
        self.num_obstacles = 0
        self.G = nx.empty_graph()
        self.agent_sps = []
        self.agents = []

    def create_graph(self):  # Should be used for .map files
        with open(self.filename, newline='') as f:
            free_index = 1
            for index, line in enumerate(f):
                # free_index += index * self.grid_size[0]
                if index == 1:
                    height = int(line.split(' ')[1])

                elif index == 2:
                    width = int(line.split(' ')[1])
                    self.grid_size = [height, width]  # HeightXWidth dimensions
                    self.G = nx.DiGraph()

                elif 3 < index < self.grid_size[0] + 4:
                    # Do for all lines representing the grid. grid_size[1] is the grid width
                    for cell_index, cell in enumerate(line):
                        if cell == '.':
                            mark_cell_as_free(cell_index, index - 4, self.grid_size, self.G, free_index)
                            free_index += 1
                        elif cell == '@' or cell == 'T':
                            mark_cell_as_obstacle(cell_index, index - 4, self.grid_size, self.G)
                            self.num_obstacles += 1

                elif index == self.grid_size[0] + 4:  # Number of agents line
                    self.num_agents = int(line)

                elif index > self.grid_size[0] + 4:
                    agent_data = line.split(',')
                    start, goal = agent_points_from(agent_data, self.grid_size[1])
                    self.agents.append((start, goal))

    def write_neibs_to_picat(self, file):
        neibs = {}
        for index, (g_node, data) in enumerate(self.G.nodes(data=True)):
            if data['color'] != 'white':
                continue

            g_node_neibs = self.G[g_node]  # all neighbors are free
            if len(g_node_neibs) == 0:
                continue
            neibs[g_node] = [data['free_index']]
            for neib in g_node_neibs:
                neib_index = self.G.nodes[neib]['free_index']
                neibs[g_node].append(neib_index)

            file.write(
                f'\n    $neibs({data["free_index"]},{str(neibs[g_node]).replace(" ", "")})')
            if (index + 1 < len(self.G)):
                file.write(',')

            for neib in g_node_neibs:
                neibs[g_node].append(neib)
        file.write('\n    ],')

        return neibs

    def write_agents_to_picat(self, file):
        for agent in self.agents:
            print(agent)

    def save_graph_to(self, filename):
        nx.write_edgelist(self.G, filename)

    def save_gexf_graph_to(self, filename):
        nx.write_gexf(self.G, filename)
