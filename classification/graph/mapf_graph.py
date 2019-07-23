from graph_utils import mark_cell_as_free, mark_cell_as_obstacle, agent_points_from, \
    rotate_positions_90_clockwise

import networkx as nx
import matplotlib.pyplot as plt


class MapfGraph:
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

    def create_graph(self):
        with open(self.filename) as f:
            for index, line in enumerate(f):
                if index == 0:
                    self.instance = line.split(',')[0]
                    # Done for cases when the instance is of the format X,MORETEXT => X
                elif index == 2:
                    self.grid_size = [int(x) for x in line.split(',')]  # HeightXWidth dimensions
                    self.G = nx.DiGraph()
                elif 2 < index < self.grid_size[0] + 3:
                    # Do for all lines representing the grid. grid_size[1] is the grid width
                    for cell_index, cell in enumerate(line):
                        if cell == '.':
                            mark_cell_as_free(cell_index, index - 3, self.grid_size, self.G)
                        elif cell == '@':
                            mark_cell_as_obstacle(cell_index, index - 3, self.grid_size, self.G)
                            self.num_obstacles += 1

                elif index == self.grid_size[0] + 4:  # Number of agents line
                    self.num_agents = int(line)

                elif index > self.grid_size[0] + 4:
                    agent_data = line.split(',')
                    start, goal = agent_points_from(agent_data, self.grid_size[1])
                    self.G.add_node(goal, color="blue", size=3)
                    self.G.add_node(start, color="green", size=3)
                    paths = nx.astar_path(self.G, start, goal, weight='weight')
                    self.agent_sps.append([p for p in paths])

            for sp in self.agent_sps:
                path_edges = list(zip(sp, sp[1:]))
                for edge in path_edges:
                    node_from, node_to = edge
                    self.G.add_edge(*edge, weight=self.G[node_from][node_to]['weight'] + 3)
                    # THIS MUST BE DONE AFTER ALL SHORTEST PATHS COMPUTED

    def save_graph_to(self, filename):
        nx.write_edgelist(self.G, filename)

    def draw_graph_to(self, filename):
        normal_pos = dict((n, (n // self.grid_size[1], n % self.grid_size[1])) for n in self.G.nodes())
        pos = {k: rotate_positions_90_clockwise(*v) for k, v in normal_pos.items()}
        # 90 degree rotation of positions is done because Networkx way of visualization
        color_map = [n[1]['color'] for n in self.G.nodes(data=True)]
        node_size = [n[1]['size'] for n in self.G.nodes(data=True)]
        weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]

        print("Visualizing grid of size", self.grid_size, self.num_agents, "Agents and", self.num_obstacles,
              "Obstacles", "Instance", self.instance)
        plt.figure(figsize=(15, 15))

        nx.draw(self.G, pos=pos, font_size=10, node_size=node_size,
                node_color=color_map, width=weights)

        plt.savefig(filename, format="jpg")
        plt.close('all')  # Used in order to not override previous plot. VERY IMPORTANT
