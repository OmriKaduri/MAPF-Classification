import glob
import os
import linecache


def mapname_from_file(mapfile):
    return mapfile.split('.map')[0].split('\\')[1]


def agents_from_scen_row(agent_row, num_agents):
    agent_data = agent_row.split()
    start_y = agent_data[4]
    start_x = agent_data[5]
    goal_y = agent_data[6]
    goal_x = agent_data[7]
    return ','.join([str(x) for x in [num_agents - 2, goal_x, goal_y, start_x, start_y]])


def generate_instance_from_scen(mapfile, scenfile, num_agents, base_file):
    instance_id = scenfile.split('.scen')[0].split('-')[-1]
    map_name = mapname_from_file(mapfile)
    if not os.path.isfile(base_file):
        print("ERROR! Can't find base file while trying to create instance!")
        return

    instance_file = 'instances/' + '-'.join([str(x) for x in [map_name, num_agents, instance_id]])

    with open(instance_file, 'w+') as instance, open(base_file, 'r') as base:
        instance.write(instance_id + ',' + map_name + '\n')
        for line in base:
            instance.write(line)
        instance.write(str(num_agents) + "\n")
        for i in range(2, num_agents + 2):
            instance.write(agents_from_scen_row(linecache.getline(scenfile, i), i) + '\n')
    # print("Creating instance",instance_file)


def create_baseinstance_from_map(mapfile):
    base_instance_file = 'base-maps/' + mapname_from_file(mapfile) + '-base'
    if os.path.isfile(base_instance_file):
        return base_instance_file

    with open(mapfile, 'r') as f, open(base_instance_file, 'w+') as instance:
        for index, line in enumerate(f):
            if index == 0:  # line should be 'type octile'
                pass  # The instance id can not be determined at the base
            if index == 1:  # line should be height VALUE
                instance.write('Grid:\n')
                height = line.split()[1]
            if index == 2:  # line should be weigth VALUE
                width = line.split()[1]
                instance.write(height + ',' + width + '\n')
            if index > 3:  # Skipping line index==3, should contain 'map'
                instance.write(line)
        instance.write('\nAgents:\n')

    return base_instance_file


def generate_instances_from_scen(map_file, scen_file):
    base_file = create_baseinstance_from_map(map_file)
    with open(scen_file) as scen:
        for index, line in enumerate(scen):
            if index == 0:
                assert ('version 1' in line)
            else:
                generate_instance_from_scen(map_file, scen_file, index, base_file)


for scen in glob.glob('../src/data/nathan/scen/scen-even/*'):
    for map in glob.glob('../src/data/nathan/maps/*'):
        map_name = mapname_from_file(map)
        if map_name in scen:
            generate_instances_from_scen(map, scen)
