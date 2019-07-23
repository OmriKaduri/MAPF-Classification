import sys


def left_edge(row_index, width, cell_index, direction='out'):
    if direction == 'out':
        return (row_index * width + cell_index), (row_index * width + cell_index - 1)
    else:
        return (row_index * width + cell_index - 1), (row_index * width + cell_index)


def right_edge(row_index, width, cell_index, direction='out'):
    if direction == 'out':
        return (row_index * width + cell_index), (row_index * width + cell_index + 1)
    else:
        return (row_index * width + cell_index + 1), (row_index * width + cell_index)


def bottom_edge(row_index, width, cell_index, direction='out'):
    if direction == 'out':
        return (row_index * width + cell_index), ((row_index + 1) * width + cell_index)
    else:
        return ((row_index + 1) * width + cell_index), (row_index * width + cell_index)


def top_edge(row_index, width, cell_index, direction='out'):
    if direction == 'out':
        return (row_index * width + cell_index), ((row_index - 1) * width + cell_index)
    else:
        return ((row_index - 1) * width + cell_index), (row_index * width + cell_index)


def mark_cell_as_obstacle(cell_index, row_index, grid_size, graph):
    graph.add_node(row_index * grid_size[1] + cell_index, color='red', size=3)
    try:
        if graph.has_edge(*left_edge(row_index, grid_size[1], cell_index, direction='in')):  # Remove edge from left
            graph.remove_edge(*left_edge(row_index, grid_size[1], cell_index, direction='in'))
        if graph.has_edge(*top_edge(row_index, grid_size[1], cell_index, direction='in')):  # Remove edge from up
            graph.remove_edge(*top_edge(row_index, grid_size[1], cell_index, direction='in'))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(cell_index, row_index)
        print("Tried to remove an edge already removed")


def mark_cell_as_free(cell_index, row_index, grid_size, graph):
    graph.add_node(row_index * grid_size[1] + cell_index, color='white', size=3)
    if cell_index > 0 and graph.has_edge(
            *left_edge(row_index, grid_size[1], cell_index, direction='in')):  # Create Edge to left
        graph.add_edge(*left_edge(row_index, grid_size[1], cell_index), weight=3)
    if cell_index < grid_size[1] - 1:  # Create Edge to Right
        graph.add_edge(*right_edge(row_index, grid_size[1], cell_index), weight=3)
    if row_index < grid_size[0] - 1:  # Create Edge to Bottom
        graph.add_edge(*bottom_edge(row_index, grid_size[1], cell_index), weight=3)
    if row_index > 0 and graph.has_edge(
            *top_edge(row_index, grid_size[1], cell_index, direction='in')):  # Create Edge to Top
        graph.add_edge(*top_edge(row_index, grid_size[1], cell_index), weight=3)


def agent_points_from(metadata, grid_width):
    goal_x = int(metadata[1])
    goal_y = int(metadata[2])
    start_x = int(metadata[3])
    start_y = int(metadata[4])

    return start_x * grid_width + start_y, goal_x * grid_width + goal_y


def rotate_positions_90_clockwise(x, y):
    return y, -x
