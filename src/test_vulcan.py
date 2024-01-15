import numpy as np
from map import Map
from agent import Agent
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from typing import Union, List, Tuple


def generate_maze(
    rows: int,
    columns: int,
    agent_locations: Union[List[List[int]], None] = None,
    gp_means: Union[List[int], None] = None,
    gp_locations: Union[List[List[int]], None] = None,
) -> Tuple[NDArray[np.bool8], NDArray[np.float64], Map]:
    """
    Generates a random maze of size rows x columns
    :param rows: Number of rows
    :param columns: Number of columns
    :return: Map object
    """
    maze = np.ones((rows, columns), dtype=np.bool8)

    if agent_locations is None:
        center = [rows // 2, columns // 2]
        agent_locations = [center]

    for agent_location in agent_locations:
        maze[agent_location] = False

    if gp_means is None:
        gp_means = [1]
    if gp_locations is None:
        gp_locations = [[rows // 2, columns // 2]]

    map = Map(maze, means=gp_means, locations=gp_locations)

    grid = np.zeros(maze.shape)
    for location_means in range(0, len(map.means)):
        linearized_location = map.locations[location_means]
        location = map.get_coordinate(linearized_location)
        sample_locations = np.random.multivariate_normal(
            location, np.eye(2), size=(1000)
        )
        for sample in sample_locations:
            sample_x, sample_y = sample
            row = int(np.round(sample_x))
            column = int(np.round(sample_y))
            if 0 <= row < maze.shape[0] and 0 <= column < maze.shape[1]:
                grid[row, column] += 1.0 * map.means[location_means]
    return maze, grid, map


if __name__ == "__main__":
    maze, grid, map = generate_maze(5, 5)
    single_agent = Agent(id=1, start_location=0, map=map)

    single_agent.adaptive_search()

    path = []
    for observation in single_agent.mdp_handle.observations:
        location = observation.location
        path.append(map.get_coordinate(location))

    plt.plot([x[0] for x in path], [x[1] for x in path], "r-")
    plt.imshow(grid)
    plt.show()
