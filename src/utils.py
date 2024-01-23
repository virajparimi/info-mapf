import numpy as np
from map import Map, Parameters
from numpy.typing import NDArray
from typing import Any, List, Union


def positive_definite_matrix(A: NDArray[Any]):
    """
    Checks if a matrix is positive definite
    :param A: Matrix to check
    """
    if np.allclose(A, A.T):  # Check if the matrix is symmetric
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def get_nearest_locations(
    location_ids: List[int], map: Map, radius: float = 1.0
) -> List[int]:
    """
    Returns the nearest locations to a given set of locations
    :param location_ids: Linearized list of locations to find the nearest locations to
    :param map: Map to query
    :param radius: Radius to search for nearest locations
    """

    locations = np.arange(map.map_size)
    location_rows = locations // map.num_of_cols  # y
    location_columns = locations % map.num_of_cols  # x
    location_coords = np.column_stack((location_rows, location_columns))

    location_id_rows = np.array(location_ids) // map.num_of_cols  # y
    location_id_columns = np.array(location_ids) % map.num_of_cols  # x
    location_id_coords = np.column_stack((location_id_rows, location_id_columns))

    distances = np.linalg.norm(
        location_coords[:, np.newaxis, :] - location_id_coords, axis=2
    )
    indices = np.any(distances <= radius, axis=1)
    linear_locations = np.arange(map.map_size)
    return list(
        set(linear_locations[i] for i in range(len(linear_locations)) if indices[i])
    )


def generate_map(
    rows: int,
    columns: int,
    maze: Union[NDArray[Any], None] = None,
    agent_locations: Union[List[List[int]], None] = None,
    gp_means: Union[List[int], None] = None,
    gp_locations: Union[List[List[int]], None] = None,
    parameters: Union[Parameters, None] = None,
) -> Map:
    """
    Generates a random maze of size rows x columns
    :param rows: Number of rows
    :param columns: Number of columns
    :param maze: Maze to use where 1 is free space and 0 is a wall
    :param agent_locations: Locations of agents
    :param gp_means: Means of Gaussians for different locations
    :param gp_locations: Locations of Gaussians corresponding to the means
    :param parameters: Parameters for the map, GP and Vulcan to use
    """
    if maze is None:
        maze = np.ones((rows, columns), dtype=np.bool_)
    else:
        maze = np.array(maze, dtype=np.bool_)

    if agent_locations is None:
        center = [0, 0]
        agent_locations = [center]

    for agent_location in agent_locations:
        maze[agent_location[0], agent_location[1]] = False

    if gp_means is None:
        gp_means = [1]
    if gp_locations is None:
        gp_locations = [[rows // 2, columns // 2]]

    map = Map(maze, means=gp_means, locations=gp_locations, params=parameters)
    return map
