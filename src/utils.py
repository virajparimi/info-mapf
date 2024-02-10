import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Tuple, Union
from map import Grid, RewardMap, Parameters


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
    location_ids: List[int], rows: int, cols: int, radius: float = 1.0
) -> List[int]:
    """
    Returns the nearest locations to a given set of locations
    :param location_ids: Linearized list of locations to find the nearest locations to
    :param rows: Number of rows in the grid
    :param cols: Number of columns in the grid
    :param radius: Radius to search for nearest locations
    """

    locations = np.arange(rows * cols)
    location_rows = locations // cols  # y
    location_columns = locations % cols  # x
    location_coords = np.column_stack((location_rows, location_columns))

    location_id_rows = np.array(location_ids) // cols  # y
    location_id_columns = np.array(location_ids) % cols  # x
    location_id_coords = np.column_stack((location_id_rows, location_id_columns))

    distances = np.linalg.norm(
        location_coords[:, np.newaxis, :] - location_id_coords, axis=2
    )
    indices = np.any(distances <= radius, axis=1)
    return locations[indices.astype(np.bool_)].tolist()


def generate_map(
    rows: int,
    columns: int,
    grid: Union[NDArray[Any], None] = None,
    agent_locations: Union[List[Tuple[int, int]], None] = None,
    gp_means: Union[List[int], None] = None,
    gp_locations: Union[List[Tuple[int, int]], None] = None,
    parameters: Union[Parameters, None] = None,
) -> Tuple[Grid, RewardMap]:
    """
    Generates a random grid and reward map of size rows x columns
    :param rows: Number of rows
    :param columns: Number of columns
    :param grid: Grid to use where 1 is free space and 0 is a wall
    :param agent_locations: Locations of agents
    :param gp_means: Means of Gaussians for different locations
    :param gp_locations: Locations of Gaussians corresponding to the means
    :param parameters: Parameters for the map, GP and Vulcan to use
    """

    grid = (
        np.ones((rows, columns), dtype=np.bool_)
        if grid is None
        else np.array(grid, dtype=np.bool_)
    )

    agent_locations = [(0, 0)] if agent_locations is None else agent_locations
    for agent_location in agent_locations:
        grid[agent_location[0], agent_location[1]] = False

    gp_means = [1] if gp_means is None else gp_means
    gp_locations = [(rows // 2, columns // 2)] if gp_locations is None else gp_locations

    reward_map = RewardMap(
        rows, columns, means=gp_means, locations=gp_locations, params=parameters
    )
    return Grid(grid), reward_map
