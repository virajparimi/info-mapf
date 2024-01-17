import numpy as np
from map import Map
from typing import Any, List
from numpy.typing import NDArray


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
