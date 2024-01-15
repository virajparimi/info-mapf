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
    location_coords = []
    for location_id in location_ids:
        location_coords.append(map.get_coordinate(location_id))

    locations = []
    for location_id in range(0, map.map_size):
        for location_coord in location_coords:
            distance = float(
                np.linalg.norm(location_coord - map.get_coordinate(location_id))
            )
            if distance <= radius:
                locations.append(location_id)
                break
    locations = list(set(locations))
    return locations
