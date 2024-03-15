import numpy as np
from csv import reader
from copy import deepcopy
from pandas import DataFrame
from numpy.typing import NDArray
from scipy.interpolate import griddata
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
    location_ids: List[int],
    rows: int,
    cols: int,
    radius: float = 1.0,
    grid: Union[Grid, None] = None,
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

    if grid is not None:
        obstacle_mask = grid.obstacle_map[location_rows, location_columns].astype(
            np.bool_
        )
        return locations[indices.astype(np.bool_) & obstacle_mask].tolist()
    else:
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
    obstacle_map = deepcopy(grid)

    agent_locations = [(0, 0)] if agent_locations is None else agent_locations
    for agent_location in agent_locations:
        grid[agent_location[0], agent_location[1]] = False

    gp_means = [1] if gp_means is None else gp_means
    gp_locations = [(rows // 2, columns // 2)] if gp_locations is None else gp_locations

    reward_map = RewardMap(
        rows, columns, means=gp_means, locations=gp_locations, params=parameters
    )
    return Grid(obstacle_map, grid), reward_map


def load_data_to_pandas(
    input_file: str, header: List[str], delimiter: str
) -> DataFrame:
    """
    Load real-world data from a file to a pandas DataFrame
    :param input_file: File to load data from
    :param header: Header for the CSV file
    """
    with open(input_file, "r") as infile:
        csv_reader = reader(
            infile, delimiter=delimiter
        )  # Assuming data is tab-separated
        data = list(csv_reader)
        data = data[1:]  # Remove header
        dataframe = DataFrame(data, columns=header)
    return dataframe


def extract_rows_and_cols_from_data(
    data: DataFrame, bounds: Tuple[float, float, float, float], cell_size_degrees: float
) -> Tuple[int, int]:
    """
    Extracts the number of rows and columns from a dataset
    :param data: DataFrame containing the data that has latitude, longitude and depth information
    :param bounds: Bounds of the grid (lat_min, lat_max, lon_min, lon_max)
    :param cell_size_degrees: Size of each cell in degrees
    """
    lat_min, lat_max, lon_min, lon_max = bounds

    # Extract data that is within the bounds
    data = data[(data["LAT"] >= lat_min) & (data["LAT"] <= lat_max)]
    data = data[(data["LON"] >= lon_min) & (data["LON"] <= lon_max)]

    lats = data["LAT"]
    lons = data["LON"]

    num_cells_lat = int((lats.max() - lats.min()) / cell_size_degrees)
    num_cells_lon = int((lons.max() - lons.min()) / cell_size_degrees)

    return num_cells_lat, num_cells_lon


def extract_grid_from_data(
    data: DataFrame,
    bounds: Tuple[float, float, float, float],
    cell_size_degrees: float,
    obstacle_threshold: float,
) -> Grid:
    """
    Extracts a grid from a dataset
    :param data: DataFrame containing the data that has latitude, longitude and depth information
    :param bounds: Bounds of the grid (lat_min, lat_max, lon_min, lon_max)
    :param cell_size_degrees: Size of each cell in degrees
    :param obstacle_threshold: Depth threshold to classify areas as obstacles
    """

    num_cells_lat, num_cells_lon = extract_rows_and_cols_from_data(
        data, bounds, cell_size_degrees
    )

    lat_min, lat_max, lon_min, lon_max = bounds

    # Extract data that is within the bounds
    data = data[(data["LAT"] >= lat_min) & (data["LAT"] <= lat_max)]
    data = data[(data["LON"] >= lon_min) & (data["LON"] <= lon_max)]

    lats = data["LAT"]
    lons = data["LON"]
    depths = data["DEPTH"]

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, num_cells_lon),
        np.linspace(lat_min, lat_max, num_cells_lat),
    )

    grid_depth = griddata((lons, lats), depths, (grid_lon, grid_lat), method="linear")
    obstacles = np.where(grid_depth < obstacle_threshold, 0, 1).astype(np.bool_)

    return Grid(obstacles, deepcopy(obstacles))


def generate_map_from_data(
    data: DataFrame,
    bounds: Tuple[float, float, float, float],
    cell_size_degrees: float,
    obstacle_threshold: float,
    agent_locations: Union[List[Tuple[int, int]], None] = None,
    gp_means: Union[List[int], None] = None,
    gp_locations: Union[List[Tuple[int, int]], None] = None,
    parameters: Union[Parameters, None] = None,
) -> Tuple[Grid, RewardMap]:
    """
    Generates a map from real-world data
    :param data: DataFrame containing the data that has latitude, longitude and depth information
    :param bounds: Bounds of the grid (lat_min, lat_max, lon_min, lon_max)
    :param cell_size_degrees: Size of each cell in degrees
    :param obstacle_threshold: Depth threshold to classify areas as obstacles
    :param agent_locations: Locations of agents
    :param gp_means: Means of Gaussians for different locations
    :param gp_locations: Locations of Gaussians corresponding to the means
    :param parameters: Parameters for the map, GP and Vulcan to use
    """
    grid = extract_grid_from_data(data, bounds, cell_size_degrees, obstacle_threshold)

    agent_locations = [(0, 0)] if agent_locations is None else agent_locations
    for agent_location in agent_locations:
        grid.grid[agent_location[0], agent_location[1]] = False

    gp_means = [1] if gp_means is None else gp_means
    gp_locations = (
        [(grid.shape[0] // 2, grid.shape[1] // 2)]
        if gp_locations is None
        else gp_locations
    )

    reward_map = RewardMap(
        grid.shape[0],
        grid.shape[1],
        means=gp_means,
        locations=gp_locations,
        params=parameters,
    )
    return grid, reward_map
