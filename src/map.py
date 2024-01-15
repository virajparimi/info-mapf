import numpy as np
from typing import Union, List
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class Observation:
    location: int  # l
    measurement: np.float64  # y


@dataclass
class Action:
    action_type: str
    location: int


@dataclass
class Parameters:
    theta_1: np.float64  # variance of isolated feature
    theta_2: np.float64  # characteristic length for covariance decay
    u_tilde: np.float64  # threshold for conditional measurement mean
    P_1: np.float64  # probability weight when conditional measurement exceeds threshold
    P_2: np.float64  # probability weight when conditional measurement is below threshold
    J: np.int64


class Map(object):
    def __init__(
        self,
        maze: NDArray[np.bool8],
        means: Union[List, None] = None,
        locations: Union[List, None] = None,
    ):
        self.map = maze
        self.num_of_rows = maze.shape[0]
        self.num_of_cols = maze.shape[1]
        self.map_size = self.num_of_rows * self.num_of_cols

        if means is None:
            self.means = [1]
        else:
            self.means = means

        if locations is None:
            self.locations = [
                self.linearize_coordinate(self.num_of_rows // 2, self.num_of_cols // 2)
            ]
        else:
            self.locations = []
            for loc in locations:
                self.locations.append(self.linearize_coordinate(loc[0], loc[1]))

        self.grid = np.zeros(self.map.shape)
        for location_means in range(0, len(self.means)):
            linearized_location = self.locations[location_means]
            location = self.get_coordinate(linearized_location)
            sample_locations = np.random.multivariate_normal(
                location, np.eye(2), size=(1000)
            )
            for sample in sample_locations:
                sample_x, sample_y = sample
                row = int(np.round(sample_x))
                column = int(np.round(sample_y))
                if 0 <= row < maze.shape[0] and 0 <= column < maze.shape[1]:
                    self.grid[row, column] += 1.0 * self.means[location_means]

        self.grid = self.grid / np.max(self.grid)

        self.params = Parameters(
            theta_1=np.float64(0.4),
            theta_2=np.float64(0.01),
            u_tilde=np.float64(1.4),
            P_1=np.float64(0.98),
            P_2=np.float64(0.002),
            J=np.int64(5),
        )

        self.measurement_noise = 0.2

    def get_row_coordinate(self, location_id: int) -> int:
        """
        Returns the row coordinate of a linearized location ID
        :param location_id: Linearized location ID
        """
        return location_id // self.num_of_cols

    def get_column_coordinate(self, location_id: int) -> int:
        """
        Returns the column coordinate of a linearized location ID
        :param location_id: Linearized location ID
        """
        return location_id % self.num_of_cols

    def get_coordinate(self, location_id: int) -> NDArray[np.int64]:
        """
        Returns the 2D coordinate of a linearized location ID
        :param location_id: Linearized location ID
        """
        return np.array(
            [
                self.get_column_coordinate(location_id),
                self.get_row_coordinate(location_id),
            ]
        )

    def linearize_coordinate(self, row: int, column: int) -> int:
        """
        Returns the linearized location ID of a given 2D coordinate
        :param row: Row coordinate
        :param column: Column coordinate
        """
        return self.num_of_cols * row + column

    def get_manhattan_distance(self, location_id_a: int, location_id_b: int) -> int:
        """
        Returns the Manhattan distance between two linearized locations IDs
        :param location_id_a: Linearized location ID A
        :param location_id_b: Linearized location ID B
        """
        location_a = self.get_coordinate(location_id_a)
        location_b = self.get_coordinate(location_id_b)
        return np.sum(np.abs(location_a - location_b))

    def valid_move(self, current: int, next: int) -> bool:
        """
        Returns whether a move from linearized location A to linearized location B is valid.
        A valid move does not go out of map bounds and does not go through obstacles.
        Further, a valid move is only possible if the Manhattan distance between the two locations is less than 2.
        :param current: Current linearized location of an agent
        :param next: Next linearized location of an agent
        """

        next_location = self.get_coordinate(next)
        if (
            next < 0
            or next >= self.map_size
            or not self.map[next_location[0], next_location[1]]
        ):
            return False
        return self.get_manhattan_distance(current, next) < 2

    def get_neighbors(self, current: int) -> List[Action]:
        """
        Returns the valid neighbors that an agent can move to from a given linearized location
        :param current: Current linearized location of an agent
        """
        neighbors = []
        candidates = [
            Action("Right", current + 1),
            Action("Left", current - 1),
            Action("Down", current + self.num_of_cols),
            Action("Up", current - self.num_of_cols),
        ]
        for next in candidates:
            if self.valid_move(current, next.location):
                neighbors.append(next)
        return neighbors

    def update_agent_location(self, current: int, next: int):
        """
        Updates the feasible locations on a map where True implies that the location is
        traversable and False means otherwise.
        :param current: Current location of an agent
        :param next: Next location of an agent
        """

        next_location = self.get_coordinate(next)
        assert self.map[next_location[0], next_location[1]]
        current_location = self.get_coordinate(current)
        self.map[current_location[0], current_location[1]] = True
        self.map[next_location[0], next_location[1]] = False

    def mean_function(self, location_ids: List[int]) -> NDArray[np.float64]:
        """
        Defines the mean function "m" for the Gaussian Process
        :param location_ids: List of linearized locations to compute the means for
        """
        means = np.zeros(len(location_ids))
        for idx, location_id in enumerate(location_ids):
            location = self.get_coordinate(location_id)
            means[idx] = self.grid[location[0], location[1]]
        return means

    # Defines the exponential covariance function between two locations for the Gaussian Process
    def covariance_function(self, location_id_a: int, location_id_b: int) -> np.float64:
        """
        Defines the exponential covariance function for the Gaussian Process between two linearized locations
        :param location_id_a: Linearized location ID A
        :param location_id_b: Linearized location ID B
        """
        location_a = self.get_coordinate(location_id_a)
        location_b = self.get_coordinate(location_id_b)
        distance = np.float64(np.linalg.norm(location_a - location_b))
        covariance = self.params.theta_1 * np.exp(
            -distance / np.power(self.params.theta_2, 2)
        )
        return covariance

    def kernel_function(
        self, location_ids_a: List[int], location_ids_b: List[int]
    ) -> NDArray[np.float64]:
        """
        Defines the kernel function "k" for the Gaussian Process using the covariance function
        :param location_ids_a: List of linearized location IDs A
        :param location_ids_b: List of linearized location IDs B
        """
        kernel_matrix = np.zeros((len(location_ids_a), len(location_ids_b)))
        for idx_a, location_id_a in enumerate(location_ids_a):
            for idx_b, location_id_b in enumerate(location_ids_b):
                kernel_matrix[idx_a, idx_b] = self.covariance_function(
                    location_id_a, location_id_b
                )
        return kernel_matrix

    def get_observation(self, location_id: int) -> Observation:
        """
        Returns the observation at a given linearized location
        :param location_id: Linearized location ID
        """
        mean = self.mean_function([location_id])[0]
        covariance = self.covariance_function(location_id, location_id)
        sample_measurement = np.random.normal(loc=mean, scale=covariance)
        return Observation(
            location=location_id,
            measurement=np.float64(sample_measurement),
        )
