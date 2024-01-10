import numpy as np
from dataclasses import dataclass


@dataclass
class Observation:
    location: float  # l
    measurement: float  # y


class Map(object):
    def __init__(self, maze, means=None, locations=None):
        self.map = (
            maze  # Should be a boolean array indicating where the agents/obstacles are
        )
        self.num_of_rows, self.num_of_cols, self.map_size = (
            maze.shape[0],
            maze.shape[1],
            maze.shape[0] * maze.shape[1],
        )

        if means is None:
            self.means: list = [1]
        else:
            self.means: list = means

        if locations is None:
            self.locations: list = [
                self.linearize_coordinate(self.num_of_rows // 2, self.num_of_cols // 2)
            ]
        else:
            self.locations: list = []
            for loc in locations:
                self.locations.append(self.linearize_coordinate(loc[0], loc[1]))

        self.location_means = np.zeros(self.map_size)
        for location_id, mean in zip(self.locations, self.means):
            self.location_means[location_id] = mean

        self.theta_1 = 0.4  # variance of isolated feature
        self.theta_2 = 0.01  # characteristic length for covariance decay

        self.measurement_noise = 0.2

    def get_row_coordinate(self, location_id):
        return location_id // self.num_of_cols

    def get_column_coordinate(self, location_id):
        return location_id % self.num_of_cols

    def get_coordinate(self, location_id):
        return np.array(
            [
                self.get_row_coordinate(location_id),
                self.get_column_coordinate(location_id),
            ]
        )

    def linearize_coordinate(self, row: int, column: int) -> int:
        return self.num_of_cols * row + column

    def get_manhattan_distance(self, location_id_a, location_id_b):
        location_a = self.get_coordinate(location_id_a)
        location_b = self.get_coordinate(location_id_b)
        return abs(location_a[0] - location_b[0]) + abs(location_a[1] - location_b[1])

    def valid_move(self, current, next):
        if next < 0 or next >= self.map_size or self.map[next]:
            return False
        return self.get_manhattan_distance(current, next) < 2

    def get_neighbors(self, current):
        neighbors = []
        candidates = [
            current + 1,
            current - 1,
            current + self.num_of_cols,
            current - self.num_of_cols,
        ]
        for next in candidates:
            if self.valid_move(current, next):
                neighbors.append(next)
        return neighbors

    # Defines the mean function "m" for the Gaussian Process
    def mean_function(self, location_ids):
        means = np.zeros(len(location_ids))
        for idx, location_id in enumerate(location_ids):
            means[idx] = self.location_means[location_id]
        return means

    def covariance_function(self, location_id_a, location_id_b):
        assert len(location_id_a) == 1
        location_a = self.get_coordinate(location_id_a)
        location_b = self.get_coordinate(location_id_b)
        distance = float(np.linalg.norm(location_a - location_b))
        covariance = self.theta_1 * np.exp(-distance / (self.theta_2**2))
        return covariance

    def kernel_function(self, location_ids_a, location_ids_b):
        kernel_matrix = np.zeros((len(location_ids_a), len(location_ids_b)))
        for idx_a, location_id_a in enumerate(location_ids_a):
            for idx_b, location_id_b in enumerate(location_ids_b):
                kernel_matrix[idx_a, idx_b] = self.covariance_function(
                    location_id_a, location_id_b
                )
        return kernel_matrix

    def get_feature_prior(self, location):
        # p(u_i)
        return self.feature_priors[location[0], location[1]]

    def get_measurement_noise_prior(self, location):
        # p(y_i | u_i)
        return self.measurement_noise_priors[location[0], location[1]]
