# In this scenario we have multiple agents with rewards spread across an area.
# After each agent arrives in a space with a reward, the reward is halved for the next agent

import numpy as np
from warnings import warn

class RewardMap:

    def __init__(self, width, height):

        self.grid = np.zeros((height, width))
        self.width = width
        self.height = height

    def get_reward_from_trajectory(self, trajectory):
        """
        Return the reward from a trajectory
        :param trajectory: A list of lists of coordinates
        :return: The reward gained from traversing the given trajectories
        """
        grid_modify = self.grid.copy()
        reward = 0
        for point in trajectory:
            point_x, point_y = point
            reward += grid_modify[point_x, point_y]
            grid_modify[point_x, point_y] = grid_modify[point_x, point_y] / 2.0
        return reward

    def get_grid_after_trajectory(self, trajectory):
        grid_modify = self.grid.copy()
        for point in trajectory:
            point_x, point_y = point    
            grid_modify[point_x, point_y] = grid_modify[point_x, point_y] / 2
        return grid_modify


class GaussianRewardMap(RewardMap):

    def __init__(self, width, height, means = None, locations = None, samples = None):
        super().__init__(width, height)
        self.means = means
        self.locations = locations
        if locations == None:
            self.locations = np.array([[width/2, height/2]])
        if means == None:
            self.means = [1]
        if samples == None:
            samples = 1000
        self.initialize_grid(samples)

    def initialize_grid(self, samples, add_amount = 1):

        if self.means is None:
            warn("No means specified for GaussianRewardMap")
            return self.grid
        if self.locations is None:
            warn("No locations specified for GaussianRewardMap")
            return self.grid

        for location_means in range(0, len(self.means)):
            location = self.locations[location_means]
            sample_locations = np.random.multivariate_normal(location, np.eye(2), size=(samples))
            for sample in sample_locations:
                sample_x, sample_y = sample
                row = int(np.round(sample_x))
                column = int(np.round(sample_y))
                if 0 <= row < self.height and 0 <= column < self.width:
                    self.grid[row, column] += add_amount * self.means[location_means]


