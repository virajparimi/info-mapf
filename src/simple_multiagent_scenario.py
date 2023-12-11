#In this scenario we have multiple agents with rewards spread across an area.
#After each agent arrives in a space with a reward, the reward is halved for the next agent

import numpy as np

class RewardMap:

    def __init__(self, width, height):

        self.grid = np.zeros((height, width))
        self.width = width
        self.height = height

    def get_reward_from_traj(self, trajs):
        """
        Return the reward from a list of trajectories
        :param trajs: a list of lists of coordinates
        :return: the reward gained from traversing the given trajectories
        """
        grid_modify = self.grid.copy()
        reward = 0
        for t in trajs:
            for p in t:
                reward += grid_modify[p[0], p[1]]
                grid_modify[p[0], p[1]] = grid_modify[p[0], p[1]]/2
        return reward

    def get_grid_after_traj(self, trajs):
        grid_modify = self.grid.copy()
        for t in trajs:
            for p in t:
                grid_modify[p[0], p[1]] = grid_modify[p[0], p[1]] / 2
        return grid_modify




class GaussianRewardMap(RewardMap):

    def __init__(self, width, height, means=None, locations=None, samples = None):
        super().__init__(width, height)
        self.locations = locations
        self.means = means
        if locations == None:
            self.locations = np.array([[width/2, height/2]])
        if means == None:
            self.means = [1]
        if samples == None:
            samples = 1000
        self.initialize_grid(samples)

    def initialize_grid(self, samples, add_amount = 1):
        for k in range(0, len(self.means)):
            loc = self.locations[k]
            sample_locs = np.random.multivariate_normal(loc, np.eye(2),size=(samples))
            for s in sample_locs:
                row = int(np.round(s[0]))#loc[0] + s[0]
                col = int(np.round(s[1]))#loc[1] + s[1]
                if 0 <= row < self.height and 0 <= col < self.width:
                    self.grid[row, col] += add_amount*self.means[k]


