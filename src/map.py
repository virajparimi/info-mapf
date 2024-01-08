import numpy as np
from dataclasses import dataclass


@dataclass
class Observation:
    location: float  # l
    observation: float  # y


class Map(object):
    def __init__(self, maze, start):
        self.map = maze
        self.start = start
        self.height, self.width = maze.shape[0], maze.shape[1]

        # Map's decay rate
        self.theta = 0.01

        # Uniform measurement prior - p(u_i) - set to 0.5
        self.feature_priors = np.ones((self.height, self.width)) * 0.5

        # Noisy mesurement prior - p(y_i | u_i) - set to 0.1 i.e 10% chance of being wrong
        self.measurement_noise_priors = np.ones((self.height, self.width)) * 0.1

        # Phenomenons - p(x_i)
        self.phenomenon_likelihoods = np.zeros((self.height, self.width))

    def get_feature_prior(self, location):
        # p(u_i)
        return self.feature_priors[location[0], location[1]]

    def get_measurement_noise_prior(self, location):
        # p(y_i | u_i)
        return self.measurement_noise_priors[location[0], location[1]]
