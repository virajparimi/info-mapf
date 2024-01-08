import numpy as np


class MarkovDecisionProcess(object):
    def __init__(self, start, map):
        observation = map.get_observation(
            start
        )  # discrete observation for now ->  should be a tuple of (location, observation)
        self.observations = [observation]
        self.init_state = [
            self.observations,
        ]
        self.actions = ["Left", "Right", "Up", "Down"]

    def compute_conditional_measurement(self, location, map):
        feature_prior = map.get_feature_prior(location)  # p(u_i)

        # Equation 4.24
        feature_likelihood = 0  # p(u_i | y_{0:t})
        for observation in self.observations:  # k = 0 .. t
            local_feature_likelihood = 1.0
            for local_feature_prior in map.get_feature_prior(observation.location):
                local_distance = float(np.linalg.norm(location - observation.location))
                probabilistic_correlation = feature_prior + np.exp(
                    (-1.0 * local_distance) / (map.theta**2)
                )
                local_feature_likelihood += (
                    map.get_measurement_noise_prior(observation.location)
                    * local_feature_prior
                ) / feature_prior
