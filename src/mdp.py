import numpy as np


class MarkovDecisionProcess(object):
    def __init__(self, start, map):
        observation = map.get_observation(
            start
        )  # continuous observation for now ->  should be a tuple of (location, measurement)
        self.observations = [observation]
        self.init_state = [
            self.observations,
        ]
        self.actions = ["Left", "Right", "Up", "Down"]

    # Should return p(y_i | y^{0:t})
    def compute_conditional_measurement(self, location_ids, map):
        # m(l) and v(l, l)
        mean_of_location = map.mean_function(location_ids)
        covariance_of_location = map.kernel_function(location_ids, location_ids)

        # l^{0:t}
        observation_locations = [
            observation.location for observation in self.observations
        ]
        # y^{0:t}
        observation_measurements = [
            observation.measurement for observation in self.observations
        ]

        # v(l, l^{0:t})
        covariance_location_observations = map.kernel_function(
            location_ids, observation_locations
        )

        # v(l^{0:t}, l)
        covariance_observations_location = map.kernel_function(
            observation_locations, location_ids
        )

        # v(l^{0:t}, l^{0:t})
        covariance_observations = map.kernel_function(
            observation_locations, observation_locations
        )

        mean = mean_of_location + covariance_location_observations @ np.linalg.inv(
            covariance_observations
            + np.eye(len(self.observations) * map.measurement_noise)
        ) @ (
            np.array(observation_measurements)
            - map.mean_function(observation_locations)
        )
        covariance = (
            covariance_of_location
            - covariance_location_observations
            @ np.linalg.inv(
                covariance_observations
                + np.eye(len(self.observations)) * map.measurement_noise
            )
            @ covariance_observations_location
            + np.eye(len(location_ids)) * map.measurement_noise
        )
