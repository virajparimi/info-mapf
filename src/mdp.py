import numpy as np
from warnings import warn
from scipy.special import erf
from typing import List, Tuple
from numpy.typing import NDArray
from map import Map, Observation
from utils import positive_definite_matrix


class MarkovDecisionProcess(object):
    def __init__(self, start: int, map: Map):
        self.states = []
        self.observations = []
        self.actions = ["Left", "Right", "Up", "Down"]

        self.update(start, map)

    def update(self, location: int, map: Map, unobserved_phenomenon: bool = False):
        """
        Updates the state that the agent is in
        :param location: Location that the agent is in now after executing an action
        :param map: Map object to query the underlying GP
        """

        observation = map.get_observation(location)
        self.observations.append(observation)

        feature_probabilities = np.zeros(map.map_size)
        feature_means, feature_covariances = self.measurement_function(
            [location_id for location_id in range(map.map_size)], map, self.observations
        )
        feature_probabilities = np.random.multivariate_normal(
            feature_means, feature_covariances
        )

        if len(self.states) == 0:
            # We should not use unobserved phenonmenon probabilities for the first state
            unobserved_phenomenon = False

        phenomenon_probabilities = self.phenomenon_probability_function(
            [location_id for location_id in range(map.map_size)],
            map,
            self.observations,
            unobserved_phenomenon,
        )

        self.states.append(
            [self.observations, feature_probabilities, phenomenon_probabilities]
        )

    def measurement_function(
        self, location_ids: List[int], map: Map, observations: List[Observation]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the conditional measurement probability's p(u^{t+1:t+h} | y^{0:t}) mean and covariance - Equation 4.18
        :param location_ids: List of future locations to compute the conditional measurement over i.e L^{t+1|t+h}
        :param map: Map object to query the underlying GP
        :param observations: List of observations y^{0:t}
        """
        # m(l^{t+1:t+h}) and v(l^{t+1:t_h}, l^{t+1:t+h})
        mean_of_futures = map.mean_function(location_ids)
        covariance_of_futures = map.kernel_function(location_ids, location_ids)

        # l^{0:t}
        observation_locations = [observation.location for observation in observations]
        # y^{0:t}
        observation_measurements = [
            observation.measurement for observation in observations
        ]

        # v(l^{t+1:t+h}, l^{0:t})
        covariance_futures_observations = map.kernel_function(
            location_ids, observation_locations
        )

        # v(l^{0:t}, l^{0:t})
        covariance_observations = map.kernel_function(
            observation_locations, observation_locations
        )

        inverse_term = np.linalg.inv(
            covariance_observations + np.eye(len(observations)) * np.power(0.01, 2)
        )

        mean = mean_of_futures + covariance_futures_observations @ inverse_term @ (
            np.array(observation_measurements)
            - map.mean_function(observation_locations)
        )
        covariance = (
            covariance_of_futures
            - covariance_futures_observations
            @ inverse_term
            @ covariance_futures_observations.T
        )

        if not positive_definite_matrix(covariance):
            warn("Covariance matrix is not positive definite!", stacklevel=2)

        return mean, covariance

    def noisy_measurement_function(
        self, location_ids: List[int], map: Map, observations: List[Observation]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the conditional measurement probability p(y^{t+1:t+h} | y^{0:t})
        :param location_ids: List of future locations to compute the conditional measurement over i.e L^{t+1|t+h}
        :param map: Map object to query the underlying GP
        :param observations: List of observations y^{0:t}
        """
        # m(l^{t+1:t+h}) and v(l^{t+1:t_h}, l^{t+1:t+h})
        mean_of_futures = map.mean_function(location_ids)
        covariance_of_futures = map.kernel_function(location_ids, location_ids)

        # l^{0:t}
        observation_locations = [observation.location for observation in observations]
        # y^{0:t}
        observation_measurements = [
            observation.measurement for observation in observations
        ]

        # v(l^{t+1:t+h}, l^{0:t})
        covariance_futures_observations = map.kernel_function(
            location_ids, observation_locations
        )

        # v(l^{0:t}, l^{0:t})
        covariance_observations = map.kernel_function(
            observation_locations, observation_locations
        )

        inverse_term = np.linalg.inv(
            covariance_observations
            + np.eye(len(observations)) * np.power(map.params.measurement_noise, 2)
        )

        mean = mean_of_futures + covariance_futures_observations @ inverse_term @ (
            np.array(observation_measurements)
            - map.mean_function(observation_locations)
        )
        covariance = (
            covariance_of_futures
            - covariance_futures_observations
            @ inverse_term
            @ covariance_futures_observations.T
        )

        if not positive_definite_matrix(covariance):
            warn("Covariance matrix is not positive definite!", stacklevel=2)

        return mean, covariance

    def phenomenon_probability_function(
        self,
        location_ids: List[int],
        map: Map,
        observations: List[Observation],
        unobserved_phenomenon: bool = True,
    ):
        """
        Returns the phenomenon probability of existing at a set of locations (generally the whole grid) given a set of
        observations i.e p(x_i = 1 | y^{0:t}) - Equation 4.20
        :param location_ids: List of locations to compute the phenomenon probability over
        :param map: Map object to query the underlying GP
        :param observations: List of observations y^{0:t}
        """
        means, covariances = self.measurement_function(location_ids, map, observations)

        erf_quantity_numerator = (np.ones(means.shape[0]) * map.params.u_tilde) - means
        erf_quantity_denominator = np.diag(
            np.sqrt(2 * covariances)
        )  # TODO: Should we be using diagonal entries only?

        high_probability_factors = (np.divide(map.params.P_1, 2)) * (
            1.0 - erf(erf_quantity_numerator / erf_quantity_denominator)
        )
        low_probability_factors = (np.divide(map.params.P_2, 2)) * (
            1.0 + erf(erf_quantity_numerator / erf_quantity_denominator)
        )
        phenomenon_probabilities = high_probability_factors + low_probability_factors

        if unobserved_phenomenon:
            last_observation_location = observations[-1].location
            last_observation_location_index = location_ids.index(
                last_observation_location
            )
            phenomenon_probabilities[last_observation_location_index] = 0.0

        return phenomenon_probabilities
