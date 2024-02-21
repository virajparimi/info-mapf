import numpy as np
from warnings import warn
from scipy.special import erf
from scipy.linalg import solve
from typing import List, Tuple
from numpy.typing import NDArray
from utils import positive_definite_matrix
from map import RewardMap, Observation, ActionType


class MarkovDecisionProcess(object):
    def __init__(self, start: int, reward_map: RewardMap):
        self.observations = []
        self.actions = [action_type.value for action_type in ActionType]

        self.update(start, reward_map)

    def update(self, location: int, reward_map: RewardMap):
        """
        Updates the observations that the agent is getting
        :param location: Location that the agent is in now after executing an action
        :param reward_map: Map object to query the underlying GP
        """

        observation = reward_map.get_observation(location)
        self.observations.append(observation)

    def measurement_function(
        self,
        location_ids: List[int],
        reward_map: RewardMap,
        observations: List[Observation],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the conditional measurement probability's p(u^{t+1:t+h} | y^{0:t}) mean and covariance - Equation 4.18
        :param location_ids: List of future locations to compute the conditional measurement over i.e L^{t+1|t+h}
        :param reward_map: Map object to query the underlying GP
        :param observations: List of observations y^{0:t}
        """
        # m(l^{t+1:t+h}) and v(l^{t+1:t_h}, l^{t+1:t+h})
        mean_of_futures = reward_map.mean_function(location_ids)
        covariance_of_futures = reward_map.kernel_function(location_ids, location_ids)

        # l^{0:t}
        observation_locations = [observation.location for observation in observations]
        # y^{0:t}
        observation_measurements = [
            observation.measurement for observation in observations
        ]

        # v(l^{t+1:t+h}, l^{0:t})
        covariance_futures_observations = reward_map.kernel_function(
            location_ids, observation_locations
        )

        # v(l^{0:t}, l^{0:t})
        covariance_observations = reward_map.kernel_function(
            observation_locations, observation_locations
        )

        # inverse_term = np.linalg.inv(
        #     covariance_observations + np.eye(len(observations)) * np.power(0.01, 2)
        # )

        # mean = mean_of_futures + covariance_futures_observations @ inverse_term @ (
        #     np.array(observation_measurements)
        #     - reward_map.mean_function(observation_locations)
        # )
        # covariance = (
        #     covariance_of_futures
        #     - covariance_futures_observations
        #     @ inverse_term
        #     @ covariance_futures_observations.T
        # )

        speed_up_term = solve(
            covariance_observations + np.eye(len(observations)) * np.power(0.01, 2),
            covariance_futures_observations.T,
            assume_a="pos",
        ).T

        mean = mean_of_futures + speed_up_term @ (
            np.array(observation_measurements)
            - reward_map.mean_function(observation_locations)
        )
        covariance = covariance_of_futures - (
            speed_up_term @ covariance_futures_observations.T
        )

        if not positive_definite_matrix(covariance):
            warn("Covariance matrix is not positive definite!", stacklevel=2)

        return mean, covariance

    def noisy_measurement_function(
        self,
        location_ids: List[int],
        reward_map: RewardMap,
        observations: List[Observation],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the conditional measurement probability p(y^{t+1:t+h} | y^{0:t})
        :param location_ids: List of future locations to compute the conditional measurement over i.e L^{t+1|t+h}
        :param reward_map: Map object to query the underlying GP
        :param observations: List of observations y^{0:t}
        """
        # m(l^{t+1:t+h}) and v(l^{t+1:t_h}, l^{t+1:t+h})
        mean_of_futures = reward_map.mean_function(location_ids)
        covariance_of_futures = reward_map.kernel_function(location_ids, location_ids)

        # l^{0:t}
        observation_locations = [observation.location for observation in observations]
        # y^{0:t}
        observation_measurements = [
            observation.measurement for observation in observations
        ]

        # v(l^{t+1:t+h}, l^{0:t})
        covariance_futures_observations = reward_map.kernel_function(
            location_ids, observation_locations
        )

        # v(l^{0:t}, l^{0:t})
        covariance_observations = reward_map.kernel_function(
            observation_locations, observation_locations
        )

        # inverse_term = np.linalg.inv(
        #     covariance_observations
        #     + np.eye(len(observations))
        #     * np.power(reward_map.params.measurement_noise, 2)
        # )

        # mean = mean_of_futures + covariance_futures_observations @ inverse_term @ (
        #     np.array(observation_measurements)
        #     - reward_map.mean_function(observation_locations)
        # )
        # covariance = (
        #     covariance_of_futures
        #     - covariance_futures_observations
        #     @ inverse_term
        #     @ covariance_futures_observations.T
        # )

        speed_up_term = solve(
            covariance_observations
            + np.eye(len(observations))
            * np.power(reward_map.params.measurement_noise, 2),
            covariance_futures_observations.T,
            assume_a="pos",
        ).T

        mean = mean_of_futures + speed_up_term @ (
            np.array(observation_measurements)
            - reward_map.mean_function(observation_locations)
        )
        covariance = covariance_of_futures - (
            speed_up_term @ covariance_futures_observations.T
        )

        if not positive_definite_matrix(covariance):
            warn("Covariance matrix is not positive definite!", stacklevel=2)

        return mean, covariance

    def phenomenon_probability_function(
        self,
        location_ids: List[int],
        reward_map: RewardMap,
        observations: List[Observation],
        unobserved_phenomenon: bool = True,
    ):
        """
        Returns the phenomenon probability of existing at a set of locations (generally the whole grid) given a set of
        observations i.e p(x_i = 1 | y^{0:t}) - Equation 4.20
        :param location_ids: List of locations to compute the phenomenon probability over i.e the future locations of
               the agent l^{t+1:t+h}
        :param map: Map object to query the underlying GP
        :param observations: List of observations y^{0:t}
        """
        means, covariances = self.measurement_function(
            location_ids, reward_map, observations
        )

        erf_quantity_numerator = (
            np.ones(means.shape[0]) * reward_map.params.u_tilde
        ) - means
        erf_quantity_denominator = np.sqrt(2 * covariances)
        erf_term = erf(erf_quantity_numerator @ np.linalg.inv(erf_quantity_denominator))

        high_probability_factors = (np.divide(reward_map.params.P_1, 2)) * (
            1.0 - erf_term
        )
        low_probability_factors = (np.divide(reward_map.params.P_2, 2)) * (
            1.0 + erf_term
        )
        phenomenon_probabilities = high_probability_factors + low_probability_factors

        if unobserved_phenomenon:
            for observation_location in observations:
                if observation_location.location in location_ids:
                    phenomenon_probabilities[
                        location_ids.index(observation_location.location)
                    ] = 0.0

        return phenomenon_probabilities
