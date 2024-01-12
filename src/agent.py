import numpy as np
from typing import List
from map import Map, Observation
from scipy.special import kl_div
from mdp import MarkovDecisionProcess

# Default agent class with 4 cardinal deterministic actions


class Agent(object):
    def __init__(
        self,
        start_location: int,
        map: Map,
        mission_duration: int = 5,
        planning_horizon: int = 2,
    ):
        self.timer = 0
        self.map = map
        self.current_location = start_location
        self.mission_duration = mission_duration
        self.planning_horizon = planning_horizon
        self.mdp_handle = MarkovDecisionProcess(start_location, self.map)

    def adaptive_search(self):
        while self.timer < self.mission_duration:
            horizon = min(self.planning_horizon, self.mission_duration - self.timer)
            best_action = self.extract_action(
                self.timer, self.timer + horizon, self.mdp_handle.observations
            )
            self.current_location = self.execute_action(
                best_action
            )  # TODO: This function should update the map
            self.mdp_handle.update(self.current_location, self.map)
            self.timer += 1

    def extract_action(
        self,
        current_timestep: int,
        planning_horizon: int,
        observations: List[Observation],
    ) -> str:
        """
        Extracts the best action to execute at the current timestep
        :param current_timestep: Current timestep k
        :param planning_horizon: Planning horizon k + h
        :param observations: List of observations y_{0:k}
        """

        abscissae, weights = np.polynomial.hermite.hermgauss(self.map.params.J)
        for next in self.map.get_neighbors(self.current_location):
            next_action, next_location = next.action_type, next.location
            action_reward = 0
            (
                future_measurement_mean,
                future_measurement_covariance,
            ) = self.mdp_handle.noisy_measurement_function(
                [next_location], self.map, self.mdp_handle.observations
            )
            for index in range(self.map.params.J):
                future_noisy_measurement = (
                    abscissae[index]
                    * np.diag(
                        np.sqrt(2 * future_measurement_covariance)
                    )  # TODO: Do we need extract the diagnoal entries here?
                    + future_measurement_mean
                )

                # p(x_i | y_{0:k})
                current_phenomenon_probabilities = (
                    self.mdp_handle.phenomenon_probability_function(
                        [location_id for location_id in range(self.map.map_size)],
                        self.map,
                        observations,
                    )
                )

                # y_{0:k+1}
                future_observations = observations.copy()
                future_observations.append(
                    Observation(next_location, future_noisy_measurement)
                )

                # p(x_i | y_{0:k+1})
                future_phenomenon_probabilities = (
                    self.mdp_handle.phenomenon_probability_function(
                        [location_id for location_id in range(self.map.map_size)],
                        self.map,
                        future_observations,
                    )
                )

                # \sum_{i=1}^n D_KL(p(x_i | y_{0:k+1}) || p(x_i | y_{0:k}))
                kl_divergence = np.sum(
                    kl_div(
                        future_phenomenon_probabilities,
                        current_phenomenon_probabilities,
                    )
                )
                action_reward += (weights[index] / np.sqrt(np.pi)) * kl_divergence
