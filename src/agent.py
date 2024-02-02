from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing import List, Tuple
from scipy.special import kl_div
from mdp import MarkovDecisionProcess
from utils import get_nearest_locations
from map import Action, Map, Observation


class Agent(object):
    def __init__(
        self,
        id: int,
        start_location: int,
        map: Map,
        mission_duration: int = 5,
        planning_horizon: int = 2,
        use_vulcan: bool = True,
    ):
        self.id = id
        self.timer = 0
        self.map = map
        self.use_vulcan = use_vulcan
        self.current_location = start_location
        self.mission_duration = mission_duration
        self.planning_horizon = planning_horizon
        self.mdp_handle = MarkovDecisionProcess(start_location, self.map)

    def adaptive_search(self):
        """
        Performs the regular Vulcan adaptive search algorithm for a single agent assuming they are
        not in communication range of other agents
        """
        while self.timer < self.mission_duration:
            print("Time = ", self.timer)
            horizon = min(self.planning_horizon, self.mission_duration - self.timer)
            _, best_action = self.extract_action(
                self.current_location,
                self.timer,
                self.timer + horizon,
                self.mdp_handle.observations,
                deepcopy(self.map),
            )
            print(best_action)
            self.current_location = self.execute_action(best_action)
            self.mdp_handle.update(self.current_location, self.map)
            self.timer += 1

    def extract_action(
        self,
        current_location: int,
        current_timestep: int,
        planning_horizon: int,
        observations: List[Observation],
        map_object: Map,
    ) -> Tuple[np.float64, Action]:
        """
        Extracts the best action to execute at the current timestep
        :param current_location: Current location of the agent
        :param current_timestep: Current timestep k
        :param planning_horizon: Planning horizon k + h
        :param observations: List of observations y_{0:k}
        :param map_object: Map object to query the neighbors of the current location
        """

        abscissae, weights = np.polynomial.hermite.hermgauss(self.map.params.J)
        valid_neighbors = map_object.get_neighbors(current_location)
        action_rewards = np.zeros(len(valid_neighbors))
        for idx, next in enumerate(valid_neighbors):
            _, next_location = next.action_type, next.location
            (
                future_measurement_mean,
                future_measurement_covariance,
            ) = self.mdp_handle.noisy_measurement_function(
                [next_location], map_object, self.mdp_handle.observations
            )
            for index in range(map_object.params.J):
                future_noisy_measurement = (
                    abscissae[index]
                    * np.linalg.inv(np.sqrt(2 * future_measurement_covariance))
                    + future_measurement_mean
                )
                future_noisy_measurement = future_noisy_measurement[0][0]

                # y_{0:k+1}
                future_observations = observations.copy()
                future_observations.append(
                    Observation(next_location, future_noisy_measurement)
                )

                if map_object.params.distance_simplification:
                    locations_to_consider = get_nearest_locations(
                        [
                            observation.location for observation in future_observations
                        ],  # Using the distance simplification
                        map_object,
                        np.multiply(
                            map_object.params.theta_1, 5.0
                        ),  # TODO: Should we be using theta_1 or theta_2 here?
                    )
                else:
                    locations_to_consider = [
                        location_id for location_id in range(map_object.map_size)
                    ]

                # p(x_i | y_{0:k})
                current_phenomenon_probabilities = (
                    self.mdp_handle.phenomenon_probability_function(
                        locations_to_consider,
                        map_object,
                        observations,
                        unobserved_phenomenon=False,
                    )
                )

                # p(\hat{x_i} | y_{0:k+1})
                future_phenomenon_probabilities = (
                    self.mdp_handle.phenomenon_probability_function(
                        locations_to_consider,
                        map_object,
                        future_observations,
                        unobserved_phenomenon=self.use_vulcan,
                    )
                )

                # Note: Should use the unobserved phenonmenon for k+1 observations and
                # observed phenomenon for k observations

                # \sum_{i=1}^n D_KL(p(\hat{x_i} = 0 | y_{0:k+1}) || p(x_i = 0 | y_{0:k}))
                kl_divergence_not_exist = np.sum(
                    kl_div(
                        1.0 - future_phenomenon_probabilities,
                        1.0 - current_phenomenon_probabilities,
                    )
                )

                # \sum_{i=1}^n D_KL(p(x_i = 1 | y_{0:k+1}) || p(x_i = 1 | y_{0:k}))
                kl_divergence_exist = np.sum(
                    kl_div(
                        future_phenomenon_probabilities,
                        current_phenomenon_probabilities,
                    )
                )
                kl_divergence = kl_divergence_not_exist + kl_divergence_exist

                action_rewards[idx] += (weights[index] / np.sqrt(np.pi)) * kl_divergence

                if current_timestep + 1 < planning_horizon:

                    recursive_map_object = deepcopy(map_object)

                    current_location_coord = recursive_map_object.get_coordinate(
                        current_location
                    )
                    next_location_coord = recursive_map_object.get_coordinate(
                        next_location
                    )
                    recursive_map_object.map[
                        current_location_coord[0], current_location_coord[1]
                    ] = True
                    recursive_map_object.map[
                        next_location_coord[0], next_location_coord[1]
                    ] = False

                    next_action_reward, _ = self.extract_action(
                        next_location,
                        current_timestep + 1,
                        planning_horizon,
                        future_observations,
                        recursive_map_object,
                    )

                    action_rewards[idx] += (
                        weights[index] / np.sqrt(np.pi)
                    ) * next_action_reward

        best_reward = np.max(action_rewards)
        best_action = valid_neighbors[np.argmax(action_rewards)]

        return best_reward, best_action

    def execute_action(self, action: Action) -> int:
        """
        Executes an action returned by the extract_action function where it updates the agent's location
        :param action: Action to execute
        """
        _, action_location = action.action_type, action.location
        self.map.update_agent_location(self.current_location, action_location)
        return action_location
