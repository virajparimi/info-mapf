from __future__ import annotations

import logging
import numpy as np
from copy import deepcopy
from scipy.special import kl_div
from numpy.typing import NDArray
from mdp import MarkovDecisionProcess
from utils import get_nearest_locations
from typing import List, Tuple, Union, Dict
from map import Action, Grid, RewardMap, Observation


class Agent(object):
    def __init__(
        self,
        id: int,
        start_location: int,
        grid: Grid,
        reward_map: RewardMap,
        mission_duration: int = 5,
        planning_horizon: int = 2,
        use_vulcan: bool = True,
    ):
        self.id = id
        self.timer = 0
        self.grid = grid
        self.reward_map = reward_map
        self.use_vulcan = use_vulcan
        self.current_location = start_location
        self.mission_duration = mission_duration
        self.planning_horizon = planning_horizon
        self.visited_locations = [start_location]
        self.mdp_handle = MarkovDecisionProcess(start_location, self.reward_map)

    def __eq__(self, __value: Agent) -> bool:
        return self.id == __value.id

    def __hash__(self) -> int:
        return hash(self.id)

    def adaptive_search(self):
        """
        Performs the regular Vulcan adaptive search algorithm for a single agent assuming they are
        not in communication range of other agents
        """
        while self.timer < self.mission_duration:
            logging.info("Time = %d", self.timer)
            horizon = min(self.planning_horizon, self.mission_duration - self.timer)
            _, best_action = self.extract_action(
                self.current_location,
                self.timer,
                self.timer + horizon,
                self.mdp_handle.observations,
                deepcopy(self.grid),
                self.reward_map,
            )
            self.current_location = self.execute_action(best_action)  # type: ignore
            self.mdp_handle.update(self.current_location, self.reward_map)
            self.timer += 1

    def extract_action(
        self,
        current_location: int,
        current_timestep: int,
        planning_horizon: int,
        observations: List[Observation],
        grid: Grid,
        reward_map: RewardMap,
        agent_future_measurements: Union[
            Dict[int, Dict[int, List[Observation]]], None
        ] = None,
        extract_all_actions: bool = False,
    ) -> Union[Tuple[np.float64, Action], Tuple[NDArray[np.float64], List[Action]]]:
        """
        Extracts the best action to execute at the current timestep
        :param current_location: Current location of the agent
        :param current_timestep: Current timestep k
        :param planning_horizon: Planning horizon k + h
        :param observations: List of observations y_{0:k}
        :param grid: Grid object to query the neighbors of the current location
        :param reward_map: Map object to query the underlying GP
        :param agent_future_measurements: Cached Future measurements of the agent. Only needed when computing h-val
        :param extract_all_actions: Flag to extract all actions instead of the best action
        """

        abscissae, weights = np.polynomial.hermite.hermgauss(self.reward_map.params.J)
        valid_neighbors = grid.get_neighbors(current_location)
        action_rewards = np.zeros(len(valid_neighbors))
        for idx, next in enumerate(valid_neighbors):
            indexed_observations = observations.copy()
            for index in range(reward_map.params.J):

                if agent_future_measurements is not None:
                    indexed_observations += agent_future_measurements[index][self.id]

                _, next_location = next.action_type, next.location
                (
                    future_measurement_mean,
                    future_measurement_covariance,
                ) = self.mdp_handle.noisy_measurement_function(
                    [next_location], reward_map, indexed_observations
                )

                future_noisy_measurement = (
                    abscissae[index]
                    * np.linalg.inv(np.sqrt(2 * future_measurement_covariance))
                    + future_measurement_mean
                )
                future_noisy_measurement = future_noisy_measurement[0][0]

                # y_{0:k+1}
                future_observations = indexed_observations.copy()
                future_observations.append(
                    Observation(next_location, future_noisy_measurement)
                )

                if reward_map.params.distance_simplification:
                    locations_to_consider = get_nearest_locations(
                        [
                            observation.location for observation in future_observations
                        ],  # Using the distance simplification
                        reward_map.num_of_rows,
                        reward_map.num_of_cols,
                        np.multiply(
                            reward_map.params.theta_1, 3.0
                        ),  # TODO: Should we be using theta_1 or theta_2 here?
                        grid,
                    )
                else:
                    locations_to_consider = [
                        location_id for location_id in range(grid.map_size)
                    ]

                # p(x_i | y_{0:k})
                current_phenomenon_probabilities = (
                    self.mdp_handle.phenomenon_probability_function(
                        locations_to_consider,
                        reward_map,
                        indexed_observations,
                        unobserved_phenomenon=self.use_vulcan,
                    )
                )

                # p(\hat{x_i} | y_{0:k+1})
                future_phenomenon_probabilities = (
                    self.mdp_handle.phenomenon_probability_function(
                        locations_to_consider,
                        reward_map,
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

                    recursive_grid_object = deepcopy(grid)
                    recursive_grid_object.update_agent_location(
                        current_location, next_location
                    )
                    next_action_reward, _ = self.extract_action(
                        next_location,
                        current_timestep + 1,
                        planning_horizon,
                        future_observations,
                        recursive_grid_object,
                        reward_map,
                        agent_future_measurements,
                    )

                    action_rewards[idx] += (
                        weights[index] / np.sqrt(np.pi)
                    ) * next_action_reward

        if extract_all_actions:
            return action_rewards, valid_neighbors
        else:
            best_reward = np.max(action_rewards)
            best_action = valid_neighbors[np.argmax(action_rewards)]

            return best_reward, best_action

    def execute_action(self, action: Action) -> int:
        """
        Executes an action returned by the extract_action function where it updates the agent's location
        :param action: Action to execute
        """
        _, action_location = action.action_type, action.location
        self.grid.update_agent_location(self.current_location, action_location)
        self.visited_locations.append(action_location)
        return action_location
