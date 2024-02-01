from __future__ import annotations

import queue
import numpy as np
from copy import deepcopy
from scipy.special import kl_div
from mdp import MarkovDecisionProcess
from utils import get_nearest_locations
from typing import List, Tuple, Union, Dict
from map import Action, Map, Observation, ActionType


class MultiAgentSearchNode(object):
    def __init__(
        self,
        parent: Union[MultiAgentSearchNode, None],
        agents_actions: Dict[int, List[str]],
        agent_locations: Dict[int, int],
    ):
        self.parent = parent
        self.action_prefixes = agents_actions
        self.agent_locations = agent_locations

        self.f = 0.0

    @property
    def g(self):
        # Combined multi-agent information gain - true estimate
        return self.g

    @property
    def h(self):
        # Sum of single-agent information gain - heuristic estimate
        return self.h

    @g.setter
    def g(self, value: np.float64):
        self.g = value
        self.f = np.add(self.g, self.h)

    @h.setter
    def h(self, value: np.float64):
        self.h = value
        self.f = np.add(self.g, self.h)

    def __repr__(self):
        return (
            f"Multi-Agent SearchNode Summary:\n\tAgent Action Prefixes: {self.action_prefixes}"
            f"\n\tG-val (Multi-Agent Information Gain): {self.g}"
            f"\n\tH-val (Sum of Single-Agent Information Gain: {self.h}"
        )

    # Defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f > other.f

    # Defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f < other.f

    def extract_action_prefix_extensions(
        self, agent_actions: Union[List[str], None] = None
    ) -> List[Dict[int, List[str]]]:
        """
        Return a list of action sequences corresponding to one step extensions of the node's path prefix
        :param agent_actions: The possible actions agents can take
        """

        if agent_actions is None:
            agent_actions = [action_type.value for action_type in ActionType]

        extensions = []
        for extension in iter.product(agent_actions, repeat=len(self.action_prefixes)):
            updated_extension = deepcopy(self.action_prefixes)
            for idx, agent_id in enumerate(self.action_prefixes.keys()):
                agent_prefix = updated_extension[agent_id]
                agent_prefix.append(extension[idx])
            extensions.append(updated_extension)

        return extensions


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
            )
            self.current_location = self.execute_action(best_action)
            self.mdp_handle.update(self.current_location, self.map, self.use_vulcan)
            self.timer += 1

    def extract_action(
        self,
        current_location: int,
        current_timestep: int,
        planning_horizon: int,
        observations: List[Observation],
    ) -> Tuple[np.float64, Action]:
        """
        Extracts the best action to execute at the current timestep
        :param current_location: Current location of the agent
        :param current_timestep: Current timestep k
        :param planning_horizon: Planning horizon k + h
        :param observations: List of observations y_{0:k}
        """

        abscissae, weights = np.polynomial.hermite.hermgauss(self.map.params.J)
        valid_neighbors = self.map.get_neighbors(current_location)
        action_rewards = np.zeros(len(valid_neighbors))
        for idx, next in enumerate(valid_neighbors):
            _, next_location = next.action_type, next.location
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
                future_noisy_measurement = future_noisy_measurement[0]

                # y_{0:k+1}
                future_observations = observations.copy()
                future_observations.append(
                    Observation(next_location, future_noisy_measurement)
                )

                if self.map.params.distance_simplification:
                    locations_to_consider = get_nearest_locations(
                        [
                            observation.location for observation in future_observations
                        ],  # Using the distance simplification
                        self.map,
                        np.multiply(
                            self.map.params.theta_1, 5.0
                        ),  # TODO: Should we be using theta_1 or theta_2 here?
                    )
                else:
                    locations_to_consider = [
                        location_id for location_id in range(self.map.map_size)
                    ]

                # p(x_i | y_{0:k})
                current_phenomenon_probabilities = (
                    self.mdp_handle.phenomenon_probability_function(
                        locations_to_consider,
                        self.map,
                        observations,
                        unobserved_phenomenon=False,
                    )
                )

                # p(\hat{x_i} | y_{0:k+1})
                future_phenomenon_probabilities = (
                    self.mdp_handle.phenomenon_probability_function(
                        locations_to_consider,
                        self.map,
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
                    next_action_reward, _ = self.extract_action(
                        next_location,
                        current_timestep + 1,
                        planning_horizon,
                        future_observations,
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

    def update_min_costs(self, node: MultiAgentSearchNode, reward: np.float64):
        node.g = reward
        if node.parent is not None:
            self.update_min_costs(node.parent, reward)

    def multi_agent_search(
        self,
        agent_bubbles: List[Agent],
        current_timestep: int,
        planning_horizon: int,
        shared_observations: List[Observation],
        map: Map,
    ) -> Tuple[np.float64, Action]:
        """
        Performs the multi-agent search algorithm for a single agent assuming they are
        in communication range of other agents
        :param agent_bubbles: List of agents that the agent is within communication range
        :param current_timestep: Current timestep k
        :param planning_horizon: Planning horizon k + h
        :param shared_observations: List of shared observations between the agents inside the communication range
        :param map: Map object to query the updated locations for each agent in the bubble
        """

        # Make the maps of the agents in the bubble consistent with each other
        self.map.map = map.map
        for agent in agent_bubbles:
            agent.map.map = map.map

        root_node = MultiAgentSearchNode(
            None,
            {self.id: []} | {agent.id: [] for agent in agent_bubbles},
            {self.id: self.current_location}
            | {agent.id: agent.current_location for agent in agent_bubbles},
        )

        # TODO: Probably not needed to compute the h-value for the root node
        best_reward, best_action = self.extract_action(
            self.current_location,
            self.timer,
            self.timer + self.planning_horizon,
            shared_observations,
        )
        root_node.h = np.add(root_node.h, best_reward)
        for agent in agent_bubbles:
            best_reward, best_action = agent.extract_action(
                agent.current_location,
                agent.timer,
                agent.timer + agent.planning_horizon,
                shared_observations,
            )
            root_node.h = np.add(root_node.h, best_reward)

        open_set = queue.PriorityQueue()
        open_set.put(root_node)

        best_gain = np.float64(0.0)
        best_action = Action(ActionType.Wait, self.current_location)

        while not open_set.empty():
            current = open_set.get()

            if current.f < best_gain:
                return best_gain, best_action

            if len(current.action_prefixes) >= self.planning_horizon:
                # We have reached our planning horizon

                if current.g > best_gain:
                    best_gain = current.g
                    best_action = current.action_prefixes[0]
                    self.update_min_costs(
                        current, best_gain
                    )  # TODO: Check the logic here again!
            else:
                action_prefix_extensions = current.extract_action_prefix_extensions()
                for action_prefixes in action_prefix_extensions:

                    next_locations = {}
                    # Validate whether the action prefix can be executed
                    invalid_action_prefix = False
                    next_pos = self.map.extract_next_location(
                        current.agent_locations[self.id], action_prefixes[self.id][-1]
                    )
                    next_locations[self.id] = next_pos
                    if not self.map.valid_move(
                        current.agent_locations[self.id], next_pos
                    ):
                        continue

                    for other_agent in agent_bubbles:
                        next_pos = other_agent.map.extract_next_location(
                            current.agent_locations[other_agent.id],
                            action_prefixes[other_agent.id][-1],
                        )
                        next_locations[self.id] = next_pos
                        if not other_agent.map.valid_move(
                            current.agent_locations[other_agent.id], next_pos
                        ):
                            invalid_action_prefix = True
                            break

                    if invalid_action_prefix:
                        continue

                    child_node = MultiAgentSearchNode(
                        current, action_prefixes, next_locations
                    )
                    open_set.put(child_node)

                    # Extract the g-val and h-val for these nodes!

        return self.extract_action(
            self.current_location,
            self.timer,
            self.timer + self.planning_horizon,
            self.mdp_handle.observations,
        )
