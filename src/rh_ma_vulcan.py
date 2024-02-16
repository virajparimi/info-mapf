from __future__ import annotations

import queue
import logging
import numpy as np
import itertools as iter
from copy import deepcopy
from scipy.special import kl_div
from utils import get_nearest_locations
from map import Grid, RewardMap, ActionType
from agent import Agent, Observation, Action
from typing import List, Union, Dict, Tuple, Set, Generator, Any


class MultiAgentSearchNode(object):
    def __init__(
        self,
        parent: Union[MultiAgentSearchNode, None],
        agents_actions: Dict[int, List[str]],
        agent_locations: Dict[int, int],
        grid: Union[Grid, None] = None,
    ):
        self.timestep = 0
        self.grid = grid
        self.parent = parent
        self.action_prefixes = agents_actions
        self.agent_locations = agent_locations
        self.cached_h_values = {agent: {} for agent in agent_locations.keys()}
        self.agent_action_h = {agent: {} for agent in agent_locations.keys()}

        self._g = np.float64(0.0)
        self._h = np.float64(0.0)
        self._f = np.add(self._g, self._h)

    @property
    def g(self):
        # Combined multi-agent information gain - true estimate
        return self._g

    @property
    def h(self):
        # Sum of single-agent information gain - heuristic estimate
        return self._h

    @g.setter
    def g(self, value: np.float64):
        self._g = value
        self._f = np.add(self._g, self._h)

    @h.setter
    def h(self, value: np.float64):
        self._h = value
        self._f = np.add(self._g, self._h)

    def __repr__(self):
        return (
            f"Multi-Agent SearchNode Summary:\n\tAgent Action Prefixes: {self.action_prefixes}"
            f"\n\tG-val (Multi-Agent Information Gain): {self._g}"
            f"\n\tH-val (Sum of Single-Agent Information Gain: {self._h}"
        )

    # Defining less than for purposes of heap queue
    def __lt__(self, other):
        return self._f > other._f

    # Defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self._f < other._f

    def extract_action_prefix_extensions(
        self, agent_actions: Union[List[str], None] = None
    ) -> Generator[Dict[int, List[str]], Any, Any]:
        """
        Return a list of action sequences corresponding to one step extensions of the node's path prefix
        :param agent_actions: The possible actions agents can take
        """

        if agent_actions is None:
            agent_actions = [action_type.value for action_type in ActionType]

        for extension in iter.product(agent_actions, repeat=len(self.action_prefixes)):
            updated_extension = deepcopy(self.action_prefixes)
            for idx, agent_id in enumerate(self.action_prefixes.keys()):
                agent_prefix = updated_extension[agent_id]
                agent_prefix.append(extension[idx])
            yield updated_extension


class MultiAgentVulcan(object):
    def __init__(
        self,
        grid: Grid,
        reward_map: RewardMap,
        agents: List[Agent],
        communication_range: int = 5,
    ):
        self.timer = 0
        self.grid = grid
        self.agents = agents
        self.reward_map = reward_map
        self.communication_range = communication_range
        self.mission_duration = max([agent.mission_duration for agent in agents])

        self.children = 0
        self.nodes_expanded = 0
        self.nodes_generated = 0

    def planner(self):
        while self.timer < self.mission_duration:
            logging.info(f"Time = {self.timer}")
            agent_actions = {}
            # Collect agents within communication range
            agent_bubbles = self.within_range_agents()
            # Command each agent to execute their adaptive search algorithm for one step

            skip_agent = {agent.id: False for agent in self.agents}
            for idx, agent in enumerate(self.agents):
                if (
                    idx in agent_bubbles.keys() and len(agent_bubbles[idx]) > 1
                ):  # uniqueness of the agent bubbles is required here!
                    # Start multi-agent search algorithm with respect to this agent
                    shared_observations = []
                    for agent_in_comm_range in agent_bubbles[idx]:
                        shared_observations += (
                            agent_in_comm_range.mdp_handle.observations
                        )
                    shared_observations = list(set(shared_observations))
                    # TODO: Which measurement should we use for the locations that are common among these agents?

                    horizon = min(
                        agent.planning_horizon, agent.mission_duration - agent.timer
                    )

                    for agent_in_comm_range in agent_bubbles[idx]:
                        agent_in_comm_range.mdp_handle.observations = (
                            shared_observations
                        )

                    _, best_action = self.multi_agent_search(
                        agent,
                        agent_bubbles[idx],
                        horizon,
                        shared_observations,
                    )

                    logging.debug(
                        f"Total children created for this search: {self.children}"
                    )
                    self.children = 0

                    assert best_action is not None

                    for agent_in_comm_range in agent_bubbles[idx]:
                        if agent.id == agent_in_comm_range.id:
                            agent_actions[agent_in_comm_range.id] = best_action[
                                agent_in_comm_range.id
                            ]
                        else:
                            if agent_in_comm_range.id not in agent_bubbles.keys():
                                agent_actions[agent_in_comm_range.id] = best_action[
                                    agent_in_comm_range.id
                                ]
                                skip_agent[agent_in_comm_range.id] = True

                    logging.debug(
                        "Ratio of nodes expanded to nodes generated: "
                        + str(self.nodes_expanded / self.nodes_generated)
                    )

                elif not skip_agent[agent.id]:
                    # Re-use vulcan for this single agent
                    horizon = min(
                        agent.planning_horizon, agent.mission_duration - agent.timer
                    )
                    _, best_action = agent.extract_action(
                        agent.current_location,
                        agent.timer,
                        agent.timer + horizon,
                        agent.mdp_handle.observations,
                        deepcopy(agent.grid),
                        agent.reward_map,
                    )

                    agent_actions[agent.id] = best_action

            assert len(agent_actions) == len(self.agents)

            # Once we have extracted the best actions for each agent, we execute them
            old_locations = [agent.current_location for agent in self.agents]
            old_locations_coords = [
                self.grid.get_coordinate(location) for location in old_locations
            ]
            new_locations = [action.location for action in agent_actions.values()]
            new_locations_coords = [
                self.grid.get_coordinate(location) for location in new_locations
            ]

            for old_loc in old_locations_coords:
                self.grid.grid[old_loc[0], old_loc[1]] = True
            for new_loc in new_locations_coords:
                self.grid.grid[new_loc[0], new_loc[1]] = False

            for agent in self.agents:
                agent.current_location = agent_actions[agent.id].location
                agent.visited_locations.append(agent.current_location)
                agent.mdp_handle.update(agent.current_location, agent.reward_map)
                agent.timer += 1
            self.timer += 1

    def find_minimal_disjoint_sets(
        self, agent_bubbles: List[Set[Agent]]
    ) -> List[Set[Agent]]:
        """
        Given a list of sets, find the minimal disjoint sets
        :param agent_bubbles: List of sets
        """
        minimal_disjoint_sets = []

        for bubble in agent_bubbles:
            intersecting_sets = []
            for idx, existing_set in enumerate(minimal_disjoint_sets):
                if set(bubble).intersection(existing_set):
                    intersecting_sets.append(idx)

            if not intersecting_sets:
                minimal_disjoint_sets.append(set(bubble))
            else:
                merged_set = set(bubble)
                for idx in sorted(intersecting_sets, reverse=True):
                    merged_set |= minimal_disjoint_sets.pop(idx)
                minimal_disjoint_sets.append(merged_set)

        return minimal_disjoint_sets

    def within_range_agents(self) -> Dict[int, List[Agent]]:
        """
        Returns a list of agents within communication range
        """
        agent_bubbles = [[self.agents[i]] for i in range(len(self.agents))]
        for agent_i in self.agents:
            for agent_j in self.agents:
                if agent_i.id == agent_j.id:
                    continue
                agent_i_location = agent_i.current_location
                agent_j_location = agent_j.current_location
                if (
                    self.grid.get_manhattan_distance(agent_i_location, agent_j_location)
                    < self.communication_range
                ):
                    agent_bubbles[agent_i.id].append(agent_j)

            agent_bubbles[agent_i.id] = sorted(
                agent_bubbles[agent_i.id], key=lambda x: x.id
            )

        modified_agent_bubbles = [set(bubble) for bubble in agent_bubbles]
        minimal_disjoint_sets = self.find_minimal_disjoint_sets(modified_agent_bubbles)
        mapped_agent_bubbles = {}
        for idx, bubble in enumerate(minimal_disjoint_sets):
            list_bubble = list(bubble)
            min_id = min([agent.id for agent in list_bubble])
            mapped_agent_bubbles[min_id] = list_bubble

        return mapped_agent_bubbles

    def update_min_costs(self, node: MultiAgentSearchNode, reward: np.float64):
        node.g = reward
        if node.parent is not None and np.less(node.parent.g, reward):
            self.update_min_costs(node.parent, reward)

    def construct_node(
        self,
        parent: Union[MultiAgentSearchNode, None],
        action_prefixes: Dict[int, List[str]],
        agent_locations: Dict[int, int],
        agent_bubbles: List[Agent],
        planning_horizon: int,
        shared_observations: List[Observation],
    ) -> MultiAgentSearchNode:
        """
        Construct the multi-agent search node and sets the timestep, g-val and h-val
        :param parent: The parent node of the current node
        :param action_prefixes: The action prefixes for each agent
        :param agent_locations: The locations of each agent after executing the action prefixes
        :param agent_bubbles: List of agents that the agent is within communication range including itself
        :param planning_horizon: Planning horizon k + h
        :param shared_observations: List of shared observations between the agents inside the communication range
        """
        # logging.debug(
        #     "Constructing a multi-agent search node for action prefixes:"
        #     + str(action_prefixes)
        # )
        agents_future_measurements = None
        node = MultiAgentSearchNode(parent, action_prefixes, agent_locations)
        if parent is not None:
            node.timestep = parent.timestep + 1

        if parent is None:
            node.g = np.float64(0.0)
        elif node.timestep < planning_horizon:
            result = self.compute_multi_agent_information_gain(
                node, agent_bubbles, planning_horizon, shared_observations
            )
            node.g, agents_future_measurements = result
        else:
            result = self.compute_multi_agent_information_gain(
                node, agent_bubbles, planning_horizon, shared_observations
            )
            node.g = result[0]

        if node.timestep >= planning_horizon:
            self.children += 1
            node.h = np.float64(0.0)
        else:

            recursive_grid_object = deepcopy(
                agent_bubbles[0].grid
            )  # Take any agent's grid map

            # Update the locations of the agents in the map as its required for updated valid neighbor computation
            for other_agent in agent_bubbles:
                if node.parent is not None:
                    older_coords = recursive_grid_object.get_coordinate(
                        node.parent.agent_locations[other_agent.id]
                    )
                    new_coords = recursive_grid_object.get_coordinate(
                        node.agent_locations[other_agent.id]
                    )
                    recursive_grid_object.grid[older_coords[0], older_coords[1]] = True
                    recursive_grid_object.grid[new_coords[0], new_coords[1]] = False

            node.grid = recursive_grid_object

            for agent_in_comm_range in agent_bubbles:

                dict_key = str(node.action_prefixes[agent_in_comm_range.id])
                if (
                    node.parent is not None
                    and dict_key in node.parent.cached_h_values[agent_in_comm_range.id]
                ):
                    node.h = np.add(
                        node.h,
                        node.parent.cached_h_values[agent_in_comm_range.id][dict_key],
                    )
                else:

                    action_rewards, actions = agent_in_comm_range.extract_action(
                        agent_locations[agent_in_comm_range.id],
                        node.timestep,
                        planning_horizon - node.timestep,
                        shared_observations,
                        recursive_grid_object,
                        agent_in_comm_range.reward_map,
                        agents_future_measurements,
                        extract_all_actions=True,
                    )

                    assert isinstance(actions, list)
                    assert isinstance(action_rewards, np.ndarray)

                    action_rewards_mapping = {
                        action.action_type.value: action_rewards[idx]
                        for idx, action in enumerate(actions)
                    }
                    action_rewards_mapping = dict(
                        sorted(
                            action_rewards_mapping.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )

                    best_reward = np.max(action_rewards)
                    node.h = np.add(node.h, best_reward)

                    node.cached_h_values[agent_in_comm_range.id].update(
                        {dict_key: best_reward}
                    )
                    node.agent_action_h[agent_in_comm_range.id].update(
                        action_rewards_mapping
                    )

        self.nodes_generated += 1
        return node

    def recursive_information_gain(
        self,
        agent: Agent,
        current_timestep: int,
        planning_horizon: int,
        agent_locations_history: Dict[int, Dict[int, int]],
        observations: List[Observation],
        agents_future_measurements: Dict[int, Dict[int, List[Observation]]],
        use_vulcan: bool = True,
    ) -> Tuple[np.float64, Dict[int, List[Observation]]]:
        g_val = np.float64(0.0)
        measurements = {current_timestep: []}

        abscissae, weights = np.polynomial.hermite.hermgauss(self.reward_map.params.J)
        agent_location = agent_locations_history[current_timestep][agent.id]

        indexed_observations = observations.copy()
        for index in range(self.reward_map.params.J):

            for agent_id in agent_locations_history[current_timestep].keys():
                indexed_observations += agents_future_measurements[index][agent_id]
            (
                future_measurement_mean,
                future_measurement_covariance,
            ) = agent.mdp_handle.noisy_measurement_function(
                [agent_location], agent.reward_map, indexed_observations
            )

            future_noisy_measurement = (
                abscissae[index]
                * np.linalg.inv(np.sqrt(2 * future_measurement_covariance))
                + future_measurement_mean
            )
            future_noisy_measurement = future_noisy_measurement[0][0]

            # y_{0:k+1}
            future_observations = indexed_observations.copy()
            new_observation = Observation(agent_location, future_noisy_measurement)
            future_observations.append(new_observation)
            measurements[current_timestep].append(new_observation)

            if self.reward_map.params.distance_simplification:
                locations_to_consider = get_nearest_locations(
                    [
                        observation.location for observation in future_observations
                    ],  # Using the distance simplification
                    self.reward_map.num_of_rows,
                    self.reward_map.num_of_cols,
                    np.multiply(
                        self.reward_map.params.theta_1, 3.0
                    ),  # TODO: Should we be using theta_1 or theta_2 here?
                )
            else:
                locations_to_consider = np.arange(self.grid.map_size).tolist()

            # p(x_i | y_{0:k})
            current_phenomenon_probabilities = (
                agent.mdp_handle.phenomenon_probability_function(
                    locations_to_consider,
                    self.reward_map,
                    indexed_observations,
                    unobserved_phenomenon=False,
                )
            )

            # p(\hat{x_i} | y_{0:k+1})
            future_phenomenon_probabilities = (
                agent.mdp_handle.phenomenon_probability_function(
                    locations_to_consider,
                    self.reward_map,
                    future_observations,
                    unobserved_phenomenon=use_vulcan,
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
            g_val += (weights[index] / np.sqrt(np.pi)) * kl_divergence

            if current_timestep + 1 < planning_horizon:
                future_g_val, future_measurements = self.recursive_information_gain(
                    agent,
                    current_timestep + 1,
                    planning_horizon,
                    agent_locations_history,
                    future_observations,
                    agents_future_measurements,
                )
                for f_timestep, f_measurement in future_measurements.items():
                    measurements[f_timestep] = f_measurement
                g_val += (weights[index] / np.sqrt(np.pi)) * future_g_val

        return g_val, measurements

    def extract_agent_locations_history(
        self,
        current: MultiAgentSearchNode,
        agent_locations_history: Dict[int, Dict[int, int]],
    ):
        """
        Extract the agent locations history from the current node
        :param current: The current node that is used to go up till the root
        :param agent_locations_history: The agent locations history
        """
        node = current
        agent_locations_history[node.timestep] = node.agent_locations
        while node.parent is not None and node.parent.timestep != 0:
            agent_locations_history[node.parent.timestep] = node.parent.agent_locations
            node = node.parent

        return agent_locations_history

    def compute_multi_agent_information_gain(
        self,
        current: MultiAgentSearchNode,
        agent_bubbles: List[Agent],
        planning_horizon: int,
        shared_observations: List[Observation],
    ) -> Tuple[np.float64, Dict[int, Dict[int, List[Observation]]]]:
        """
        Compute the multi-agent information gain
        :param current: The current node for which the multi-agent information gain is being computed
        :param agent_bubbles: List of agents that the agent is within communication range including itself
        :param planning_horizon: Planning horizon h
        :param shared_observations: List of shared observations between the agents inside the communication range
        """
        g_val = np.float64(0.0)

        agent_locations_history = self.extract_agent_locations_history(current, {})
        agents_future_measurements = {
            index: {} for index in range(self.reward_map.params.J)
        }
        for index in range(self.reward_map.params.J):
            for agent in agent_bubbles:
                agents_future_measurements[index][agent.id] = []

        assert current.parent is not None

        for idx, agent in enumerate(agent_bubbles):
            agent_observation_handle = deepcopy(shared_observations)
            future_g_val, agent_f_measurements = self.recursive_information_gain(
                agent,
                1,
                planning_horizon + current.timestep - 1,
                agent_locations_history,
                agent_observation_handle,
                agents_future_measurements,
                agent.use_vulcan,
            )

            for index in range(self.reward_map.params.J):
                for f_timestep, f_agents_measurements in agent_f_measurements.items():
                    agents_future_measurements[index][agent.id].append(
                        f_agents_measurements[index]
                    )
            g_val = np.add(g_val, future_g_val)

        return (g_val, agents_future_measurements)

    def multi_agent_search(
        self,
        target_agent: Agent,
        agent_bubbles: List[Agent],
        planning_horizon: int,
        shared_observations: List[Observation],
    ) -> Tuple[np.float64, Union[Dict[int, Action], None]]:
        """
        Performs the multi-agent search algorithm for a single agent assuming they are
        in communication range of other agents
        :param target_agent: The agent for which the multi-agent search algorithm is being performed
        :param agent_bubbles: List of agents that the agent is within communication range including itself
        :param planning_horizon: Planning horizon h
        :param shared_observations: List of shared observations between the agents inside the communication range
        """

        logging.debug(
            "Starting multi-agent search algorithm for "
            + str(len(agent_bubbles))
            + " agents"
        )

        root_node = self.construct_node(
            None,
            {agent.id: [] for agent in agent_bubbles},
            {agent.id: agent.current_location for agent in agent_bubbles},
            agent_bubbles,
            planning_horizon,
            shared_observations,
        )

        open_set = queue.PriorityQueue()
        open_set.put(root_node)

        best_gain = np.float64(0.0)
        best_action = {
            agent.id: Action(ActionType.Wait, agent.current_location)
            for agent in agent_bubbles
        }

        while not open_set.empty():
            current = open_set.get()
            self.nodes_expanded += 1

            if current._f <= best_gain and current.timestep < planning_horizon:
                logging.debug("Size of the open set: " + str(open_set.qsize()))
                logging.debug("Best action: " + str(best_action))
                return best_gain, best_action

            if current.timestep >= planning_horizon:
                # We have reached our planning horizon

                logging.debug("Current is a leaf!")
                if current.parent is not None:
                    self.update_min_costs(
                        current.parent, current.g
                    )  # TODO: Check the logic here again!

                if current.g >= best_gain:
                    logging.debug("Best action was updated!")
                    best_gain = current.g
                    for agent_in_comm_range in agent_bubbles:
                        best_action_str = current.action_prefixes[
                            agent_in_comm_range.id
                        ][0]
                        best_action_location = (
                            agent_in_comm_range.grid.extract_next_location(
                                agent_in_comm_range.current_location,
                                best_action_str,
                            )
                        )
                        best_action[agent_in_comm_range.id] = Action(
                            best_action_str, best_action_location
                        )

            else:

                logging.debug("Current is not a leaf!")
                assert current.grid is not None
                valid_actions = set()
                for agent_in_comm_range in agent_bubbles:
                    valid_neighbors = current.grid.get_neighbors(
                        current.agent_locations[agent_in_comm_range.id]
                    )
                    for valid_neighbor in valid_neighbors:
                        valid_actions.add(valid_neighbor.action_type.value)
                valid_actions = list(valid_actions)

                child_best_gain = np.float64(0.0)
                for action_prefixes in current.extract_action_prefix_extensions(
                    valid_actions
                ):

                    # Validate whether the action prefix can be executed
                    next_locations = {}
                    invalid_action_prefix = False
                    temp_h_val = np.float64(0.0)
                    for agent_idx, agent_in_comm_range in enumerate(agent_bubbles):
                        action = action_prefixes[agent_in_comm_range.id][-1]
                        next_pos = current.grid.extract_next_location(
                            current.agent_locations[agent_in_comm_range.id],
                            action,
                        )
                        if (
                            next_pos < 0
                            or next_pos >= current.grid.map_size
                            or current.grid.get_manhattan_distance(
                                current.agent_locations[agent_in_comm_range.id],
                                next_pos,
                            )
                            > 1
                        ):
                            invalid_action_prefix = True
                            break

                        next_locations[agent_in_comm_range.id] = next_pos
                        temp_h_val = np.add(
                            temp_h_val,
                            current.agent_action_h[agent_in_comm_range.id][action],
                        )

                    if invalid_action_prefix:
                        continue

                    # Check for vertex collisions and edge collisions given the paths of these agents
                    for agent_i in agent_bubbles:
                        for agent_j in agent_bubbles:
                            if agent_i.id == agent_j.id:
                                continue
                            # Vertex collision
                            if next_locations[agent_i.id] == next_locations[agent_j.id]:
                                invalid_action_prefix = True
                                break
                            # Edge collision
                            if (
                                current.agent_locations[agent_i.id]
                                == next_locations[agent_j.id]
                                and next_locations[agent_i.id]
                                == current.agent_locations[agent_j.id]
                            ):
                                invalid_action_prefix = True
                                break

                    if invalid_action_prefix:
                        continue

                    # Pruning the nodes before generation!
                    if current.timestep + 1 >= planning_horizon:
                        temp_f_val = current.g + temp_h_val
                        if temp_f_val <= child_best_gain:
                            continue

                    child_node = self.construct_node(
                        current,
                        action_prefixes,
                        next_locations,
                        agent_bubbles,
                        planning_horizon,
                        shared_observations,
                    )

                    if child_node.timestep >= planning_horizon:
                        assert child_node._f <= current._f or np.isclose(
                            child_node._f, current._f
                        )

                    if child_node._f > child_best_gain:
                        child_best_gain = child_node._f

                    open_set.put(child_node)

        logging.debug("U: Size of the open set: {open_set.qsize()}")
        return best_gain, best_action
