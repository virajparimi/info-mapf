from __future__ import annotations

import queue
import numpy as np
from copy import deepcopy
from map import Map, ActionType
from typing import List, Union, Dict, Tuple
from agent import Agent, Observation, Action


class MultiAgentSearchNode(object):
    def __init__(
        self,
        parent: Union[MultiAgentSearchNode, None],
        agents_actions: Dict[int, List[str]],
        agent_locations: Dict[int, int],
    ):
        self.timestep = 0
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


class Multi_Agent_Vulcan(object):
    def __init__(self, map: Map, agents: List[Agent], communication_range: int = 5):
        self.timer = 0
        self.map = map
        self.agents = agents
        self.communication_range = communication_range
        self.mission_duration = max([agent.mission_duration for agent in agents])

    def planner(self):
        while self.timer < self.mission_duration:
            print("Time = ", self.timer)
            agent_actions = []
            # Collect agents within communication range
            agent_bubbles = self.within_range_agents()
            # Command each agent to execute their adaptive search algorithm for one step
            for idx, agent in enumerate(self.agents):
                if len(agent_bubbles[idx]) > 0:
                    # Start multi-agent search algorithm with respect to this agent
                    shared_observations = []
                    for agent_in_comm_range in agent_bubbles[idx]:
                        shared_observations += (
                            agent_in_comm_range.mdp_handle.observations
                        )
                    shared_observations = list(
                        set(shared_observations)
                    )  # ensure uniqueness of the observations

                    horizon = min(
                        agent.planning_horizon, agent.mission_duration - self.timer
                    )

                    for agent_in_comm_range in agent_bubbles[idx]:
                        agent_in_comm_range.mdp_handle.observations = (
                            shared_observations
                        )
                        # States of the MDP handle are not updated here as I dont think we need to

                    _, best_action = self.multi_agent_search(
                        agent,
                        agent_bubbles[idx],
                        self.timer,
                        self.timer + horizon,
                        shared_observations,
                    )
                else:
                    # Re-use vulcan for this single agent
                    horizon = min(
                        agent.planning_horizon, agent.mission_duration - self.timer
                    )
                    _, best_action = agent.extract_action(
                        agent.current_location,
                        self.timer,
                        self.timer + horizon,
                        agent.mdp_handle.observations,
                    )

                agent_actions.append(best_action)

            # Once we have extracted the best actions for each agent, we execute them
            for agent in self.agents:
                agent_old_location = deepcopy(agent.current_location)
                agent.current_location = agent.execute_action(agent_actions[agent.id])
                self.map.update_agent_location(
                    agent_old_location, agent.current_location
                )
                agent.mdp_handle.update(agent.current_location, self.map, True)
                agent.timer += 1
                self.timer += 1

    def within_range_agents(self) -> List[List[Agent]]:
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
                    self.map.get_manhattan_distance(agent_i_location, agent_j_location)
                    < self.communication_range
                ):
                    agent_bubbles[agent_i.id].append(agent_j)
        return agent_bubbles

    def update_min_costs(self, node: MultiAgentSearchNode, reward: np.float64):
        node.g = reward
        if node.parent is not None:
            self.update_min_costs(node.parent, reward)

    def multi_agent_search(
        self,
        target_agent: Agent,
        agent_bubbles: List[Agent],
        current_timestep: int,
        planning_horizon: int,
        shared_observations: List[Observation],
    ) -> Tuple[np.float64, Action]:
        """
        Performs the multi-agent search algorithm for a single agent assuming they are
        in communication range of other agents
        :param target_agent: The agent for which the multi-agent search algorithm is being performed
        :param agent_bubbles: List of agents that the agent is within communication range including itself
        :param current_timestep: Current timestep k
        :param planning_horizon: Planning horizon k + h
        :param shared_observations: List of shared observations between the agents inside the communication range
        """

        # Make the maps of the agents in the bubble consistent with each other
        for agent in agent_bubbles:
            agent.map.map = self.map.map

        root_node = MultiAgentSearchNode(
            None,
            {agent.id: [] for agent in agent_bubbles},
            {agent.id: agent.current_location for agent in agent_bubbles},
        )

        # TODO: Probably not needed to compute the h-value for the root node
        for agent_in_comm_range in agent_bubbles:
            best_reward, best_action = agent_in_comm_range.extract_action(
                agent_in_comm_range.current_location,
                current_timestep,
                current_timestep + planning_horizon,
                shared_observations,
            )
            root_node.h = np.add(root_node.h, best_reward)

        root_node.g = np.float64(0.0)

        open_set = queue.PriorityQueue()
        open_set.put(root_node)

        best_gain = np.float64(0.0)
        best_action = Action(ActionType.Wait, target_agent.current_location)

        while not open_set.empty():
            current = open_set.get()

            if current.f < best_gain:
                return best_gain, best_action

            if current.timestep >= planning_horizon:
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
                    for agent_in_comm_range in agent_bubbles:
                        next_pos = agent_in_comm_range.map.extract_next_location(
                            current.agent_locations[agent_in_comm_range.id],
                            action_prefixes[agent_in_comm_range.id][-1],
                        )
                        next_locations[agent_in_comm_range.id] = next_pos
                        if not agent_in_comm_range.map.valid_move(
                            current.agent_locations[agent_in_comm_range.id], next_pos
                        ):
                            invalid_action_prefix = True
                            break

                    if invalid_action_prefix:
                        continue

                    child_node = MultiAgentSearchNode(
                        current, action_prefixes, next_locations
                    )
                    child_node.timestep = current.timestep + 1
                    assert planning_horizon - child_node.timestep >= 0

                    # Extract the g-val and h-val for these nodes!

                    for agent_in_comm_range in agent_bubbles:
                        best_reward, best_action = agent_in_comm_range.extract_action(
                            agent_in_comm_range.current_location,
                            child_node.timestep,
                            planning_horizon - child_node.timestep,
                            shared_observations,
                        )
                        child_node.h = np.add(child_node.h, best_reward)

                    child_node.g = np.float64(
                        0.0
                    )  # Compute the combined multi-agent information gain

                    open_set.put(child_node)

        return self.extract_action(
            self.current_location,
            self.timer,
            self.timer + self.planning_horizon,
            self.mdp_handle.observations,
        )
