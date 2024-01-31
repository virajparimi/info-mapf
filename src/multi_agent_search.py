from __future__ import annotations

import copy
import itertools as iter
import queue
from typing import List, Tuple, Union

import numpy as np

from astar import astar
from map import Map


class SearchNode:
    """
    The list of action sequences of each agent that is represented by this node
    action_seqs[i] should be the sequence of actions taken by the ith agent
    """

    action_seqs = []

    """
    A lower bound on the amount of information that can be gained from this node. This value should be initially
    calculated as the multi-agent information gain from the path prefixes that this node represents
    """
    minimum_info_gain = 0

    # The current Multi-Agent information gain from the actions taken at this node
    current_info_gain = 0

    """
    An upper bound on the amount of information that can be gained from this node. Initially estimated as the sum of
    the single agent information gained with each agent starting at the end of these path prefixes. This is updated
    as the search explores and re-evaluates nodes deeper into the graph
    """
    maximum_info_gain = np.inf

    parent = None

    def __repr__(self):
        return (
            f"SearchNode Summary:\n\tAgent Action Prefixes: {self.action_seqs}"
            f"\n\tCurrent Information Gain: {self.current_info_gain}"
            f"\n\t Maximum Information Gain: {self.maximum_info_gain}"
        )

    # Defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.maximum_info_gain > other.maximum_info_gain

    # Defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.maximum_info_gain < other.maximum_info_gain

    def __init__(
        self,
        actions_taken: List[List[str]],
        parent: Union[SearchNode, None],
        min_info: float = 0.0,
        max_info: float = np.inf,
        current_gain: float = 0.0,
    ):
        self.action_seqs = actions_taken
        self.parent = parent

        self.minimum_info_gain = min_info
        self.maximum_info_gain = max_info
        self.current_info_gain = current_gain

    def get_children_seqs(
        self, agent_actions: Union[List[str], None] = None
    ) -> List[List[List[str]]]:
        """
        Return a list of action sequences corresponding to one step extensions of the node's path prefix
        :param agent_actions: The possible actions agents can take
        """

        if agent_actions is None:
            agent_actions = ["Right", "Left", "Up", "Down"]

        extensions = []
        for ext in iter.product(agent_actions, repeat=len(self.action_seqs)):
            new_ext = copy.deepcopy(self.action_seqs)
            for agent in range(len(self.action_seqs)):
                agent_path = new_ext[agent]
                agent_path.append(ext[agent])
            extensions.append(new_ext)

        return extensions

    def extract_paths(
        self, init_poses: List[Tuple[int, int]], map: Map
    ) -> Union[List[List[int]], None]:
        """
        Extract the positions of the path prefixes given the initial positions of the agents
        :param init_pos: initial positions of the agents
        """
        paths = []
        for i in range(len(self.action_seqs)):
            agent_path = [map.linearize_coordinate(init_poses[i][0], init_poses[i][1])]
            for j in range(len(self.action_seqs[i])):
                next_pos = map.extract_next_location(
                    agent_path[-1], self.action_seqs[i][j]
                )
                if next_pos is False:
                    return None
                agent_path.append(next_pos)
            paths.append(agent_path)
        return paths

    def get_path_size(self) -> int:
        """
        Returns the length of the path prefix for the first agent
        """
        return len(self.action_seqs[0])


def update_min_costs(node: SearchNode, reward: np.float64):
    node.minimum_info_gain = reward

    if node.parent is not None:
        update_min_costs(node.parent, reward)


def multi_agent_search(
    map: Map,
    init_poses: List[Tuple[int, int]],
    planning_horizon: int,
) -> Tuple[Union[List[List[int]], None], np.float64]:
    agent_paths = [[] for _ in range(0, len(init_poses))]

    open_set = queue.PriorityQueue()
    open_set.put(SearchNode(agent_paths, None))

    best_gain = np.float64(0.0)
    best_action_seqs = None

    while not open_set.empty():
        current = open_set.get()

        """
        If the top of our open set is doing worse than our best path so far, then we terminate because that
        means that even our optimistic estimates are doing worse than our current best path
        """
        if current.maximum_info_gain < best_gain:
            agents_paths = current.extract_paths(init_poses, map)
            return agents_paths, best_gain

        # Whenever we examine a new action node, we first check to see if we have reached our planning horizon
        if current.get_path_size() >= planning_horizon:
            # If so, then we want to update its parent paths with this newly found multi-agent information gain
            update_min_costs(current, current.current_info_gain)

            # Then, update our best path if this is in fact a better path
            if current.current_info_gain > best_gain:
                best_action_seqs = current.action_seqs
                best_gain = current.current_info_gain
                agents_paths = current.extract_paths(init_poses, map)
                assert agents_paths is not None
                reward = map.get_reward_from_traj(agents_paths)
                print(
                    f"Update Summary\n\tBest Action Sequences: {best_action_seqs}\n\t"
                    f"Best Gain: {best_gain}\n\t"
                    f"Agents Paths: {agents_paths}\n\t"
                    f"Reward: {reward}"
                )
        else:
            """
            If we have not reached a terminus, then we need to add the neighbors and compute their mutual_info_gain
            and then add it to the priority queue
            """

            children = current.get_children_seqs()
            for child in children:
                node = SearchNode(child, current)
                agents_paths = node.extract_paths(init_poses, map)
                if agents_paths is None:
                    continue

                # The reward thus far and the minimum reward received from this node as it accounts for other agents
                multi_agent_info_gain = map.get_reward_from_traj(agents_paths)

                """
                Now, we can compute the best information gain for each individual agent from this location. This will
                act as a heuristic of the information gain to come
                NOTE: I am not sure if we should calculate this mutual information gain with no prior observations, or
                assuming that we have seen the most likely observations
                """
                single_agent_gains = 0
                for single_agent_path in agents_paths:
                    time_steps_left = planning_horizon - len(single_agent_path)
                    if time_steps_left > 0:
                        start_from = single_agent_path[-1]
                        path = astar(map, start_from, time_steps_left)
                        single_agent_gains += map.get_reward_from_traj([path])

                """
                And then we update the node with this information
                NOTE: If we want to exploit the structure of the problem more, one strategy would be, instead of
                recalculating for each node the optimistic information gain, we can store our past heuristic
                calculations and search them to see if any path prefixes end at this location. But at an earlier
                time step, if one does, then we can use that node's optimistic info gain as a potential over-optimistic
                information gain estimate for this node.
                """
                node.current_info_gain = multi_agent_info_gain
                node.minimum_info_gain = multi_agent_info_gain
                node.maximum_info_gain = multi_agent_info_gain + single_agent_gains
                open_set.put(node)

    # Return None and best_gain if the open_set is empty
    return None, best_gain
