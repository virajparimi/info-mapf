#This file is for the actual search algorithm, not including the information gain calculations

import numpy as np
import itertools as iter
import queue
from dataclasses import dataclass, field
from typing import Any
from astar import astar
import copy

class SearchNode:

    #The list of action sequences of each agent that is represented by this node
    #action_seqs[i] should be the sequence of actions taken by the ith agent
    action_seqs = []

    #A lower bound on the amount of information that can be gained from this node
    #This value should be initially calculated as the multi-agent information gain from the path prefixes that this node
    #represents
    minimum_info_gain = 0

    #The current Multi-Agent information gain from the actions taken at this node
    current_info_gain = 0

    #An upper bound on the amount of information that can be gained from this node
    #Initially estimated as the sum of the single agent information gained with each agent starting at the end of these path prefixes
    #Is updated as the search explores and re-evaluates nodes deeper into the graph
    maximum_info_gain = np.inf
    parent = None

    def __repr__(self):
        return f"{self.action_seqs} - current: {self.current_info_gain} max: {self.maximum_info_gain}"

    def __init__(self, actions_taken, parent, min_info=0, max_info=np.inf, current_gain=0):
        self.action_seqs = actions_taken
        self.minimum_info_gain = min_info
        self.maximum_info_gain = max_info
        self.current_info_gain = current_gain
        self.parent = parent

    def get_children_seqs(self, agent_actions):
        """
        Return all possible path prefix extensions.
        :param agent_actions: The possible actions agents can take
        :return: A list of action sequences corresponding to one step extensions of this node's path prefix
        """

        extensions = []
        exts = []
        for ext in iter.product(agent_actions, repeat=len(self.action_seqs)):
            new_ext = copy.deepcopy(self.action_seqs)
            exts.append(ext)
            for i in range(0, len(self.action_seqs)):
                agent_path = new_ext[i]
                agent_path.append(ext[i])
            extensions.append(new_ext)

        return extensions

    def extract_paths(self, init_pos, action_model):
        """
        Extract the positions of the path prefixes given the initial positions of the agents
        :param init_pos: initial positions of the agents
        :return: the path prefixes as locations
        """
        paths = []
        for i in range(0, len(self.action_seqs)):
            agent_path = [init_pos[i]]
            for j in range(0, len(self.action_seqs[i])):
                next_pos = action_model(self.action_seqs[i][j], agent_path[-1])
                agent_path.append(next_pos)
            paths.append(agent_path)
        return paths


    def get_path_size(self):
        return len(self.action_seqs[0])

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.maximum_info_gain > other.maximum_info_gain

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.maximum_info_gain < other.maximum_info_gain


def update_min_costs(node, reward):
    node.minimum_info_gain = reward

    if node.parent is not None:
        update_min_costs(node.parent, reward)

def multi_agent_search(agent_actions, map, init_pos, planning_horizon, num_agents, action_model):

    open_set = queue.PriorityQueue()
    agent_paths = []
    for _ in range(0, num_agents):
        agent_paths.append([])
    open_set.put(SearchNode(agent_paths, None))
    best_paths = None
    best_path_gain = 0

    while not open_set.empty():

        current = open_set.get()
        #If the top of our open set doing worst than our best path so far, then
        #we terminate because that means that even our optimistic estimates are doing worse
        #than our current best path
        if current.maximum_info_gain < best_path_gain:
            return best_paths, best_path_gain

        #whenever we examine a new action node, we first check to see if we have reached our planning horizon
        if current.get_path_size() >= planning_horizon:
            #if so, then we want to update its parent paths with this newly found multi-agent information gain
            update_min_costs(current, current.current_info_gain)

            #then, update our best path if this is in fact a better path
            if current.current_info_gain > best_path_gain:
                best_paths = current.action_seqs
                best_path_gain = current.current_info_gain
                location_path = current.extract_paths(init_pos, action_model)
                reward = map.get_reward_from_traj(location_path)
                print(f"paths: {best_paths} gives {best_path_gain} from going to {location_path} giving {reward}")
        else:
            #If we have not reached a terminus, then we need to add the neighbors and compute their mutual_info_gain, and then add it to the priority queue
            children = current.get_children_seqs(agent_actions)
            for c in children:
                node = SearchNode(c, current)
                location_paths = node.extract_paths(init_pos, action_model)
                is_in_bounds = True
                for l in location_paths:
                    end_point = l[-1]
                    if 0 > end_point[0] or end_point[0] >= map.height or 0 > end_point[1] or end_point[1] >= map.width:
                        is_in_bounds = False
                if not is_in_bounds:
                    continue
                #the reward thus far, and therefore the minimum reward received from this node because it accounts for the other agents
                multi_agent_info_gain = map.get_reward_from_traj(location_paths)

                #Now, we can compute the best information gain for each individual agent from this location
                #this will act as a heuristic of the information gain to come
                #NOTE: I am not sure if we should calculate this mutual information gain with no prior observations, or
                #assuming that we have seen the most likely observations
                single_agent_gains = 0
                for l in location_paths:
                    time_steps_left = planning_horizon - len(l)
                    if time_steps_left > 0:
                        start_from = l[-1]
                        path = astar(map, start_from, time_steps_left)
                        single_agent_gains += map.get_reward_from_traj([path])

                #And then we update the node with this information
                #NOTE: If we want to exploit the structure of the problem more, one strategy would be, instead of
                #recalculating for each node the optimistic information gain, we can store our past heuristic calculations
                #and search them to see if any path prefixes end at this location. but at an earlier time step
                #If one does, then we can use that node's optimistic info gain as a potentially over-optimistic info gain estimate
                #for this node.
                #
                node.current_info_gain = multi_agent_info_gain
                node.minimum_info_gain = multi_agent_info_gain
                node.maximum_info_gain = multi_agent_info_gain+single_agent_gains
                open_set.put(node)















