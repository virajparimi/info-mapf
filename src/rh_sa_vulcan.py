from __future__ import annotations

import logging
import numpy as np
from agent import Agent
from copy import deepcopy
from typing import List, Dict
from map import Grid, RewardMap, ActionType


class SingleAgentVulcan(object):
    def __init__(
        self,
        grid: Grid,
        reward_map: RewardMap,
        agents: List[Agent],
    ):
        self.timer = 0
        self.grid = grid
        self.agents = agents
        self.reward_map = reward_map
        self.mission_duration = max([agent.mission_duration for agent in agents])

    def collision_check(
        self, old_locations: Dict[int, int], new_locations: Dict[int, int]
    ):

        for agent_i in self.agents:
            for agent_j in self.agents:
                if agent_i.id != agent_j.id:
                    agent_i_location = new_locations[agent_i.id]
                    agent_j_location = new_locations[agent_j.id]

                    # Vertex collision
                    if agent_i_location == agent_j_location:
                        return True
                    # Edge collision
                    if (
                        old_locations[agent_i.id] == agent_j_location
                        and old_locations[agent_j.id] == agent_i_location
                    ):
                        return True
        return False

    def planner(self):
        while self.timer < self.mission_duration:
            logging.info(f"Time = {self.timer}")

            agent_actions = {}
            agent_next_locations = {}
            for idx, agent in enumerate(self.agents):

                # Re-use vulcan for this single agent
                horizon = min(
                    agent.planning_horizon, agent.mission_duration - agent.timer
                )
                action_rewards, actions = agent.extract_action(
                    agent.current_location,
                    agent.timer,
                    agent.timer + horizon,
                    agent.mdp_handle.observations,
                    deepcopy(agent.grid),
                    agent.reward_map,
                    extract_all_actions=True,
                )

                assert isinstance(actions, list)
                assert isinstance(action_rewards, np.ndarray)

                agent_action_mapping = {
                    action.action_type.value: action_rewards[idx]
                    for idx, action in enumerate(actions)
                }
                agent_action_mapping = dict(
                    sorted(
                        agent_action_mapping.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                agent_next_locations[agent.id] = {
                    action.action_type.value: action.location for action in actions
                }
                agent_actions[agent.id] = agent_action_mapping

            min_actions = min(
                [len(agent_next_locations[agent.id]) for agent in self.agents]
            )

            action_idx = 0
            collision = True

            valid_actions = [ActionType.Wait.value for _ in self.agents]
            while collision and action_idx < min_actions:

                # Once we have extracted the best actions for each agent, we execute them
                old_locations = {
                    agent.id: agent.current_location for agent in self.agents
                }
                new_locations = {agent.id: -1 for agent in self.agents}

                for agent_id, agent_action_mapping in agent_actions.items():
                    local_idx = 0
                    for action_type, _ in agent_action_mapping.items():
                        if local_idx == action_idx:
                            valid_actions[agent_id] = action_type
                            new_locations[agent_id] = agent_next_locations[agent_id][
                                action_type
                            ]
                            break
                        local_idx += 1

                collision = self.collision_check(old_locations, new_locations)
                action_idx += 1

            if collision:
                valid_actions = [ActionType.Wait.value for _ in self.agents]

            # Once we have extracted the best actions for each agent, we execute them
            old_locations = [agent.current_location for agent in self.agents]
            old_locations_coords = [
                self.grid.get_coordinate(location) for location in old_locations
            ]
            new_locations = []
            for agent in self.agents:
                if valid_actions[agent.id] not in agent_next_locations[agent.id]:
                    assert valid_actions[agent.id] == ActionType.Wait.value
                    new_locations.append(agent.current_location)
                else:
                    new_locations.append(
                        agent_next_locations[agent.id][valid_actions[agent.id]]
                    )
            new_locations_coords = [
                self.grid.get_coordinate(location) for location in new_locations
            ]

            for old_loc in old_locations_coords:
                self.grid.grid[old_loc[0], old_loc[1]] = True
            for new_loc in new_locations_coords:
                self.grid.grid[new_loc[0], new_loc[1]] = False

            for agent in self.agents:
                agent.current_location = agent_next_locations[agent.id][
                    valid_actions[agent.id]
                ]
                agent.visited_locations.append(agent.current_location)
                agent.mdp_handle.update(agent.current_location, agent.reward_map)
                agent.timer += 1
            self.timer += 1
