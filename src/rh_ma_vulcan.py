from map import Map
from agent import Agent
from typing import List
from copy import deepcopy


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
                    shared_observations = agent.mdp_handle.observations
                    for other_agents in agent_bubbles[idx]:
                        shared_observations += other_agents.mdp_handle.observations
                    shared_observations = list(
                        set(shared_observations)
                    )  # ensure uniqueness of the observations

                    horizon = min(
                        agent.planning_horizon, agent.mission_duration - self.timer
                    )

                    agent.mdp_handle.observations = shared_observations
                    for other_agents in agent_bubbles[idx]:
                        other_agents.mdp_handle.observations = shared_observations
                        # States of the MDP handle are not updated here as I dont think we need to

                    _, best_action = agent.multi_agent_search(
                        agent_bubbles[idx],
                        self.timer,
                        self.timer + horizon,
                        shared_observations,
                        self.map,
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
        agent_bubbles = [[] for i in range(len(self.agents))]
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
