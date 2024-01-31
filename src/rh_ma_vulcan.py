from typing import List

from agent import Agent
from map import Map


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
            # Collect agents within communication range
            agent_bubbles = self.within_range_agents()
            agent_actions = []
            # Command each agent to execute their adaptive search algorithm for one step
            for idx, agent in enumerate(self.agents):
                if len(agent_bubbles[idx]) > 0:
                    # Start multi-agent search algorithm with respect to this agent
                    shared_observations = agent.mdp_handle.observations
                    for other_agent in agent_bubbles[idx]:
                        shared_observations += other_agent.mdp_handle.observations

                    _, best_action = agent.multi_agent_search(
                        agent_bubbles[idx], shared_observations
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
                agent.current_location = agent.execute_action(agent_actions[agent.id])
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
