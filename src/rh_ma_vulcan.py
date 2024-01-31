from map import Map
from agent import Agent
from typing import List


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
            # Command each agent to execute their adaptive search algorithm for one step
            for agent in self.agents:
                horizon = min(
                    agent.planning_horizon, agent.mission_duration - self.timer
                )
                _, best_action = agent.extract_action(
                    agent.current_location,
                    self.timer,
                    self.timer + horizon,
                    agent.mdp_handle.observations,
                )
                agent.current_location = agent.execute_action(best_action)
                agent.mdp_handle.update(agent.current_location, self.map, True)
                agent.timer += 1
            self.timer += 1

    def within_range_agents(self) -> List[Agent]:
        """
        Returns a list of agents within communication range
        """
        pass
