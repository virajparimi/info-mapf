import os
import sys
import rospy
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from scipy.stats import multivariate_normal

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA
from map import Grid, RewardMap, Parameters, Action  # NOQA


class PlanDispatchNode:

    def __init__(
        self,
        base_command_topic: str,
        agent_namespaces: Dict[int, str],
        speed: float = 0.2,  # m/s
        publish_rate: int = 10,
        step_size: float = 1.0,  # meters
        turn_time: float = 2.0,  # seconds
    ):

        self.dispatchers = {}
        for agent_id, namespace in agent_namespaces.items():
            print(f"Generating publisher: {namespace}/{base_command_topic}")
            self.dispatchers[agent_id] = rospy.Publisher(
                namespace + "/" + base_command_topic, Twist, queue_size=10
            )

        rospy.init_node("InfoMAPFDispatch")
        self.rate = rospy.Rate(publish_rate)
        self.speed = speed
        self.turn_time = turn_time
        self.step_size = step_size
        self.base_command_topic = base_command_topic
        self.agent_namespaces = agent_namespaces
        self.agent_orientations = {
            agent_id: 0.0 for agent_id in agent_namespaces.keys()
        }

        # Multi-Agent Vulcan Setup
        self.params = Parameters(
            theta_1=np.float64(0.4),
            theta_2=np.float64(0.01),
            u_tilde=np.float64(1.4),
            P_1=np.float64(0.98),
            P_2=np.float64(0.002),
            J=np.int64(5),
            measurement_noise=np.float64(0.2),
            distance_simplification=True,
        )

        (
            self.rows,
            self.cols,
            self.max_gps,
            self.num_agents,
            self.mission_duration,
            self.communication_range,
        ) = (
            11,
            11,
            4,
            2,
            35,
            5,
        )

        self.agent_locations = [(0, 0), (10, 10)]

        self.grid, self.reward_map = generate_map(
            self.rows,
            self.cols,
            agent_locations=self.agent_locations,
            gp_means=np.ones(self.max_gps).tolist(),
            gp_locations=[(1, 1), (8, 2), (5, 5), (2, 8)],
            parameters=self.params,
        )

        self.vulcan_agents = []
        for agent in range(self.num_agents):
            agent_location_linearized = self.grid.linearize_coordinate(
                self.agent_locations[agent][0], self.agent_locations[agent][1]
            )
            vulcan_agent = Agent(
                id=agent,
                start_location=agent_location_linearized,
                grid=self.grid,
                reward_map=self.reward_map,
                mission_duration=self.mission_duration,
            )
            self.vulcan_agents.append(vulcan_agent)

        self.rh_ma_vulcan = MultiAgentVulcan(
            grid=self.grid,
            reward_map=self.reward_map,
            agents=self.vulcan_agents,
            communication_range=self.communication_range,
        )

        self.agent_colors = ["r", "b", "g", "y", "m", "c", "k"]

        x = np.linspace(-1, self.reward_map.num_of_rows + 1, 1000)
        y = np.linspace(-1, self.reward_map.num_of_cols + 1, 1000)
        xx, yy = np.meshgrid(x, y)
        meshgrid = np.dstack((xx, yy))
        self.zz = np.zeros_like(xx)
        for i in range(len(self.reward_map.locations)):
            linear_location = self.reward_map.locations[i]
            location_coord = self.reward_map.get_coordinate(linear_location)
            gaussian = self.reward_map.means[i] * multivariate_normal.pdf(
                meshgrid, mean=location_coord, cov=1
            )
            self.zz += gaussian
        self.zz /= np.max(self.zz)

    def run_planner(self):

        while (
            self.rh_ma_vulcan.timer < self.mission_duration and not rospy.is_shutdown()
        ):
            agent_actions = self.rh_ma_vulcan.single_step_planner(ros=True)
            assert agent_actions is not None
            for agent_id, action in agent_actions.items():
                print(
                    f"Agent {agent_id} is executing action {action.action_type.value}"
                )
            self.execute_turns(agent_actions)
            self.execute_moves(agent_actions)

        vulcan_agents_paths = []
        for idx, agent in enumerate(self.vulcan_agents):
            print("Path for agent ", agent.id)
            vulcan_path = []
            for v_location in agent.visited_locations:
                vulcan_path.append(self.grid.get_coordinate(v_location))
            print(vulcan_path)
            plt.plot(
                [x[1] for x in vulcan_path],
                [x[0] for x in vulcan_path],
                self.agent_colors[idx] + "--",
                alpha=0.7,
            )
            vulcan_agents_paths.append(vulcan_path)

        plt.imshow(
            self.zz,
            extent=(
                -1,
                self.reward_map.num_of_rows + 1,
                self.reward_map.num_of_cols + 1,
                -1,
            ),
            cmap="hot",
        )
        plt.show()

    def execute_turns(self, agent_actions: Dict[int, Action]):

        radians_to_move = {agent_id: 0.0 for agent_id in agent_actions.keys()}
        for agent_id, action in agent_actions.items():
            if action.action_type.value == "Up":
                radians_to_move[agent_id] = 0.0
            elif action.action_type.value == "Down":
                radians_to_move[agent_id] = np.pi
            elif action.action_type.value == "Left":
                radians_to_move[agent_id] = np.pi / 2.0
            elif action.action_type.value == "Right":
                radians_to_move[agent_id] = 3 * np.pi / 2.0
            else:
                radians_to_move[agent_id] = self.agent_orientations[agent_id]

        for agent_id, radians in radians_to_move.items():
            radians = radians - self.agent_orientations[agent_id]
            speed = radians / self.turn_time
            turn_cmd = Twist()
            turn_cmd.angular.z = speed
            print(f"Commanding {agent_id} to {radians} with speed {speed}")
            self.dispatchers[agent_id].publish(turn_cmd)

        rospy.sleep(self.turn_time)

        stop_command = Twist()
        for agent_id, dispatcher in self.dispatchers.items():
            dispatcher.publish(stop_command)
            self.agent_orientations[agent_id] = radians_to_move[agent_id]

    def execute_moves(self, agent_actions: Dict[int, Action]):

        for agent_id, action in agent_actions.items():
            if action.action_type.value != "Wait":
                forward_cmd = Twist()
                forward_cmd.linear.x = self.speed
                self.dispatchers[agent_id].publish(forward_cmd)

        rospy.sleep((self.step_size / self.speed))

        stop_command = Twist()
        for agent_id, dispatcher in self.dispatchers.items():
            dispatcher.publish(stop_command)


if __name__ == "__main__":

    node = PlanDispatchNode("cmd_vel", {0: "tb3_1", 1: "tb3_5"})
    node.run_planner()
