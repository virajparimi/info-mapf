import os
import sys
import rospy
import logging
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from scipy.stats import multivariate_normal
from tf.transformations import euler_from_quaternion

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA
from rh_sa_vulcan import SingleAgentVulcan  # NOQA
from map import Grid, RewardMap, Parameters, Action, ActionType  # NOQA


class PlanDispatchNode:

    def __init__(
        self,
        base_command_topic: str,
        agent_namespaces: Dict[int, str],
        speed: float = 0.1,  # m/s
        publish_rate: int = 10,
        step_size: float = 0.3,  # meters
        turn_time: float = 10.0,  # seconds
        planner: str = "rh_ma_vulcan",
    ):

        self.speed = speed
        self.turn_time = turn_time
        self.step_size = step_size
        self.base_command_topic = base_command_topic
        self.agent_namespaces = agent_namespaces
        self.agent_orientations = {
            agent_id: 0.0 for agent_id in agent_namespaces.keys()
        }
        self.agent_eulers = {
            agent_id: (0.0, 0.0, 0.0) for agent_id in agent_namespaces.keys()
        }

        self.dispatchers, self.subscribers = {}, {}
        for agent_id, namespace in agent_namespaces.items():
            print(f"Generating publisher: {namespace}/{base_command_topic}")
            self.dispatchers[agent_id] = rospy.Publisher(
                namespace + "/" + base_command_topic, Twist, queue_size=10
            )
            print(f"Generating subscriber: {namespace}/odom")
            self.subscribers[agent_id] = rospy.Subscriber(
                namespace + "/odom", Odometry, self.get_rotation
            )

        rospy.init_node("InfoMAPFDispatch")
        self.rate = rospy.Rate(publish_rate)

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
            8,
            8,
            4,
            len(agent_namespaces),
            25,
            5,
        )

        self.agent_locations = [(0, 0), (7, 7)]

        if self.num_agents > len(self.agent_locations) and self.num_agents == 3:
            self.agent_locations += [(0, 7)]
            self.max_gps = 5
        elif self.num_agents > len(self.agent_locations) and self.num_agents == 4:
            self.agent_locations += [(7, 0), (7, 0)]
            self.max_gps = 6

        gp_locations = [(1, 1), (6, 2), (5, 5), (2, 6)]
        if self.max_gps == 5:
            gp_locations += [(1, 6)]
        elif self.max_gps == 6:
            gp_locations += [(1, 6), (6, 1)]

        self.grid, self.reward_map = generate_map(
            self.rows,
            self.cols,
            agent_locations=self.agent_locations,
            gp_means=np.ones(self.max_gps).tolist(),
            gp_locations=gp_locations,
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

        if planner == "rh_ma_vulcan":
            self.planner = MultiAgentVulcan(
                grid=self.grid,
                reward_map=self.reward_map,
                agents=self.vulcan_agents,
                communication_range=self.communication_range,
            )
        elif planner == "rh_mcts_ma_vulcan":
            self.planner = MultiAgentVulcan(
                grid=self.grid,
                reward_map=self.reward_map,
                agents=self.vulcan_agents,
                communication_range=self.communication_range,
                use_mcts=True,
            )
        elif planner == "sa_ca_vulcan":
            self.planner = SingleAgentVulcan(
                grid=self.grid,
                reward_map=self.reward_map,
                agents=self.vulcan_agents,
            )

        self.agent_colors = ["r", "b", "g", "y", "m", "c", "k"]

        x = np.linspace(0, self.reward_map.num_of_cols, 1000)
        y = np.linspace(0, self.reward_map.num_of_rows, 1000)
        xx, yy = np.meshgrid(x, y)
        meshgrid = np.dstack((xx, yy))
        self.zz = np.zeros_like(xx)
        for i in range(len(self.reward_map.locations)):
            linear_location = self.reward_map.locations[i]
            location_coord = self.reward_map.get_coordinate(linear_location)
            location_coord = np.array([location_coord[1], location_coord[0]])
            gaussian = self.reward_map.means[i] * multivariate_normal.pdf(
                meshgrid, mean=location_coord, cov=1
            )
            self.zz += gaussian
        self.zz /= np.max(self.zz)

    def get_rotation(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        if roll < 0:
            roll += 2 * np.pi
        if pitch < 0:
            pitch += 2 * np.pi
        if yaw < 0:
            yaw += 2 * np.pi

        if np.allclose(roll, 2 * np.pi, rtol=0.001):
            roll = 0
        if np.allclose(pitch, 2 * np.pi, rtol=0.001):
            pitch = 0
        if np.allclose(yaw, 2 * np.pi, rtol=0.001):
            yaw = 0
        for agent_id, namespace in self.agent_namespaces.items():
            if namespace in msg.header.frame_id:
                self.agent_eulers[agent_id] = (roll, pitch, yaw)
                self.agent_orientations[agent_id] = yaw
                break

    def run_planner(self):

        while self.planner.timer < self.mission_duration and not rospy.is_shutdown():
            agent_actions = self.planner.single_step_planner(ros=True)
            assert agent_actions is not None

            # For debugging only!
            # agent_action_a = input("Enter action for agent 0: ")
            # if agent_action_a == "Up":
            #     agent_action_a = Action(action_type=ActionType.Up, location=0)
            # elif agent_action_a == "Down":
            #     agent_action_a = Action(action_type=ActionType.Down, location=0)
            # elif agent_action_a == "Left":
            #     agent_action_a = Action(action_type=ActionType.Left, location=0)
            # elif agent_action_a == "Right":
            #     agent_action_a = Action(action_type=ActionType.Right, location=0)
            # else:
            #     agent_action_a = Action(action_type=ActionType.Wait, location=0)
            # agent_action_b = input("Enter action for agent 1: ")
            # if agent_action_b == "Up":
            #     agent_action_b = Action(action_type=ActionType.Up, location=0)
            # elif agent_action_b == "Down":
            #     agent_action_b = Action(action_type=ActionType.Down, location=0)
            # elif agent_action_b == "Left":
            #     agent_action_b = Action(action_type=ActionType.Left, location=0)
            # elif agent_action_b == "Right":
            #     agent_action_b = Action(action_type=ActionType.Right, location=0)
            # else:
            #     agent_action_b = Action(action_type=ActionType.Wait, location=0)

            # agent_actions = {0: agent_action_a, 1: agent_action_b}
            print(agent_actions)

            con = input("Continue? (Y/n)")
            if con == "n":
                break

            self.execute_turns(agent_actions)
            print("Turns executed")
            print("Executing moves")
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
                0,
                self.reward_map.num_of_cols,
                self.reward_map.num_of_rows,
                0,
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

        turn_rate = rospy.Rate(25)

        first = True
        all_done = False
        done = [False for _ in range(len(radians_to_move))]
        while not all_done:
            for agent_id, radians in radians_to_move.items():
                print(f"Agent {agent_id} is at {self.agent_eulers[agent_id][2]}")
                print(f"Agent {agent_id} is going to {radians}")
                if first:
                    if np.allclose(radians, self.agent_eulers[agent_id][2], atol=0.05):
                        done[agent_id] = True

                if (
                    not np.allclose(radians, self.agent_eulers[agent_id][2], atol=0.05)
                    and not done[agent_id]
                ):
                    turn_cmd = Twist()
                    turn_cmd.angular.z = 0.3
                    self.dispatchers[agent_id].publish(turn_cmd)
                    turn_rate.sleep()
                    done[agent_id] = False
                else:
                    stop_command = Twist()
                    self.dispatchers[agent_id].publish(stop_command)
                    done[agent_id] = True
            for d in done:
                if not d:
                    all_done = False
                    break
                else:
                    all_done = True
            first = False

        # for agent_id, radians in radians_to_move.items():
        #     # radians = radians - self.agent_orientations[agent_id]
        #     # speed = radians / self.turn_time
        #     # turn_cmd = Twist()
        #     # turn_cmd.angular.z = speed
        #     # print(f"Commanding {agent_id} to {radians} with speed {speed}")
        #     # self.dispatchers[agent_id].publish(turn_cmd)

        #     # if agent_id == 0:
        #     #     continue

        #     # radians = radians - self.agent_orientations[agent_id]
        #     commanded_z = abs(0.1 * (radians - self.agent_eulers[agent_id][2]))
        #     print(f"Euler: {self.agent_eulers[agent_id]}")
        #     print(f"Going to {radians}")
        #     while abs(self.agent_eulers[agent_id][2] - radians) > 0.05:
        #         commanded_z = abs(0.1 * (radians - self.agent_eulers[agent_id][2]))
        #         turn_cmd = Twist()
        #         turn_cmd.angular.z = 0.2
        #         print(f"Commanding {agent_id} to {radians} with speed {commanded_z}")
        #         print(f"Current orientation: {self.agent_eulers[agent_id]}")
        #         self.dispatchers[agent_id].publish(turn_cmd)
        #         print(self.dispatchers[agent_id].get_num_connections())
        #         turn_rate.sleep()

        #     stop_command = Twist()
        #     for agent_id, dispatcher in self.dispatchers.items():
        #         dispatcher.publish(stop_command)
        #         # self.agent_orientations[agent_id] = radians_to_move[agent_id]

        # rospy.sleep(self.turn_time)
        print("Turns executed inside ")

    def execute_moves(self, agent_actions: Dict[int, Action]):

        for agent_id, action in agent_actions.items():
            if action.action_type.value != "Wait":
                print(f"Commanding {agent_id} to move {action.action_type.value}")
                forward_cmd = Twist()
                forward_cmd.linear.x = self.speed
                self.dispatchers[agent_id].publish(forward_cmd)

        rospy.sleep((self.step_size / self.speed))

        stop_command = Twist()
        for agent_id, dispatcher in self.dispatchers.items():
            dispatcher.publish(stop_command)


if __name__ == "__main__":

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    node = PlanDispatchNode("cmd_vel", {0: "tb3_1", 1: "tb3_5"})
    node.run_planner()

    # You can also run this with more agents and different planners
    # 1. "rh_ma_vulcan"
    # 2. "rh_mcts_ma_vulcan"
    # 3. "sa_ca_vulcan"

    # When recording the videos make sure to "clap" once the agents take a step to help with video synchronization
