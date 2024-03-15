import os
import sys
import math
import rospy
import logging
import numpy as np
from typing import Dict, List
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
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
        problem_type="empty_2",
    ):

        self.speed = speed
        self.turn_time = turn_time
        self.step_size = step_size
        self.base_command_topic = base_command_topic

        self.agent_namespaces = agent_namespaces
        self.agent_orientations = {agent_id: 0 for agent_id in agent_namespaces.keys()}
        self.agent_eulers = {
            agent_id: (0.0, 0.0, 0.0) for agent_id in agent_namespaces.keys()
        }
        self.current_positions = {
            agent_id: [0, 0] for agent_id in agent_namespaces.keys()
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
            5,
            len(agent_namespaces),
            20,
            5,
        )

        if problem_type == "empty_2":
            self.agent_locations = [(0, 0), (0, 2)]
        elif problem_type == "empty_3":
            self.agent_locations = [(0, 0), (0, 2), (2, 0)]
        elif problem_type == "maze_2":
            self.agent_locations = [(0, 0), (0, 3)]
        else:
            raise ValueError("Invalid problem type")

        if len(self.agent_locations) != self.num_agents:
            raise ValueError(
                "Number of agents and number of ROS namespaces do not match"
            )

        gp_locations = [(1, 1), (7, 7), (4, 4), (1, 7), (7, 1)]
        self.grid, self.reward_map = generate_map(
            self.rows,
            self.cols,
            agent_locations=self.agent_locations,
            gp_means=np.ones(len(gp_locations)).tolist(),
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
        else:
            raise ValueError("Invalid planner")

        self.agent_colors = ["r", "b", "g", "y", "m", "c", "k"]

        self.direction_map = {
            ActionType.Up.value: 0,
            ActionType.Left.value: 1,
            ActionType.Down.value: 2,
            ActionType.Right.value: 3,
        }

        self.angle_map = {
            0: 0.0,
            1: np.pi / 2.0,
            2: -np.pi,
            3: -np.pi / 2.0,
        }

    def get_rotation(self, msg):
        """
        Callback for the Odometry messages that updates the agent's orientation and position
        :param msg: Odometry message
        """
        orientation_q = msg.pose.pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        # If the yaw is close to 2 * pi or 0, set it to 0
        if np.allclose(yaw, 2 * np.pi, rtol=0.001):
            yaw = 0
        if np.allclose(yaw, 0, atol=0.001):
            yaw = 0

        for agent_id, namespace in self.agent_namespaces.items():
            if namespace in msg.header.frame_id:
                self.agent_eulers[agent_id] = (roll, pitch, yaw)
                self.current_positions[agent_id] = [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                ]
                break

    def run_planner(self) -> List[Action]:
        """
        Executes the planner for the mission duration
        """

        list_of_agent_actions = []
        while self.planner.timer < self.mission_duration and not rospy.is_shutdown():

            print("Current time: ", self.planner.timer)

            agent_actions = self.planner.single_step_planner(ros=True)
            list_of_agent_actions.append(agent_actions)
            assert agent_actions is not None

            print("Going to execute the actions: ", agent_actions)

            self.execute_turns(agent_actions)

            reorient = input("Reorient? (Y/n)")
            if reorient == "y" or reorient == "Y":
                self.execute_turns(agent_actions)

            self.execute_moves(agent_actions)

        return list_of_agent_actions

    def angle_difference(self, target_angle: float, current_angle: float) -> float:
        """
        Returns the difference between two angles and takes care of the wrap-around in radians
        :param target_angle: Target angle in radians
        :param current_angle: Current angle in radians
        """
        diff = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
        return diff if diff < math.pi else diff - (2 * math.pi)

    def execute_turns(self, agent_actions: Dict[int, Action]):
        """
        Executes the turns for the agents
        :param agent_actions: Dictionary of agent ids and their actions
        """

        intended_directions = {agent_id: 0 for agent_id in agent_actions.keys()}
        radians_to_move = {agent_id: 0.0 for agent_id in agent_actions.keys()}

        for agent_id, action in agent_actions.items():
            rotation_needed = int(
                (
                    self.direction_map[action.action_type.value]
                    - self.agent_orientations[agent_id]
                )
                % 4
            )
            intended_directions[agent_id] = self.direction_map[action.action_type.value]
            radians_to_move[agent_id] = self.angle_map[rotation_needed]

        turn_rate = rospy.Rate(25)

        all_done = False
        done = [False for _ in range(len(radians_to_move))]

        target_orientations = {
            agent_id: self.angle_map[self.agent_orientations[agent_id]]
            + radians_to_move[agent_id]
            for agent_id in radians_to_move.keys()
        }
        initial_diff = {
            agent_id: self.angle_difference(
                target_orientations[agent_id], self.agent_eulers[agent_id][2]
            )
            for agent_id in radians_to_move.keys()
        }
        initial_sign = {
            agent_id: math.copysign(1, initial_diff[agent_id])
            for agent_id in radians_to_move.keys()
        }
        while not all_done:
            for agent_id, radians in radians_to_move.items():
                current_diff = self.angle_difference(
                    target_orientations[agent_id], self.agent_eulers[agent_id][2]
                )
                current_sign = math.copysign(1, current_diff)
                if current_sign == initial_sign[agent_id] and not done[agent_id]:
                    turn_cmd = Twist()
                    turn_cmd.angular.z = 0.2 if initial_sign[agent_id] > 0 else -0.2
                    self.dispatchers[agent_id].publish(turn_cmd)
                    turn_rate.sleep()
                    done[agent_id] = False
                else:
                    stop_command = Twist()
                    self.dispatchers[agent_id].publish(stop_command)
                    done[agent_id] = True
                    self.agent_orientations[agent_id] = intended_directions[agent_id]
            for d in done:
                if not d:
                    all_done = False
                    break
                else:
                    all_done = True

    def execute_moves(self, agent_actions: Dict[int, Action]):
        """
        Executes the moves for the agents
        :param agent_actions: Dictionary of agent ids and their actions
        """

        all_done = False
        done = [False for _ in range(len(agent_actions))]
        for agent_id, action in agent_actions.items():
            if action.action_type.value != "Wait":
                forward_cmd = Twist()
                forward_cmd.linear.x = self.speed
                self.dispatchers[agent_id].publish(forward_cmd)

        start_positions = {
            agent_id: self.current_positions[agent_id]
            for agent_id in agent_actions.keys()
        }
        while not all_done:
            for agent_id, action in agent_actions.items():
                if action.action_type.value != "Wait":
                    distance_travelled = np.sqrt(
                        (
                            start_positions[agent_id][0]
                            - self.current_positions[agent_id][0]
                        )
                        ** 2
                        + (
                            start_positions[agent_id][1]
                            - self.current_positions[agent_id][1]
                        )
                        ** 2
                    )
                    if distance_travelled >= self.step_size:
                        stop_command = Twist()
                        self.dispatchers[agent_id].publish(stop_command)
                        done[agent_id] = True
            self.rate.sleep()

            for d in done:
                if not d:
                    all_done = False
                    break
                else:
                    all_done = True


if __name__ == "__main__":

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    node = PlanDispatchNode("cmd_vel", {0: "tb3_1", 1: "tb3_5", 2: "tb3_0"})
    list_of_agent_actions = node.run_planner()
    print("History of actions executed by the agents: ", list_of_agent_actions)

    # You can also run this with more agents and different planners
    # 1. "rh_ma_vulcan"
    # 2. "rh_mcts_ma_vulcan"
    # 3. "sa_ca_vulcan"
