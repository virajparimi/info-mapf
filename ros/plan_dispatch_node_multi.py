import rospy

import argparse

from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry

import numpy as np

import math
import os


import math


def euler_from_quaternion(x, y, z, w):
    """

    Convert a quaternion into euler angles (roll, pitch, yaw)

    roll is rotation around x in radians (counterclockwise)

    pitch is rotation around y in radians (counterclockwise)

    yaw is rotation around z in radians (counterclockwise)

    """

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class PlanDispatchNode:

    def __init__(
        self,
        topic_name,
        sub_name,
        name_space_list,
        rate,
        init_poses,
        init_rots,
        step_size,
        plan,
        speed=0.2,
        rot_vel=45,
        dist_variance=0.1,
        rad_variance=0.05,
        turn_time=4,
    ):

        self.dispatchers = []
        for n in name_space_list:
            print(f"making publisher: {n}/{topic_name}")
            self.dispatchers.append(
                rospy.Publisher(n + "/" + topic_name, Twist, queue_size=10)
            )

        rospy.init_node("InfoMAPFDispatch")
        self.rate = rospy.Rate(rate)
        self.current_orientations = init_rots
        self.step_size = step_size
        self.dist_var = dist_variance
        self.rad_var = rad_variance
        self.mode = "wait"
        self.plan = plan
        self.speed = speed
        self.rot_vel = math.radians(rot_vel)
        self.name_space_list = name_space_list
        self.turn_time = turn_time

    def execute_turn_to_directions(self, acts):
        diff = []
        for i, c in enumerate(self.current_orientations):
            a = acts[i]
            if a == "up":
                diff.append(0)

            if a == "down":
                diff.append(math.radians(180))

            if a == "left":
                diff.append(math.radians(270))

            if a == "right":
                diff.append(math.radians(90))

        for i, d in enumerate(self.dispatchers):
            dist = diff[i] - self.current_orientations[i]
            speed = dist / self.turn_time
            turn_cmd = Twist()
            turn_cmd.angular.z = speed
            print(f"commanding {i} to {diff[i]} with speed {speed}")
            d.publish(turn_cmd)
        rospy.sleep(self.turn_time)
        stop_cmd = Twist()
        for i, d in enumerate(self.dispatchers):
            d.publish(stop_cmd)
            self.current_orientations = diff[i]

    def execute_plan(self):
        while len(self.plan) > 0:
            next_acts = self.plan.pop(0)
            self.execute_turn_to_directions(next_acts)

            forward_cmd = Twist()
            forward_cmd.linear.x = self.speed
            for d in self.dispatchers:
                d.publish(forward_cmd)
            rospy.sleep((self.step_size / self.speed))

            stop_cmd = Twist()
            for d in self.dispatchers:
                d.publish(stop_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name")
    args = parser.parse_args()

    plan_test = [["up", "down"], ["left", "right"]]

    node = PlanDispatchNode(
        "cmd_vel", "odom", ["tb3_1", "tb3_2"], 10, 0.5, [0, 0], 0.5, plan_test
    )
    node.execute_plan()
