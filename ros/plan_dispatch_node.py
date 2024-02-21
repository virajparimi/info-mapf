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



    def __init__(self, topic_name, sub_name, name_space_list, rate, init_pos, step_size, plan,
                 speed=0.2, rot_vel=45, dist_variance=0.1, rad_variance=0.05):

        self.dispatcher = rospy.Publisher(topic_name, Twist, queue_size=10)
        rospy.init_node('InfoMAPFDispatch')
        self.rate = rospy.Rate(rate)
        self.monitor = rospy.Subscriber(sub_name, Odometry, self.update_odom)
        self.current_position=init_pos
        self.current_orientation=0
        self.target_orientation = 0
        self.tagret_position=init_pos
        self.step_size = step_size
        self.dist_var = dist_variance
        self.rad_var = rad_variance
        self.mode = "wait"
        self.plan = plan
        self.speed = speed
        self.rot_vel=math.radians(rot_vel)
        self.name_space_list=name_space_list


    def execute_plan_step(self):



    def execute_action(self, a):
        print(f"executing {a}")

        self.target_idx = 1

        if a == "up":

            self.target_orientation = 0

            self.target_position = (self.current_position[0]+self.step_size, self.current_position[1])

            self.target_idx = 0

        if a == "down":

            self.target_orientation = math.radians(180)

            self.target_position = (self.current_position[0]-self.step_size, self.current_position[1])

            self.target_idx = 0

        if a == "left":

            self.target_orientation = math.radians(270)

            self.target_position = (self.current_position[0], self.current_position[1]-self.step_size)

        if a == "right":

            self.target_orientation = math.radians(90)

            self.target_position = (self.current_position[0], self.current_position[1]+self.step_size)



        
        print(self.current_orientation - self.target_orientation)


        


        stop_cmd = Twist()

        self.dispatcher.publish(stop_cmd)



    def execute_next_action(self):
        if len(self.plan) > 0:
            a = self.plan.pop(0)
            self.execute_action(a)
            self.mode = "turn"

    def spin(self):
        rospy.spin()

    def update_odom(self, msg):

        data = msg.pose.pose

        self.current_position = (data.position.x, data.position.y)

        euler = euler_from_quaternion(data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w)

        self.current_orientation = euler[2]
        print(f"curent pos: {self.current_position}\ncurrent orientation: {self.current_orientation}")

        stop_cmd = Twist()
        
        if self.mode == "turn":
            move_cmd = Twist()
            move_cmd.angular.z = math.radians(15)
            print("moving")
            if -self.rad_var > self.current_orientation - self.target_orientation or self.current_orientation-self.target_orientation > self.rad_var:
                self.dispatcher.publish(move_cmd)
                print(self.current_orientation - self.target_orientation)
                #self.rate.sleep()
            else:
                self.mode= "move"
                self.dispatcher.publish(stop_cmd)
        elif self.mode == "move":
            forward_cmd = Twist()
            forward_cmd.linear.x = 0.05
            if -self.dist_var > self.current_position[self.target_idx] - self.target_position[self.target_idx] or  self.current_position[self.target_idx] - self.target_position[self.target_idx]  > self.dist_var:
                print(self.current_position[self.target_idx])
                self.dispatcher.publish(forward_cmd)
            else:
                self.mode = "wait"
                self.dispatcher.publish(stop_cmd)
                self.execute_next_action()

            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name')
    args = parser.parse_args()
    os.environ['ROS_NAMESPACE'] = args.name
    
    plan_test = ["up","down", "left", "right"]

    node = PlanDispatchNode("tb3_1/cmd_vel", "tb3_1/odom", 10, (0,0), 0.5, plan_test)
    node.execute_next_action()
    node.spin()
