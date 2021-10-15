#!/usr/bin/env python

import rospy

import numpy as np
from typing import Tuple

# Sensor message types
from sensor_msgs.msg import LaserScan

# the velocity command message
from geometry_msgs.msg import Twist

def scan_msg_to_points(scan_msg: LaserScan)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Transform distances into coordinates of points
    # Get distances and corresponding thetas as 1D np arrays
    distances = np.array(scan_msg.ranges)
    thetas = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))

    # Transform distances and thetas into 2D array of points with x and y relative to lidar
    # Positive x is from lidar to front of turtlebot
    # Positive y is from lidar to left of turtlebot
    # Each row is a point. Col 0 is x. Col 1 is y.
    distances_x = distances * np.cos(thetas)
    print("d", distances_x)
    distances_y = distances * np.sin(thetas)
    points = np.vstack((distances_x, distances_y)).T
    print("p", points.shape)
    return points, distances, thetas

def lidar_callback(scan_msg):
    # Get points from lidar
    points, distances, thetas = scan_msg_to_points(scan_msg)

    # Set thresholds for collision in meters
    # y collision is set such that robot will stop if there are any points within collision_y
    # on either the left OR right side
    robot_width = 3.06
    extra_spacing_x = -0.2
    collision_x = 1
    collision_y = robot_width/2 + extra_spacing_x

    # Check if any points hit the collision thresholds with a boolean matrix
    # Values that hit the collision thresholds are marked as True in boolean matrix
    stop_robot = False
    print("x Collision front")
    collision_bool_x_front = points[:,0] < collision_x
    for bool_x, point, distance, theta in zip(collision_bool_x_front, points, distances, thetas):
        print(bool_x, point[0], point[1], distance, theta)
    # print("cfx", collision_bool_x_front)
    collision_bool_x_back = points[:,0] > 0
    collision_bool_x = np.all(np.vstack((collision_bool_x_front, collision_bool_x_back)), axis=0)
    # print("x collision: ", np.any(collision_bool_x))
    # print(np.where(collision_bool_x))
    collision_bool_y_left = points[:,1] < collision_y
    collision_bool_y_right = points[:,1] > -collision_y
    collision_bool_y = np.all(np.vstack((collision_bool_y_left, collision_bool_y_right)), axis=0)
    collision_bool = np.all(np.vstack((collision_bool_y, collision_bool_x)), axis=0)
    if np.any(collision_bool):
        stop_robot = True

    # Determine speed based on whether a collision is imminent
    command = Twist()
    if stop_robot:
        command.linear.x = 0.0
    else:
        command.linear.x = 0.05
    command.linear.y = 0.0
    command.linear.z = 0.0
    command.angular.x = 0.0
    command.angular.y = 0.0
    command.angular.z = 0.0

    # Publish the speed
    pub.publish(command)


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('lab1', log_level=rospy.DEBUG)

    # subscribe to lidar laser scan message
    lidar_sub = rospy.Subscriber('/scan', LaserScan, lidar_callback)

    # publish twist message
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Turn control over to ROS
    rospy.spin()
