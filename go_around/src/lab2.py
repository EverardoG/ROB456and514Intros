#!/usr/bin/env python

import rospy
import numpy as np

# Sensor message types
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# the velocity command message
from geometry_msgs.msg import Twist

GOAL = (-2, 3)
ODOM = None

def lidar_callback(scan_msg):
    global GOAL, ODOM
    # Let's make a new twist message
    command = Twist()

    # Fill in the fields.  Field values are unspecified 
    # until they are actually assigned. The Twist message 
    # holds linear and angular velocities.
    command.linear.x = 0.0
    command.linear.y = 0.0
    command.linear.z = 0.0
    command.angular.x = 0.0
    command.angular.y = 0.0
    command.angular.z = 0.0

    # Lidar properties (unpacked for your ease of use)
    # find current laser angle, max scan length, distance array for all scans, and number of laser scans
    maxAngle = scan_msg.angle_max
    minAngle = scan_msg.angle_min
    angleIncrement = scan_msg.angle_increment

    maxScanLength = scan_msg.range_max
    distances = scan_msg.ranges
    numScans = len(distances)

    # print(ODOM)
    # Problem 1: move the robot toward the goal
    goal_reached = False
    # Calculate delta position and yaw to move towards
    if ODOM is None:
        return None
    x_pos, y_pos, yaw = ODOM
    curr_pos = np.array([x_pos, y_pos])
    goal_pos = np.array([GOAL[0], GOAL[1]])
    delta_pos = goal_pos - curr_pos
    target_yaw = np.arctan2(delta_pos[1], delta_pos[0])
    # delta yaw is how the robot should change its yaw
    delta_yaw = yaw - target_yaw
    # Handle edge cases
    if delta_yaw < -np.pi:
        delta_yaw += 2*np.pi
    elif delta_yaw > np.pi:
        delta_yaw -= 2*np.pi

    # Use proportional speed control for yaw
    max_angular_speed = 0.5
    angular_speed = - max_angular_speed * delta_yaw/(np.pi/2)
    if np.abs(angular_speed) > max_angular_speed:
        angular_speed = np.sign(angular_speed) * max_angular_speed
    elif np.abs(delta_yaw) <= 2 * np.pi/180:
        angular_speed = 0

    # Use proportional speed control for distance
    distance = np.linalg.norm(delta_pos)
    max_linear_speed = 0.5
    linear_speed = max_linear_speed * distance
    if linear_speed > max_linear_speed:
        linear_speed = max_linear_speed
    elif distance < 0.01:
        linear_speed = 0
        goal_reached = True
    if abs(delta_yaw) > 0.1:
        linear_speed = 0
    # Only update command if goal has not been reached yet
    if not goal_reached:
        command.angular.z = angular_speed
        command.linear.x = linear_speed
    # End problem 1

    currentLaserTheta = minAngle
    # for each laser scan
    for i, scan in enumerate(distances):
        # for each laser scan, the angle is currentLaserTheta, the index is i, and the distance is scan
        # Problem 2: avoid obstacles based on laser scan readings
        # TODO YOUR CODE HERE
        # End problem 2
        # After this loop is done, we increment the currentLaserTheta
        currentLaserTheta = currentLaserTheta + angleIncrement

    pub.publish(command)


def odom_callback(msg):
    """
    Subscribes to the odom message, unpacks and transforms the relevent information, and places it in the global variable ODOM
    ODOM is structured as follows:
    ODOM = (x, y, yaw)

    :param: msg: Odometry message
    :returns: None
    """
    global ODOM
    position = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    (r, p, yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
    ODOM = (position.x, position.y, yaw)


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('lab2', log_level=rospy.DEBUG)

    # subscribe to sensor messages
    lidar_sub = rospy.Subscriber('/scan', LaserScan, lidar_callback)
    odom_sub = rospy.Subscriber('/odom', Odometry, odom_callback)

    # publish twist message
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Turn control over to ROS
    rospy.spin()
