#!/usr/bin/env python

import rospy

import numpy as np
from typing import Tuple, Optional

# Sensor message types
from sensor_msgs.msg import LaserScan

# the velocity command message
from geometry_msgs.msg import Twist

# visualizing markers
from visualization_msgs.msg import MarkerArray, Marker

from pprint import PrettyPrinter
PP = PrettyPrinter()

def new_marker(id: int, action: int, timestamp: rospy.rostime.Time)->Marker:
    """Make a new marker with a particular action"""
    marker = Marker()
    marker.header.frame_id = "base_scan"
    marker.header.stamp = timestamp
    marker.ns = "points"
    marker.id = id
    marker.action = action
    return marker

def build_marker(point: np.ndarray, id: int, timestamp: rospy.rostime.Time)->Marker:
    """Build a marker from a point"""
    marker = new_marker(id, Marker.ADD, timestamp)
    marker.type = Marker.SPHERE
    marker.pose.position.x = point[0]
    marker.pose.position.y = point[1]
    marker.pose.position.z = 0.5
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 0.9 # Don't forget to set the alpha!
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    return marker

def build_marker_array(points: np.ndarray, timestamp: rospy.rostime.Time)->MarkerArray:
    """Build a marker array message from a set of points""" 
    marker_array = MarkerArray()
    for row in range(points.shape[0]):
        marker = build_marker(points[row], row, timestamp)
        marker_array.markers.append(marker)
    while len(marker_array.markers) < 360:
        delete_marker = new_marker(len(marker_array.markers), Marker.DELETE, timestamp)
        marker_array.markers.append(delete_marker)
    return marker_array

def filter_scan_msg(scan_msg: LaserScan)->Tuple[np.ndarray, np.ndarray]:
    """Take a raw LaserScan msg and grab filtered distances and thetas"""
    # Grab distances and thetas
    distances = np.array(scan_msg.ranges)
    thetas = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))

    # Filter out any distances that are at infinite values
    not_inf_bool = np.logical_not(np.isinf(distances))
    filtered_distances = distances[not_inf_bool]
    filtered_thetas = thetas[not_inf_bool]
    
    return filtered_distances, filtered_thetas

def distances_to_points(filtered_distances: np.ndarray, filtered_thetas: np.ndarray)->np.ndarray:
    """Transform distances and thetas into coordinates of points"""
    # Transform distances and thetas into 2D array of points with x and y relative to lidar
    # Positive x is from lidar to front of turtlebot
    # Positive y is from lidar to left of turtlebot
    # Each row is a point. Col 0 is x. Col 1 is y.
    distances_x = filtered_distances * np.cos(filtered_thetas)
    distances_y = filtered_distances * np.sin(filtered_thetas)
    points = np.vstack((distances_x, distances_y)).T
    return points

def calculate_collision_points(points: np.ndarray)->np.ndarray:
    """Calculate points will cause a collision from input points"""
    # Set thresholds for collision in meters
    # y collision is set such that robot will detect an emminent collision if there are any points within collision_y
    # on either the left OR right side
    robot_width = .306
    extra_spacing_x = 0.05
    collision_x = 2
    collision_y = robot_width/2 + extra_spacing_x

    # Create boolean masks that have True where there is a collision and False 
    # where there is no collision. Use that to grab only points that cause collisions
    collision_x_bool = np.logical_and(points[:,0] < collision_x, points[:,0] > 0)
    collision_y_bool = np.logical_and(points[:,1] < collision_y, points[:,1] > -collision_y)
    collision_bool = np.logical_and(collision_x_bool, collision_y_bool)
    collision_points = points[collision_bool]

    return collision_points

def get_nearest_x(points: np.ndarray)->float:
    """Get the nearest x value out of all input points"""
    if points.shape[0] > 0:
        return np.min(np.abs(points[:,0]))
    else:
        return 2

def calculate_speed(x: float)->float:
    """Calculate speed based on nearest x value"""
    slowdown_x = 1.3
    stop_x = 1
    max_speed = 0.2
    speed = max_speed * (x - stop_x)/(slowdown_x - stop_x)
    if speed > max_speed:
        return max_speed
    elif speed < 0:
        return 0
    else:
        return speed

def build_twist_msg(speed: float)->Twist:
    """Build a twist msg with the input speed as linear speed in x"""
    twist_msg = Twist()
    twist_msg.linear.x = speed
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = 0.0

    return twist_msg

def lidar_callback(scan_msg)->None:
    # Get points from lidar
    filtered_distances, filtered_thetas = filter_scan_msg(scan_msg)
    points = distances_to_points(filtered_distances, filtered_thetas)

    # Publish the filtered scan data
    marker_array = build_marker_array(points, timestamp=scan_msg.header.stamp)
    scan_pub.publish(marker_array)

    # Get points that indicate an eminent collision
    collision_points = calculate_collision_points(points)
    
    # Publish points causing collision
    c_marker_array = build_marker_array(collision_points, timestamp=scan_msg.header.stamp)
    collision_pub.publish(c_marker_array)

    # Calculate the speed
    x = get_nearest_x(collision_points)
    speed = calculate_speed(x)

    # Publish speed
    twist_msg = build_twist_msg(speed)
    twist_pub.publish(twist_msg)
    
    return None

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('lab1', log_level=rospy.DEBUG)

    # subscribe to lidar laser scan message
    lidar_sub = rospy.Subscriber('/scan', LaserScan, lidar_callback)

    # publish twist message
    twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # publish lidar points
    scan_pub = rospy.Publisher('/filtered_scan', MarkerArray, queue_size=10)
    collision_pub = rospy.Publisher('/collision_points', MarkerArray, queue_size=10)

    # Turn control over to ROS
    rospy.spin()
