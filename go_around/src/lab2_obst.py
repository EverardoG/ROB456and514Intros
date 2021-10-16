#!/usr/bin/env python

import rospy

import numpy as np
from typing import Tuple

# Sensor message types
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# the velocity command message
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

# visualizing markers
from visualization_msgs.msg import MarkerArray, Marker

from pprint import PrettyPrinter
PP = PrettyPrinter()
class RosNode():
    def __init__(self):
        # Initialize the node
        rospy.init_node('lab2_obst', log_level=rospy.DEBUG)

        # subscribe to lidar laser scan message
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # publish twist message
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # publish lidar points
        self.scan_pub = rospy.Publisher('/filtered_scan', MarkerArray, queue_size=10)
        self.collision_pub = rospy.Publisher('/collision_points', MarkerArray, queue_size=10)

        # Set thresholds for collision in meters
        # y collision is set such that robot will detect an emminent collision if there are any points within collision_y
        # on either the left OR right side
        self.robot_width = .306
        self.extra_spacing_y = 0.05
        self.collision_x = 2
        self.collision_y = self.robot_width/2 + self.extra_spacing_y

        # Set variables for control
        self.slowdown_x = 1.3
        self.stop_x = 0.5
        self.max_speed = 0.2
        self.min_speed = 0.05

        self.max_yaw_speed = 0.5

        # Set goal position [x,y]
        self.goal_pos = np.array([0,0])

        # Variables for storing sensor information
        self.odom_msg = None
        self.scan_msg = None
        self.points = None
        self.collision_points = None
        self.position = None
        self.angle = None

    def new_marker(self, id: int, action: int, timestamp: rospy.rostime.Time)->Marker:
        """Make a new marker with a particular action"""
        marker = Marker()
        marker.header.frame_id = "base_scan"
        marker.header.stamp = timestamp
        marker.ns = "points"
        marker.id = id
        marker.action = action
        return marker

    def build_marker(self, point: np.ndarray, id: int, timestamp: rospy.rostime.Time)->Marker:
        """Build a marker from a point"""
        marker = self.new_marker(id, Marker.ADD, timestamp)
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

    def build_marker_array(self, points: np.ndarray, timestamp: rospy.rostime.Time)->MarkerArray:
        """Build a marker array message from a set of points""" 
        marker_array = MarkerArray()
        for row in range(points.shape[0]):
            marker = self.build_marker(points[row], row, timestamp)
            marker_array.markers.append(marker)
        while len(marker_array.markers) < 360:
            delete_marker = self.new_marker(len(marker_array.markers), Marker.DELETE, timestamp)
            marker_array.markers.append(delete_marker)
        return marker_array

    def filter_scan_msg(self, scan_msg: LaserScan)->Tuple[np.ndarray, np.ndarray]:
        """Take a raw LaserScan msg and grab filtered distances and thetas"""
        # Grab distances and thetas
        distances = np.array(scan_msg.ranges)
        thetas = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))

        # Filter out any distances that are at infinite values
        not_inf_bool = np.logical_not(np.isinf(distances))
        filtered_distances = distances[not_inf_bool]
        filtered_thetas = thetas[not_inf_bool]
        
        return filtered_distances, filtered_thetas

    def distances_to_points(self, filtered_distances: np.ndarray, filtered_thetas: np.ndarray)->np.ndarray:
        """Transform distances and thetas into coordinates of points"""
        # Transform distances and thetas into 2D array of points with x and y relative to lidar
        # Positive x is from lidar to front of turtlebot
        # Positive y is from lidar to left of turtlebot
        # Each row is a point. Col 0 is x. Col 1 is y.
        distances_x = filtered_distances * np.cos(filtered_thetas)
        distances_y = filtered_distances * np.sin(filtered_thetas)
        points = np.vstack((distances_x, distances_y)).T
        return points

    def calculate_collision_points(self, points: np.ndarray)->np.ndarray:
        """Calculate points will cause a collision from input points"""
        # Create boolean masks that have True where there is a collision and False 
        # where there is no collision. Use that to grab only points that cause collisions
        collision_x_bool = np.logical_and(points[:,0] < self.collision_x, points[:,0] > 0)
        collision_y_bool = np.logical_and(points[:,1] < self.collision_y, points[:,1] > -self.collision_y)
        collision_bool = np.logical_and(collision_x_bool, collision_y_bool)
        collision_points = points[collision_bool]

        return collision_points

    def get_nearest_x(self, points: np.ndarray)->float:
        """Get the nearest x value out of all input points"""
        if points.shape[0] > 0:
            return np.min(np.abs(points[:,0]))
        else:
            return self.collision_x

    def calculate_speed_obst(self, x: float)->float:
        """Calculate speed based on nearest x value to not hit obstacle"""
        speed = self.max_speed * (x - self.stop_x)/(self.slowdown_x - self.stop_x)
        if speed > self.max_speed:
            return self.max_speed
        elif speed < self.min_speed:
            return self.min_speed
        else:
            return speed
    
    def calculate_speed_goal(self)->float:
        # Use proportional speed control for distance
        delta_pos = self.goal_pos - self.position
        distance = np.linalg.norm(delta_pos)
        linear_speed = self.max_speed * distance
        if linear_speed > self.max_speed:
            linear_speed = self.max_speed
        elif distance < 0.03:
            linear_speed = 0
        return linear_speed
    
    def calculate_yaw_obst(self, com: np.ndarray)->float:
        """Calculate yaw speed based on y coordinate of com to avoid"""
        yaw = - self.max_yaw_speed * (self.collision_y - com[1])/self.collision_y
        if np.abs(yaw) > self.max_yaw_speed:
            yaw = np.sign(yaw) * self.max_yaw_speed
        elif np.abs(yaw) <= 0.01:
            yaw = 0
        return yaw
    
    def calculate_yaw_goal(self)->float:
        """Calculate angular speed based on lining up robot orientation towards the goal position"""
        delta_pos = self.goal_pos - self.position
        target_yaw = np.arctan2(delta_pos[1], delta_pos[0])
        # delta yaw is how the robot should change its yaw
        delta_yaw = self.angle - target_yaw
        # Handle edge cases
        if delta_yaw < -np.pi:
            delta_yaw += 2*np.pi
        elif delta_yaw > np.pi:
            delta_yaw -= 2*np.pi

        # Use proportional speed control based on minimizing delta yaw
        angular_speed = - self.max_yaw_speed * delta_yaw/(np.pi/2)
        if np.abs(angular_speed) > self.max_yaw_speed:
            angular_speed = np.sign(angular_speed) * self.max_yaw_speed
        elif np.abs(delta_yaw) <= 2 * np.pi/180:
            angular_speed = 0
        return angular_speed

    def build_twist_msg(self, speed: float, yaw: float)->Twist:
        """Build a twist msg with the input speed as linear speed in x and angular speed"""
        twist_msg = Twist()
        twist_msg.linear.x = speed
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = yaw

        return twist_msg

    def lidar_callback(self, scan_msg)->None:
        """Save points from the lidar in internal variable"""
        # Save the message
        self.scan_msg = scan_msg
        # Get points from lidar
        filtered_distances, filtered_thetas = self.filter_scan_msg(scan_msg)
        points = self.distances_to_points(filtered_distances, filtered_thetas)

        # Publish the filtered scan data
        marker_array = self.build_marker_array(points, timestamp=scan_msg.header.stamp)
        self.scan_pub.publish(marker_array)

        # Save the points
        self.points = points

        # Get points that indicate an eminent collision
        self.collision_points = self.calculate_collision_points(self.points)

        return None

    def odom_callback(self, odom_msg)->None:
        """Save position from odometry in internal variables"""
        # Save the message 
        self.odom_msg = odom_msg
        # Get the position and orientation
        pos = odom_msg.pose.pose.position
        self.position = np.array([pos.x, pos.y])
        ori = odom_msg.pose.pose.orientation
        _, _, self.angle = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        return None

    def run(self)->None:
        """Run main loop of this node until ros is shutdown"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.scan_msg is not None and self.odom_msg is not None:
                self.main_loop()
            r.sleep()
        return None

    def main_loop(self)->None:
        """Determines what the robot should do based on sensor readings
        and publishes Twist message with velocity command"""
        # Publish points causing collision
        c_marker_array = self.build_marker_array(self.collision_points, timestamp=self.scan_msg.header.stamp)
        self.collision_pub.publish(c_marker_array)

        # Avoid obstacles if collision is imminent
        if np.any(self.collision_points[:,0] < self.slowdown_x):
            # Calculate center of mass of collision points
            com = np.average(self.collision_points, axis=0)
            # Move in the opposite y direction
            # ie: Move left if com is on the right and vice versa
            yaw = self.calculate_yaw_obst(com)
            # Use the nearest x position of collision points to determine linear speed
            x = self.get_nearest_x(self.collision_points)
            speed = self.calculate_speed_obst(x)
        # Head towards the goal if no collision is imminent
        else:
            yaw = self.calculate_yaw_goal()
            # Don't move linear speed if far from alignment
            if np.abs(yaw) > (np.pi/30):
                speed = 0
            # Update linear speed towards goal if aligned
            else:
                speed = self.calculate_speed_goal()
      
        # Publish speed
        twist_msg = self.build_twist_msg(speed, yaw)
        self.twist_pub.publish(twist_msg)

if __name__ == "__main__":
    # Create my node
    rosNode = RosNode()
    # Run the node until ros is shutdown
    rosNode.run()
