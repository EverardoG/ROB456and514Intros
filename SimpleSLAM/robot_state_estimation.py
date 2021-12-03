#!/usr/bin/env python3

import numpy as np

from world_state import WorldState
from door_sensor import DoorSensor


# Belief about world/robot state
class RobotStateEstimation:
    def __init__(self):

        # Probability representation (discrete)
        self.probabilities = []
        self.reset_probabilities(10)

        # Kalman (Gaussian) probabilities
        self.mean = 0.5
        self.standard_deviation = 0.4
        self.reset_kalman()

    def reset_probabilities(self, n_probability):
        """ Initialize discrete probability resolution with uniform distribution """
        div = 1.0 / n_probability
        self.probabilities = np.ones(n_probability) * div

    def update_belief_sensor_reading(self, ws, ds, sensor_reading_has_door):
        """ Update your probabilities based on the sensor reading being true (door) or false (no door)
        :param ws World state - has where the doors are
        :param ds Door Sensor - has probabilities for door sensor readings
        :param sensor_reading_has_door - contains true/false from door sensor
        """
        # begin homework 2 : problem 3

        # Create one-hot arrays for door locations
        num_bins = len(self.probabilities)
        door_ind = (np.array(ws.doors)*num_bins - 0.5).astype("int")
        doors = np.zeros(num_bins)
        doors[door_ind] = 1
        not_doors = np.ones(num_bins)
        not_doors[door_ind] = 0

        # Calculate basic probabilities for either case
        prob_here = self.probabilities
        prob_not_here = 1 - prob_here
        num_doors = len(ws.doors)
        num_not_doors = num_bins - num_doors
        prob_door_if_not_here = (num_doors - doors)/(num_bins - 1)
        prob_no_door_if_not_here = (num_not_doors - not_doors)/(num_bins - 1)

        # Calculate probabilities of where we are based on whether or not sensor detected a door
        if sensor_reading_has_door:
            prob_see_door_if_here = ds.prob_see_door_if_door * doors + ds.prob_see_door_if_no_door * not_doors
            prob_see_door_if_not_here = ds.prob_see_door_if_door * prob_door_if_not_here + ds.prob_see_door_if_no_door * prob_no_door_if_not_here

            probabilities = (prob_see_door_if_here * prob_here) / (prob_see_door_if_here * prob_here + prob_see_door_if_not_here * prob_not_here)
        else: # sensor reading has no door
        # new_probs = np.zeros(len(self.probabilities))
            prob_dont_see_door_if_door = 1 - ds.prob_see_door_if_door
            prob_dont_see_door_if_no_door = 1 - ds.prob_see_door_if_no_door
            prob_dont_see_door_if_here = prob_dont_see_door_if_door * doors + prob_dont_see_door_if_no_door * not_doors
            prob_dont_see_door_if_not_here = prob_dont_see_door_if_door * prob_door_if_not_here + prob_dont_see_door_if_no_door * prob_no_door_if_not_here

            probabilities = (prob_dont_see_door_if_here * prob_here) / (prob_dont_see_door_if_here * prob_here + prob_dont_see_door_if_not_here * prob_not_here)
     
        # Normalize - all the denominators are the same because they're the sum of all cases
        self.probabilities = probabilities / np.sum(probabilities)
        # end homework 2 : problem 3

    # Distance to wall sensor (state estimation)
    def update_dist_sensor(self, ws, dist_reading):
        """ Update state estimation based on sensor reading
        :param ws - for standard deviation of wall sensor
        :param dist_reading - distance reading returned from the sensor, in range 0,1 (essentially, robot location) """

        # Standard deviation of error
        standard_deviation = ws.wall_standard_deviation
        # begin homework 2 : Extra credit
        # Calculate a discretized gaussian distribution representing the probability
        # we get a particular distance reading given we are a particular distance from the wall
        x = np.linspace(0,1,len(self.probabilities))
        distribution = np.exp(-(x-dist_reading)**2 / (2*standard_deviation**2) )
        norm_distribution = distribution/np.sum(distribution)
        # Multiply prior belief about where robot is by the likelihood given the new information
        probabilities = self.probabilities * norm_distribution
        # Normalize - all the denominators are the same
        self.probabilities = probabilities / np.sum(probabilities)
        # end homework 2 : Extra credit
        return self.mean, self.standard_deviation

    def update_belief_move_left(self, rs):
        """ Update the probabilities assuming a move left.
        :param rs - robot state, has the probabilities"""

        # begin homework 2 problem 4
        # Check probability of left, no, right sum to one
        # Left edge - put move left probability into zero square along with stay-put probability
        # Right edge - put move right probability into last square

        # Calculate what the probabilities would be for each possible action the robot could have taken
        probabilities_if_actual_move_left = np.zeros(len(self.probabilities))
        probabilities_if_actual_move_left[0] = np.copy(self.probabilities[0])
        probabilities_if_actual_move_left[:-1] += self.probabilities[1:]

        probabilities_if_actual_move_right = np.zeros(len(self.probabilities))
        probabilities_if_actual_move_right[-1] = np.copy(self.probabilities[-1])
        probabilities_if_actual_move_right[1:] += self.probabilities[:-1]

        probabilities_if_actual_move_stay = np.copy(self.probabilities)

        # Calculated a weighted sum of what the robot COULD have done
        # weighted by the probability it did that given the command we gave it
        probabilities = probabilities_if_actual_move_left * rs.prob_move_left_if_left + \
                        probabilities_if_actual_move_right * rs.prob_move_right_if_left + \
                        probabilities_if_actual_move_stay * rs.prob_no_move_if_left

        # Normalize - sum should be one, except for numerical rounding
        self.probabilities = probabilities / np.sum(probabilities)
        # end homework 2 problem 4

    def update_belief_move_right(self, rs):
        """ Update the probabilities assuming a move right.
        :param rs - robot state, has the probabilities"""

        # begin homework 2 problem 4
        # Check probability of left, no, right sum to one
        # Left edge - put move left probability into zero square along with stay-put probability
        # Right edge - put move right probability into last square
        
        # Calculate what the probabilities would be for each possible action the robot could have taken
        probabilities_if_actual_move_left = np.zeros(len(self.probabilities))
        probabilities_if_actual_move_left[0] = np.copy(self.probabilities[0])
        probabilities_if_actual_move_left[:-1] += self.probabilities[1:]

        probabilities_if_actual_move_right = np.zeros(len(self.probabilities))
        probabilities_if_actual_move_right[-1] = np.copy(self.probabilities[-1])
        probabilities_if_actual_move_right[1:] += self.probabilities[:-1]

        probabilities_if_actual_move_stay = np.copy(self.probabilities)

        # Calculated a weighted sum of what the robot COULD have done
        # weighted by the probability it did that given the command we gave it
        probabilities = probabilities_if_actual_move_left * rs.prob_move_left_if_right + \
                        probabilities_if_actual_move_right * rs.prob_move_right_if_right + \
                        probabilities_if_actual_move_stay * rs.prob_no_move_if_right

        # Normalize - sum should be one, except for numerical rounding
        self.probabilities = probabilities / np.sum(probabilities)
        # end homework 2 problem 4

    # Put robot in the middle with a really broad standard deviation
    def reset_kalman(self):
        self.mean = 0.5
        self.standard_deviation = 0.4

    # Given a movement, update Gaussian
    def update_kalman_move(self, rs, amount):
        """ Kalman filter update mean/standard deviation with move (the prediction step)
        :param rs : robot state - has the standard deviation error for moving
        :param amount : The requested amount to move
        :return : mean and standard deviation of my current estimated location """

        # begin homework 3 : Problem 2
        self.mean = self.mean + amount
        self.standard_deviation = np.sqrt( self.standard_deviation**2 + rs.robot_move_standard_deviation_err**2 )
        # end homework 3 : Problem 2
        return self.mean, self.standard_deviation

    # Sensor reading, distance to wall (Kalman filtering)
    def update_gauss_sensor_reading(self, ws, dist_reading):
        """ Update state estimation based on sensor reading
        :param ws - for standard deviation of wall sensor
        :param dist_reading - distance reading returned"""

        # begin homework 3 : Problem 1
        self.mean = self.mean * ws.wall_standard_deviation**2/(self.standard_deviation**2 + ws.wall_standard_deviation**2) + \
            dist_reading * self.standard_deviation**2/(self.standard_deviation**2 + ws.wall_standard_deviation**2)
        self.standard_deviation = np.sqrt( 1/(self.standard_deviation**(-2) + ws.wall_standard_deviation**(-2)) )
        # end homework 3 : Problem 1
        return self.mean, self.standard_deviation


if __name__ == '__main__':
    ws_global = WorldState()

    ds_global = DoorSensor()

    rse_global = RobotStateEstimation()

    # Check out these cases
    # We have two possibilities - either in front of door, or not - cross two sensor readings
    #   saw door versus not saw door
    uniform_prob = rse_global.probabilities[0]

    # begin homework 2 problem 4
    # Four cases - based on default door probabilities of
    # DoorSensor.prob_see_door_if_door = 0.8
    # DoorSensor.prob_see_door_if_no_door = 0.2
    #  and 10 probability divisions. Three doors visible.
    # probability saw door if door, saw door if no door, etc
    # Resulting probabilities, assuming 3 doors
    # Check that our probabilities are updated correctly
    # Spacing of bins
    # end homework 2 problem 4

    print("Passed tests")
