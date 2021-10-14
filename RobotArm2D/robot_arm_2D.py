#!/usr/bin/env python3

# Get the windowing packages
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize

from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor

from random import random

import numpy as np
from numpy import sin, cos, pi

USE_TRIG = True

# A helper class that implements a slider with given start and end value; displays values
class SliderDisplay(QWidget):
    gui = None

    def __init__(self, name, low, high, initial_value, ticks=500):
        """
        Give me a name, the low and high values, and an initial value to set
        :param name: Name displayed on slider
        :param low: Minimum value slider returns
        :param high: Maximum value slider returns
        :param initial_value: Should be a value between low and high
        :param ticks: Resolution of slider - all sliders are integer/fixed number of ticks
        """
        # Save input values
        self.name = name
        self.low = low
        self.range = high - low
        self.ticks = ticks

        # I'm a widget with a text value next to a slider
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(ticks)
        # call back - calls change_value when slider changed/moved
        self.slider.valueChanged.connect(self.change_value)

        # For displaying the numeric value
        self.display = QLabel()
        self.set_value(initial_value)
        self.change_value()

        layout.addWidget(self.display)
        layout.addWidget(self.slider)

    # Use this to get the value between low/high
    def value(self):
        """Return the current value of the slider"""
        return (self.slider.value() / self.ticks) * self.range + self.low

    # Called when the value changes - resets display text
    def change_value(self):
        if SliderDisplay.gui:
            SliderDisplay.gui.repaint()
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))

    # Use this to change the value (does clamping)
    def set_value(self, value_f):
        """Set the value of the slider
        @param value_f: value between low and high - clamps if not"""
        value_tick = self.ticks * (value_f - self.low) / self.range
        value_tick = min(max(0, value_tick), self.ticks)
        self.slider.setValue(int(value_tick))
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))


# The main class for handling the robot drawing and geometry
class DrawRobot(QWidget):
    def __init__(self, in_gui):
        super().__init__()

        # In order to get to the slider values
        self.gui = in_gui

        # Title of the window
        self.title = "Robot arm"
        # output text displayed in window
        self.text = "Not reaching"

        # Window size
        self.top = 15
        self.left = 15
        self.width = 500
        self.height = 500

        # For doing dictionaries
        self.components = ['upperarm', 'forearm', 'wrist', 'finger1', 'finger2']
        # Set geometry
        self.init_ui()

    def init_ui(self):
        self.text = "Not reaching"
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    # For making sure the window shows up the right size
    def minimumSizeHint(self):
        return QSize(self.width, self.height)

    # For making sure the window shows up the right size
    def sizeHint(self):
        return QSize(self.width, self.height)

    # What to draw - called whenever window needs to be drawn
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_text(event, qp)
        self.draw_target(qp)
        self.draw_arm(qp)
        qp.end()

    # Put some text in the bottom left
    def draw_text(self, event, qp):
        qp.setPen(QColor(168, 34, 3))
        qp.setFont(QFont('Decorative', 10))
        qp.drawText(event.rect(), Qt.AlignBottom, self.text)

    # Map from [0,1]x[0,1] to the width and height of the window
    def x_map(self, x):
        return int(x * self.width)

    # Map from [0,1]x[0,1] to the width and height of the window - need to flip y
    def y_map(self, y):
        return self.height - int(y * self.height) - 1

    # Draw a + where the target is and another where the end effector is
    def draw_target(self, qp):
        pen = QPen(Qt.darkGreen, 2, Qt.SolidLine)
        qp.setPen(pen)
        x_i = self.x_map(self.gui.reach_x.value())
        y_i = self.y_map(self.gui.reach_y.value())
        qp.drawLine(x_i-5, y_i, x_i+5, y_i)
        qp.drawLine(x_i, y_i-5, x_i, y_i+5)

        pt = self.arm_end_pt()
        pen.setColor(Qt.darkRed)
        qp.setPen(pen)

        x_i = self.x_map(pt[0])
        y_i = self.y_map(pt[1])
        qp.drawLine(x_i-5, y_i, x_i+5, y_i)
        qp.drawLine(x_i, y_i-5, x_i, y_i+5)

    # Make a rectangle with the center at the middle of the left hand edge
    # Width is 1/4 length
    # returns four corners with points as row vectors
    @staticmethod
    def make_rect(in_len):
        """Draw a rectangle of the given length; width is 1/4 of length
        @param: in_len desired length
        @return: a 1x4 array of x,y values representing the four corners of the rectangle"""
        x_l = 0
        x_r = in_len
        h = in_len/4
        y_b = -h/2
        y_t = y_b + h
        return [[x_l, y_b, 1], [x_r, y_b, 1], [x_r, y_t, 1], [x_l, y_t, 1]]

    # Apply the matrix m to the points in rect
    @staticmethod
    def transform_rect(rect, m):
        """Apply the 3x3 transformation matrix to the rectangle
        @param: rect: Rectangle from make_rect
        @param: m - 3x3 matrix
        @return: a 1x4 array of x,y values of the transformed rectangle"""
        rect_t = []
        for p in rect:
            p_new = m @ np.transpose(p)
            rect_t.append(np.transpose(p_new))
        return rect_t

    # Create a rotation matrix
    @staticmethod
    def rotation_matrix(theta):
        """Create a 3x3 rotation matrix that rotates in the x,y plane
        @param: theta - amount to rotate by in radians
        @return: 3x3 matrix, 2D rotation plus identity """
        m_rot = np.identity(3)
        m_rot[0][0] = cos(theta)
        m_rot[0][1] = -sin(theta)
        m_rot[1][0] = sin(theta)
        m_rot[1][1] = cos(theta)
        return m_rot

    # Create a translation matrix
    @staticmethod
    def translation_matrix(dx, dy):
        """Create a 3x3 translation matrix that moves by dx, dy
        @param: dx - translate by that much in x
        @param: dy - translate by that much in y
        @return: 3x3 matrix """
        m_trans = np.identity(3)
        m_trans[0, 2] = dx
        m_trans[1, 2] = dy
        return m_trans

    # Draw the given box
    def draw_rect(self, rect, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        for i in range(0, len(rect)):
            i_next = (i+1) % len(rect)
            x_i = self.x_map(rect[i][0])
            y_i = self.y_map(rect[i][1])
            x_i_next = self.x_map(rect[i_next][0])
            y_i_next = self.y_map(rect[i_next][1])
            qp.drawLine(x_i, y_i, x_i_next, y_i_next)

    # Return the matrices that move each of the components. Do this as a dictionary, just to be clean
    def get_matrices(self):
        # The values used to build the matrices
        len_upper_arm = self.gui.length_upper_arm.value()
        len_forearm = self.gui.length_lower_arm.value()
        len_wrist = self.gui.length_lower_arm.value() / 4
        len_finger = self.gui.length_fingers.value()
        h_forearm = len_forearm/4
        ang_shoulder = self.gui.theta_base.value()
        ang_elbow = self.gui.theta_elbow.value()
        ang_wrist = self.gui.theta_wrist.value()
        ang_finger = self.gui.theta_fingers.value()

        mat_ret = dict()

        # begin homework 1 : Problem 2
        # Each of these should be of the form: Translation * rotation
        mat_ret = {comp: np.identity(3) for comp in self.components}

        # Create a rotation matrix for the upper arm using its angle in world frame
        upper_arm_R = self.rotation_matrix(ang_shoulder)
        # Create a translation matrix for the upper arm using its position in world frame
        upper_arm_T = self.translation_matrix(0, 0.5)
        # Combine rotation and translation into transform from world frame to upper-arm frame
        chain_mat = upper_arm_T @ upper_arm_R
        mat_ret['upperarm'] = np.copy(chain_mat)

        # Create a rotation matrix to rotate the forearm around the origin of the upper-arm frame 
        # Translate from upper-arm frame to world frame
        upper_arm_Ti = np.linalg.inv(upper_arm_T)         
        # Rotate in world frame. The angles are in world frame.
        # By subtracting shoulder (upper-arm) angle in world frame from elbow (fore-arm) angle in world frame, we can find elbow angle in upper-arm frame.
        forearm_R = self.rotation_matrix(ang_elbow - ang_shoulder)   
        # Combine matricies such that we translate to the world-frame origin, perform the rotation in world-frame, and translate back to upper-arm frame
        rotate_forearm_around_point = upper_arm_T @ forearm_R @ upper_arm_Ti
        # Create a translation matrix in world-frame coordinates that translates from upper-arm frame to fore-arm frame
        forearm_T = self.translation_matrix( len_upper_arm * np.cos(ang_shoulder), len_upper_arm * np.sin(ang_shoulder) )
        # Combine rotation and translation with world->upper-arm transformation to get from world frame to fore-arm frame
        chain_mat = forearm_T @ rotate_forearm_around_point @ chain_mat
        mat_ret['forearm'] = np.copy(chain_mat)

        # Create a rotation matrix to rotate the wrist around the origin of the fore-arm frame
        wrist_R = self.rotation_matrix(ang_wrist - ang_elbow)
        rotate_wrist_around_forearm = chain_mat @ wrist_R @ np.linalg.inv(chain_mat)
        # Create a translation matrix to translate the wrist from fore-arm frame to wrist frame
        wrist_T = self.translation_matrix(len_forearm * np.cos(ang_elbow), len_forearm * np.sin(ang_elbow))
        chain_mat = wrist_T @ rotate_wrist_around_forearm @ chain_mat
        mat_ret['wrist'] = np.copy(chain_mat)

        # end homework 1 : Problem 2
        return mat_ret

    def draw_arm(self, qp):
        """Draw the arm as boxes
        :param: qp - the painter window
        """
        # begin homework 1: Problem 1
        if USE_TRIG:
            # Set up pen for drawing lines
            pen = QPen(Qt.blue, 4, Qt.SolidLine)
            qp.setPen(pen)
            # Set up origin for calculations
            origin_x = 0
            origin_y = 0.5
            # Calculate base x and y with trig and draw base
            base_x = self.gui.length_upper_arm.value() * np.cos(gui.theta_base.value())
            base_y = self.gui.length_upper_arm.value() * np.sin(gui.theta_base.value())
            qp.drawLine(self.x_map(origin_x), self.y_map(origin_y), self.x_map(origin_x + base_x), self.y_map(origin_y + base_y))
            # Calculate lower arm x and y with trig and draw base
            low_x = self.gui.length_lower_arm.value() * np.cos(gui.theta_elbow.value())
            low_y = self.gui.length_lower_arm.value() * np.sin(gui.theta_elbow.value())
            qp.drawLine(self.x_map(origin_x + base_x), self.y_map(origin_y + base_y), self.x_map(origin_x + base_x + low_x), self.y_map(origin_y + base_y + low_y))
        # end homework 1: Problem 1

        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        # Create a rectangle for each component then move it to the correct place then draw it
        rects = dict()
        rects['upperarm'] = self.make_rect(self.gui.length_upper_arm.value())
        rects['forearm'] = self.make_rect(self.gui.length_lower_arm.value())
        rects['wrist'] = self.make_rect(self.gui.length_lower_arm.value() / 4)
        rects['finger1'] = self.make_rect(self.gui.length_fingers.value())
        rects['finger2'] = self.make_rect(self.gui.length_fingers.value())
        h_wrist = 0.75 * self.gui.length_lower_arm.value()/4

        # begin homework 1 : Problem 2
        # Transform and draw each component using the matrices in self.get_matrices()
        # Example call:
        #   rect_transform = self.transform_rect(rects['base'], mat)
        #   self.draw_rect(rect_transform, qp)
            #   getting the translation matrix for upper arm: matrices['upperarm' + '_T']
        mat_ret = self.get_matrices()
        for comp in self.components:
            rect_transform = self.transform_rect(rects[comp], mat_ret[comp])
            self.draw_rect(rect_transform, qp)
        # end homework 1 : Problem 2

    def arm_end_pt(self):
        """ Return the end point of the arm"""
        # begin homework 1: Problem 1
        if USE_TRIG:
            # end pt x position is sum of upper arm x and lower arm x
            end_x = self.gui.length_upper_arm.value() * np.cos(gui.theta_base.value()) + \
                self.gui.length_lower_arm.value() * np.cos(gui.theta_elbow.value())
            # end pt y position is sum of origin y, upper arm y, and lower arm y
            end_y = 0.5 + self.gui.length_upper_arm.value() * np.sin(gui.theta_base.value()) + \
                self.gui.length_lower_arm.value() * np.sin(gui.theta_elbow.value())
            return np.array([end_x, end_y])
        # end homework 1: Problem 1

        matrices = self.get_matrices()
        mat_accum = np.identity(3)
        # begin homework 1 : Problem 2 (second part)
        # end homework 1 : Problem 2 (second part)
        pt_end = mat_accum[0:2, 2]
        return pt_end


class RobotArmGUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('ROB 514 2D robot arm')

        # Control buttons for the interface
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Different do reach commands
        reach_gradient_button = QPushButton('Reach gradient')
        reach_gradient_button.clicked.connect(self.reach_gradient)

        reach_jacobian_button = QPushButton('Reach Jacobian')
        reach_jacobian_button.clicked.connect(self.reach_jacobian)

        reaches = QGroupBox('Reaches')
        reaches_layout = QVBoxLayout()
        reaches_layout.addWidget(reach_gradient_button)
        reaches_layout.addWidget(reach_jacobian_button)
        reaches.setLayout(reaches_layout)

        # The parameters of the robot arm we're simulating
        parameters = QGroupBox('Arm parameters')
        parameter_layout = QVBoxLayout()
        self.theta_base = SliderDisplay('Angle base', -np.pi/2, np.pi/2, 0)
        self.theta_elbow = SliderDisplay('Angle elbow', -np.pi/2, np.pi/2, 0)
        self.theta_wrist = SliderDisplay('Angle wrist', -np.pi/2, np.pi/2, 0)
        self.theta_fingers = SliderDisplay('Angle fingers', -np.pi/4, 0, -np.pi/8)
        self.length_upper_arm = SliderDisplay('Length upper arm', 0.2, 0.4, 0.3)
        self.length_lower_arm = SliderDisplay('Length lower arm', 0.1, 0.2, 0.15)
        self.length_fingers = SliderDisplay('Length fingers', 0.05, 0.1, 0.075)
        self.theta_slds = []
        self.theta_slds.append(self.theta_base)
        self.theta_slds.append(self.theta_elbow)
        self.theta_slds.append(self.theta_wrist)

        parameter_layout.addWidget(self.theta_base)
        parameter_layout.addWidget(self.theta_elbow)
        parameter_layout.addWidget(self.theta_wrist)
        parameter_layout.addWidget(self.theta_fingers)
        parameter_layout.addWidget(self.length_upper_arm)
        parameter_layout.addWidget(self.length_lower_arm)
        parameter_layout.addWidget(self.length_fingers)

        parameters.setLayout(parameter_layout)

        # The point to reach to
        reach_point = QGroupBox('Reach point')
        reach_point_layout = QVBoxLayout()
        self.reach_x = SliderDisplay('x', 0, 1, 0.5)
        self.reach_y = SliderDisplay('y', 0, 1, 0.5)
        random_button = QPushButton('Random')
        random_button.clicked.connect(self.random_reach)
        reach_point_layout.addWidget(self.reach_x)
        reach_point_layout.addWidget(self.reach_y)
        reach_point_layout.addWidget(random_button)
        reach_point.setLayout(reach_point_layout)

        # The display for the graph
        self.robot_arm = DrawRobot(self)

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)
        left_side_layout = QVBoxLayout()
        right_side_layout = QVBoxLayout()

        left_side_layout.addWidget(reaches)
        left_side_layout.addWidget(reach_point)
        left_side_layout.addStretch()
        left_side_layout.addWidget(parameters)

        right_side_layout.addWidget(self.robot_arm)
        right_side_layout.addWidget(quit_button)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(right_side_layout)

        SliderDisplay.gui = self

    # generate a random reach point
    def random_reach(self):
        self.reach_x.set_value(random())
        self.reach_y.set_value(random())
        self.robot_arm.repaint()

    def reach_gradient(self):
        """Align the robot end point (palm) to the target point using gradient descent"""

        # Use the text field to say what happened
        self.robot_arm.text = "Not improved"

        # begin homework 2 : Problem 1
        # Keep trying smaller increments while nothing improves
            # calculate the current distance
            # Try each angle in turn
                # Gradient
        # end homework 2 : Problem 1
        self.robot_arm.repaint()

    def reach_jacobian(self):
        """ Use the Jacobian to calculate the desired angle change"""

        # An example problem of an arm with radius 3 currently at angle theta
        radius = 3
        theta = 0.2
        # Vector to the end point
        r = [radius * cos(theta), radius * sin(theta), 0]
        # Spin around z
        omega_hat = [0, 0, 1]
        # always 0 in 3rd component
        omega_cross_r = np.cross(omega_hat, r)
        # Desired x,y change
        dx_dy = np.zeros([2, 1])
        dx_dy[0, 0] = -0.01
        dx_dy[1, 0] = -0.1
        # Jacobian
        J = np.zeros([2, 1])
        J[0:2, 0] = np.transpose(omega_cross_r[0:2])
        # Solve
        d_ang = np.linalg.lstsq(J, dx_dy, rcond=None)[0]
        # Check result of solve - should be the same as dx_dy
        res = J @ d_ang
        # The actual point you end up at if you change the angle by that much
        pt_new = [radius * cos(theta + d_ang), radius * sin(theta + d_ang)]

        # begin homework 2 : Problem 2
        # Desired change in x,y
        # Use pseudo inverse to solve
        # to set text
        # self.robot_arm.text = text
        # end homework 2 problem 2
        self.robot_arm.repaint()

    def draw(self, unused_data):
        self.robot_arm.draw()


if __name__ == '__main__':
    app = QApplication([])

    gui = RobotArmGUI()

    gui.show()

    app.exec_()
