#!/usr/bin/env python3

"""
This file is used to control a simulation of the AgileX Limos in Rastic to follow a series of Waypoints.
It uses the dynamic model found in nonlinModel.py to find the simulate the motion of the agent.
To use this, create a tracker object and pass the name of any limo,
then pass a numpy array containing the sequence of waypoints to the trackTrajectory function in the following format:
waypoints = np.array([[x1,x2],[y1,y2],[theta1,theta2],[v1,v2]])
for waypoints waypoint1 = np.array([[x1],[y1],[theta1],[v1]]), waypoint2 = np.array([[x2],[y2],[theta2],[v2]]),
For best results, try to ensure that trajectories were generated based on the Limo's dynamic model.
This code was adapted by Carter Berlind with significant changes.
Original code received from Sienna Chien, original authors not listed.

Description: Waypoint tracking for limos in Rastic
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import LIMO_LQR_sim
import LIMO_PID_sim
from nonLinModel import *


class Tracker:
    def __init__(
            self,
            limo_name: str
    ) -> None:
        """
        Tracker object used to follow a sequence of waypoints in a trajectory.
        Using this sends all the infrastructure necessary to receive pose
        information form the motion capture and send controls to the robot to track a trajectory

        :param limo_name: ROS node and topic name for the Limo you are using.
            This should match the name on the motion capture software.
            -> str
        """

        # Frequency at which commands are sent to limo in Hz
        self.transmissionRate = 10
        self.dt = 1/self.transmissionRate
        # self.rate = rospy.Rate(self.transmissionRate)

        # Define agent state
        self.x = np.array([
            [0.],
            [0.],
            [0.],
            [0.]
        ])

        # Create PID controller object

        self.pid = LIMO_PID_sim.PID(dt=self.dt)
        self.theta_dot = 0

    def trackTrajectoryLQR(
            self,
            trajectory: np.ndarray,
            stab_time: int = 27,
            relin_steps: int = 1
    ) -> None:
        """
        This function receives a trajectory of states and, using the motion capture
        to localize, sends control commands to the limo to track the trajectory.
        Controls are found using a linear quadratic regulator.

        :param trajectory: array of N waypoints that you want the robot to follow
            -> 4xN NumPy array
        :param stab_time: approximate number of time steps it takes to rach each point
            -> int
        :param relin_steps: number of time steps between each relinearization
            -> int
        """

        # this stores the actual point trajectory of the robot for plotting/visualization purposes
        plot_x = []
        plot_y = []

        # visualize what you want to track
        plt.plot(trajectory[0], trajectory[1], '-g')

        # iterate through sequence of waypoints
        for i in range(0, trajectory.shape[1], 4):
            # isolate current waypoint
            xd = trajectory[:, i]

            # Approach the next waypoint for stab_time time steps
            for count in range(stab_time, 1, -1):
                # Relinearize and find new gain matrix if relin_steps time steps have passed
                if count % relin_steps == 0:
                    [self.A, self.B] = self.lqr.getAB(self.x[2, 0])
                    [self.A2, self.B2] = self.lqr.getAB(self.x_simp[2, 0])
                    K = self.lqr.getK(count, self.A, self.B)
                    K2 = self.lqr.getK(count, self.A2, self.B2)

                # Find the optimal control
                # The control is the change in angular and linear velocities repectively
                u1 = self.lqr.getControl(self.x, xd, K)

                # Find change in steering steering angle based on desire
                ang = math.atan2((u1[0, 0])*self.lqr.L/2, u1[1, 0])

                # Update speed and steering angle
                self.speed = u1[1, 0]

                # Ensure that inputs are within acceptable range
                # This is an added redundancy to ensure the viability of control inputs
                self.steering_angle = 0.7*np.clip(ang, -1., 1.)
                self.speed = np.clip(self.speed, -1., 1.)

                # Use the nonlinear model to find where the robot will go based on these controls
                self.x = nonlinearModelStep(self.x, np.array(
                    [[self.steering_angle], [self.speed]]), self.dt)

                # Update the plot of the the agent's movement
                plot_x.append(self.x[0, 0])
                plot_y.append(self.x[1, 0])
                plt.plot(plot_x, plot_y, '-r')
                plt.draw()
                plt.pause(0.1)
        # Display final trajectory
        plt.ioff()
        plt.plot(plot_x, plot_y, '-r')
        plt.plot(trajectory[0], trajectory[1], '-g')
        plt.show()

    def trackTrajectoryPID(
            self,
            trajectory: np.ndarray,
            ex=None,
            stab_time: int = 1000,  # varies
            relin_steps: int = 1,
            fig=None,
            ax=None,

    ) -> None:
        """
        This function receives a trajectory of states and, using the motion capture
        to localize, sends control commands to the limo to track the trajectory.
        Controls are found using a PID.

        :param trajectory: array of N waypoints that you want the robot to follow
            -> 4xN NumPy array
        :param stab_time: approximate number of time steps it takes to rach each point
            -> int
        :param relin_steps: number of time steps between each relinearization
            -> int
        """
        plot_x = []
        plot_y = []

        # if plotting on an existing figure, you don't need to re-plot the trajectory
        if fig is not None:
            fig = plt.figure(1)
        else:
            wl, = plt.plot(trajectory[0, :],
                           trajectory[1, :], '-g', marker='*')

        # iterate through sequence of waypoints
        for i in range(trajectory.shape[1]):
            # isolate current waypoint
            xd = trajectory[:, i:i+1]
            # find the initial error values for the PID
            # explanations of the error found in the LIMO_PID_sim.py file
            unit_theta = np.array([
                [math.cos(self.x[2, 0])],
                [math.sin(self.x[2, 0])]
            ])
            ref = xd[:2]-self.x[:2]
            e_steer_prev = np.cross(unit_theta.T, ref.T)[0]
            e_steer_int = 0
            e_vel_prev = np.dot(unit_theta.T, ref)[0, 0]
            e_vel_int = 0

            # Approach the next waypoint for stab_time time steps
            for count in range(stab_time, 1, -1):

                # Find find the controls from the PID
                u_steer, e_steer_prev, e_steer_int = self.pid.steerControl(
                    self.x, xd, e_steer_prev, e_steer_int)
                u_vel, e_vel_prev, e_vel_int = self.pid.speedControl(
                    self.x, xd, e_vel_prev, e_vel_int)

                # Apply the controls to the nonlinear model
                # self.x = nonlinearModelStep(
                # self.x, np.array([[u_steer], [u_vel]]), self.dt)
                self.x = unicycleModelStep(
                    self.x, np.array([[u_steer], [u_vel]]), self.dt)

                # Plot the agents position
                # if using the original plot from the hytoperm output,scale the plot
                # This is the case if a figure is passed
                if fig is not None:
                    plot_x.append((self.x[0, 0]+2.5)/5.)
                    plot_y.append((self.x[1, 0]+2.5)/5.)
                else:
                    plot_x.append(self.x[0, 0])
                    plot_y.append(self.x[1, 0])
                tl, = plt.plot(plot_x, plot_y, '-r', marker='*')

                if count == stab_time:
                    plt.legend([wl, tl], ['waypoints', 'trajectory'])

                plt.draw()
                plt.pause(0.1)
                print(count)

                # TODO(Justen):
                # 1) Compute distance to waypoint
                # 2) If distance small, then break. Otherwise continue

                # np is numpy class, linalg is method in numpy (np) class, xd is waypoints value - self.x coordinate value = distance
                dist = np.linalg.norm(xd[0:2, 0] - self.x[0:2, 0])
                if dist < 0.5:
                    break

        # plot final trajectory
        plt.plot(plot_x, plot_y, '-r')
        plt.plot(trajectory[0, :], trajectory[1, :], '-g')
        plt.ioff()
        plt.show()


def genCircleWaypoints(radius, num_waypoints, center: tuple):
    # Access first/second value in the "center" tuple variable
    center_x, center_y = center[0], center[1]

    # Generate angles for waypoints evenly spaced around the circle
    # array of all angles values around circle
    angles = np.linspace(0, 2 * np.pi, num_waypoints, endpoint=False)

    # Calculate x and y coordinates of waypoints
    # array of x coordinate values
    x_points = center_x + radius * np.cos(angles)
    # array of y coordinate values
    y_points = center_y + radius * np.sin(angles)

    waypoints = []  # list (can contain any object types)
    for x, y in zip(x_points, y_points):
        waypoint = np.array([
            [x],              # x position of second waypoint
            [y],              # y position of second waypoint
            [0],          # heading angle of second waypoint(?)
            [0]                 # angular velocity of second waypoint(?)
        ])
        waypoints.append(waypoint)  # inputs all the waypoints into a list
    return np.hstack(waypoints)

# ***TASK*** function to generate random waypoints, manually input #of waypoints


def genRandomWaypoints(num_Waypoints):
    x_points = np.random.uniform(0, 10, size=num_Waypoints)
    y_points = np.random.uniform(0, 10, size=num_Waypoints)

    waypoints = []  # list (can contain any object types)
    for x, y in zip(x_points, y_points):
        waypoint = np.array([
            [x],              # x position of second waypoint
            [y],              # y position of second waypoint
            [0],          # heading angle of second waypoint(?)
            [0]                 # angular velocity of second waypoint(?)
        ])
        waypoints.append(waypoint)  # inputs all the waypoints into a list\
    return np.hstack(waypoints)


if __name__ == '__main__':
    """
    Example of how to use trajectory tracking
    If this file is run on its own, this is the code that will run
    """

    # Create a Tracker object

    tracker = Tracker("limo770")

    # Define waypoints, ideally your planning algorithm will output waypoints

    # waypoint1 = np.array([
    #     [.5],               # x position of first waypoint
    #     [1],                # y position of first waypoint
    #     [np.pi/4],          # heading angle of first waypoint(?)
    #     [0]                 # angular velocity of first waypoint(?)
    # ])
    # waypoint2 = np.array([
    #     [1.0],              # x position of second waypoint
    #     [2.0],              # y position of second waypoint
    #     [np.pi/2],          # heading angle of second waypoint(?)
    #     [0]                 # angular velocity of second waypoint(?)
    # ])

    # Call the tracker to follow waypoints
    # waypoints = genCircleWaypoints(10, 25, (0, 0))
    waypoints = genRandomWaypoints(10)
    tracker.trackTrajectoryPID(waypoints)
