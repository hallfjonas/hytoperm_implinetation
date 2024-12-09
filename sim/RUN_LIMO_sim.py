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
from matplotlib.gridspec import GridSpec
import numpy as np
import math
import time
import LIMO_LQR_sim
import LIMO_PID_sim
from nonLinModel import *
from test_sim import World
from hytoperm import Experiment


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

        # Frequency at which commands are sent to limo in Hz (max is 10hz in real robot) (in proportion to scale of world)
        self.transmissionRate = 100
        self.dt = 1/self.transmissionRate
        # self.dt = 1e-5  # delete when move to real robot 1e-5 = 10khz
        # self.rate = rospy.Rate(self.transmissionRate)

        # Define agent state
        self.x = np.array([
            [0.],
            [0.],
            [0.],
            [0.]
        ])

        # Create PID controller object

        self.pid = LIMO_PID_sim.PID(
            steer_kp=0.1, steer_ki=0.0, steer_kd=0, vel_kp=0.1, vel_ki=0.0, vel_kd=0, dt=self.dt)
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
    ) -> list:
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

        values = []
        Values_u_steer = []
        Values_u_vel = []
        Values_x = []
        Values_new_x = []
        Values_t = []
        t = 0

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
            dist = 0
            # Approach the next waypoint for stab_time time steps
            for count in range(stab_time, 1, -1):

                # Find the controls from the PID # u(t) output values, errors, integral error
                u_steer, e_steer_prev, e_steer_int = self.pid.steerControl(
                    self.x, xd, e_steer_prev, e_steer_int)
                u_vel, e_vel_prev, e_vel_int = self.pid.speedControl(
                    self.x, xd, e_vel_prev, e_vel_int)

                Values_u_steer.append(u_steer)
                Values_u_vel.append(u_vel)
                Values_x.append(self.x)
                Values_new_x.append(xd)
                Values_t.append(t)

                # Apply the controls to the nonlinear model
                # self.x = nonlinearModelStep(self.x, np.array([[u_steer], [u_vel]]) , self.dt)
                t = t+self.dt
                # print(t)

                self.x = unicycleModelStep(
                    self.x, np.array([[u_steer], [u_vel]]), self.dt)

                # TODO(Justen):
                # 1) Compute distance to waypoint
                # 2) If distance small, then break. Otherwise continue

                # np is numpy class, linalg is method in numpy (np) class, distance = xd is waypoints value - self.x coordinate value
                dist = np.linalg.norm(xd[0:2, 0] - self.x[0:2, 0])
                if dist < 0.01:
                    break

        # puts each list of all values for each step into one list
        values = [Values_u_steer, Values_u_vel,
                  Values_x, Values_new_x, Values_t]
        return values


def plotLimosim(values) -> None:  # values = u_steer, u_vel, self.x, xd, t

    # gets an array of each timestep values, corresponding each index to each next time step
    u_steer = values[0]
    u_vel = values[1]
    pos = values[2]  # pos = [x,y, heading angle, ang vel.]
    x = []
    y = []
    targetpos = values[3]
    xd = []
    yd = []
    t = values[4]
    count = 0

    for i in pos:
        # count = t.index(i)
        x.append(i[0])
        y.append(i[1])
    for j in targetpos:
        xd.append(j[0])
        yd.append(j[1])

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    timesteps = values[4]  # list of t
    # print(timesteps)
    axs[0, 0].plot(t, u_steer)
    axs[0, 1].plot(t, u_vel)
    axs[1, 0].plot(xd, yd, 'r*')
    axs[1, 0].plot(x, y)

    # axs[1, 0].set_aspect('equal')

    # axs[1, 1].plot(t, targetpos)

    axs[0, 0].set_title(
        'AngularTurning PID_Controller Value [rad/s] vs. Time[s]')
    axs[0, 1].set_title('LinearSpeed PID_Controller Value [m/s] vs. Time[s]')
    axs[1, 0].set_title(
        'Simulation Environment. (Red = waypoints, blue = robot path)')
    axs[1, 1].set_title('Axis [0, 0]')

    for ax in axs.flat:
        title = ax.get_title()
        if 'v' in title:
            index = title.find('v')
            x_label = title[:index]
            y_label = title.split()[-1]

        ax.set(xlabel={x_label}, ylabel={y_label})

    axs[1, 0].set(xlabel='x-axis', ylabel='y-axis')

    plt.tight_layout()
    plt.show()

    plot_x = []
    plot_y = []

    # # if plotting on an existing figure, you don't need to re-plot the trajectory
    # if fig is not None:
    #     fig = plt.figure(1)
    # else:
    #     wl = axs[0, 0]  # wL not w"one"
    #     wl, = plt.plot(trajectory[0, :],
    #                     trajectory[1, :], '-g', marker='*')

    # Plot the agents position
    # if using the original plot from the hytoperm output,scale the plot
    # This is the case if a figure is passed
    # plot_x = []
    # plot_y = []

    # if fig is not None:
    #     plot_x.append((self.x[0, 0]+2.5)/5.)
    #     plot_y.append((self.x[1, 0]+2.5)/5.)
    # else:
    #     plot_x.append(self.x[0, 0])
    #     plot_y.append(self.x[1, 0])

    # tl, = plt.plot(plot_x, plot_y, '-r', marker='*')

    # if count == stab_time and fig is None:
    #     plt.legend([wl, tl], ['waypoints', 'trajectory'])

    # plt.draw()
    # plt.pause(0.1)
    # print(count)

    #  # plot final trajectory
    #         fig, axs = plt.subplots(2, 4)
    #         # axs[0, 0].plot(plot_x, plot_y, '-r')
    #         # axs[0, 0].plot(trajectory[0, :], trajectory[1, :], '-g')
    #         # axs[0, 0].ioff()
    #         # axs[0, 0].gca().set_aspect('equal')

    #         x = np.linspace(0, 2*np.pi, 400)
    #         y = np.sin(x**2)
    #         axs[1, 0].plot(x, y)
    #         axs.show()
    return


# values = u_steer, u_vel, self.x, xd, t
def plotCyclesim(values, world: World = None, ex: Experiment = None) -> None:
    # gets an array of each timestep values, corresponding each index to each next time step
    u_steer = values[0]
    u_vel = values[1]
    pos = values[2]  # pos = [x,y, heading angle, ang vel.]
    x = []
    y = []
    targetpos = values[3]
    xd = []
    yd = []
    t = values[4]
    count = 0

    for i in pos:
        # count = t.index(i)
        x.append(i[0])
        y.append(i[1])
    for j in targetpos:
        xd.append(j[0])
        yd.append(j[1])

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    timesteps = values[4]  # list of t
    # print(timesteps)

    axs[0, 0].plot(t, u_steer)
    axs[1, 0].plot(t, u_vel)

    ex = ex if ex is not None else world.ex if world is not None else None
    ex.plotWorld(ax=axs[0, 1])
    axs[0, 1].plot(xd, yd, 'r*')
    axs[0, 1].plot(x, y)
    # axs[0, 1].set_aspect('equal')

    # axs[1, 1].plot(t, targetpos)

    axs[0, 0].set_title(
        'AngularTurning PID_Controller Value [rad/s] vs. Time[s]')
    axs[1, 0].set_title('LinearSpeed PID_Controller Value [m/s] vs. Time[s]')
    axs[0, 1].set_title(
        'Simulation Environment. (Red = waypoints, blue = robot path)')
    axs[1, 1].axis('off')

    for ax in axs.flat:
        title = ax.get_title()
        if 'v' in title:
            index = title.find('v')
            y_label = title[:index]
            x_label = title.split()[-1]

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    axs[0, 1].set(xlabel='x-axis', ylabel='y-axis')

    plt.tight_layout()
    plt.show()

    plot_x = []
    plot_y = []

    # # if plotting on an existing figure, you don't need to re-plot the trajectory
    # if fig is not None:
    #     fig = plt.figure(1)
    # else:
    #     wl = axs[0, 0]  # wL not w"one"
    #     wl, = plt.plot(trajectory[0, :],
    #                     trajectory[1, :], '-g', marker='*')

    # Plot the agents position
    # if using the original plot from the hytoperm output,scale the plot
    # This is the case if a figure is passed
    # plot_x = []
    # plot_y = []

    # if fig is not None:
    #     plot_x.append((self.x[0, 0]+2.5)/5.)
    #     plot_y.append((self.x[1, 0]+2.5)/5.)
    # else:
    #     plot_x.append(self.x[0, 0])
    #     plot_y.append(self.x[1, 0])

    # tl, = plt.plot(plot_x, plot_y, '-r', marker='*')

    # if count == stab_time and fig is None:
    #     plt.legend([wl, tl], ['waypoints', 'trajectory'])

    # plt.draw()
    # plt.pause(0.1)
    # print(count)

    #  # plot final trajectory
    #         fig, axs = plt.subplots(2, 4)
    #         # axs[0, 0].plot(plot_x, plot_y, '-r')
    #         # axs[0, 0].plot(trajectory[0, :], trajectory[1, :], '-g')
    #         # axs[0, 0].ioff()
    #         # axs[0, 0].gca().set_aspect('equal')

    #         x = np.linspace(0, 2*np.pi, 400)
    #         y = np.sin(x**2)
    #         axs[1, 0].plot(x, y)
    #         axs.show()
    return


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
        waypoints.append(waypoint)  # inputs all the waypoints into a list
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
    # waypoints = genCircleWaypoints(10, 5000, (0, 0))
    waypoints = genRandomWaypoints(5)
    data = tracker.trackTrajectoryPID(waypoints)
    plotLimosim(data)
