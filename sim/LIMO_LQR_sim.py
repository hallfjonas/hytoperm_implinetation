#!/usr/bin/env python3
"""
Rastic Limo LQR code
LQR code adapted by Carter Berlind to fit Bicyle dynamical model (for AgileX Limo)
For more info on dynimic model, see: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357
Adapted from LQR code by original: Addison Sears-Collins https://automaticaddison.com
For original code (note that there are some significant differences) see: https://automaticaddison.com/linear-quadratic-regulator-lqr-with-python-code-example/
For more info on LQR https://stanford.edu/class/ee363/lectures/dlqr.pdf

Description: Linear Quadratic Regulator for AgileX Limo
"""
import numpy as np

class LQR:
    def __init__(
            self,
            L:float = 0.2,
            dt:float = 0.1
            ) -> None:
        """
        This class is an object in memory that contains a linear quadratic regulator
        to act as a tracking controller for the AgileX Limo robot
        Spec sheet for the Limo can be found here: https://github.com/agilexrobotics/limo-doc/blob/master/Limo%20user%20manual(EN).md

        :param L: wheel base in meters, default to 0.2m from
            LIMO spec sheet -> float
        :param dt: Time step size in seconds, this should be the rate at which
            ROS publishes -> float
        """

        self.L = L
        self.dt = dt

        # input cost matrix (you can adjust this)
        self.R = np.array([
            [10,0], #angular velocity input cost
            [0,300] #linear velocity input cost
        ])

        # state cost matrix (you can adjust this)
        self.Q = np.array([
            [10,0,0,0], #x position error cost
            [0,10,0,0], #y position error cost
            [0,0,10,0], #orientation error cost
            [0,0,0,1]  #linear velocity error
        ])

    def getAB(
            self,
            theta:float
            ) -> list:
        """
        Calculates the parameters for linearization in x_t = A @ x_prev + B @ u_prev
        This is known as discrete time linear dynamics.
        The A and B matrices are instantaneous linearizations of the nonlinear system
        Updating these more often ensures you have a more accurate model of movement
        but may cause delays due to computational expense

        :param theta: 2D orientation of vehicle in radians -> float

        :return: A matrix ---> 4x4 NumPy array (models how system responds to a a given state)
        :return: B matrix ---> 4x2 NumPy array (models how system responds to control inputs)
        """

        A = np.array([
            [1,0,0,np.cos(theta)*self.dt],
            [0,1,0,np.sin(theta)*self.dt],
            [0,0,1,0],
            [0,0,0,1]
            ])

        B = np.array([
            [0,np.cos(theta)*self.dt],
            [0,np.sin(theta)*self.dt],
            [self.dt,0],
            [0,1]
        ])
        return [A,B]

    def getK(
            self,
            time_steps:float,
            A:np.ndarray,
            B:np.ndarray
            ) -> np.ndarray:
        """
        This function retrieves a gain matrix K that will allow us to find our control based on our state.
        The control input, u, is a linear function of the state and K, i.e. u_t = K @ x_t

        :param timeSteps: integer number of time steps you want until you reach the desired state
            -> int
        :param A: A matrix in linear model x_t = A @ x_prev + B @ u_prev ->
            4x4 NumPy array
        :param B: B matrix in linear model x_t = A @ x_prev + B @ u_prev ->
            4x2 NumPy array

        :return: K matrix ---> 2x4 NumPy array (gain matrix)
        """

        N =time_steps
        # Create a list of N + 1 elements, P is time varrying cost matrix based on both state and control effort
        P = [None] * (N + 1)
        # At the final time we dont have a control input so P is just our state cost
        P[N] = self.Q

        for i in range(N, 0, -1):
            # Discrete-time Algebraic Riccati Equation (ARE) to calculate the optimal
            # P matrix. This is a dynamic programing method, meaning we start
            # from the final time and work backwards to find our desired matrix
            P[i-1] = self.Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                self.R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

        # Calculate the optimal feedback gain K using the P matrix
        K = -np.linalg.pinv(self.R + B.T @ P[1] @ B) @ B.T @ P[1] @ A
        return K

    def getControl(
            self,
            x:np.ndarray,
            xd:np.ndarray,
            K:np.ndarray
            ) -> np.ndarray:
        """
        This uses the gain end state error to find the desired control input based on state, desired state, and
        the optimal gain matrix K. We also filter based on the input limitations to the Limo robot.
        The control inputs are changes in linear velocity and angular velocity.

        :param x: current state
        4x1 NumPy Array given the state is [x,y,theta,v] --->
            [meters, meters, radians, m/s]
        :param xd: desired state
        4x1 NumPy Array given the state is [x,y,theta,v] --->
            [meters, meters, radians, m/s]
        :param K: optimal gain matrix -> 2x4 NumPy array

        :return u_star:  optimal control input [delta theta dot, delta v] --->
            [radians/s, m/s] 2x1 NumPy array with optimal control input
        """

        # Find the state error
        x_error = xd-x
        # Find the optimal control
        u_star = K@x_error
        # Create varriable to check for clipping
        u_unclipped = u_star
        # Clip control based on min and max control inputs
        # u_star[1,0] = np.clip(u_star[1,0],-1-x[3,0], 1-x[3,0]) # Clipped between -1 and 1 m/s
        # u_star[0,0] = np.clip(u_star[0,0],np.sin(-1)*(u_star[1,0]+x[3,0])/self.L,np.sin(1)*(u_star[1,0]+x[3,0])/self.L) # clipped based on steering angle between -1 and 1 rads
        # Tell user if inputs were clipped
        if u_unclipped[1,0]!=u_star[1,0]:
            print('linear velocity clipped')
        if u_unclipped[0,0] != u_star[0,0]:
            print('angular velocity clipped')
        return u_star

    def lqr(
            self, x:np.ndarray,
            xd:np.ndarray,
            time:int,
            A:np.ndarray,
            B:np.ndarray
            )->np.ndarray:
        """
        Discrete-time linear quadratic regulator for the LIMO, you can either use this which finds a new gain matrix
        at every time step, or the previous three functions to tune how often you linearize and find the gains.

        :param x: The current state of the system
            4x1 NumPy Array given the state is [x,y,theta,v] ->
            [meters, meters, radians, m/s]
        :param xd: The desired state of the system
            4x1 NumPy Array given the state is [x,y,theta,v] ->
            [meters, meters, radians, m/s]
        :param time: The number of timesteps to stabalize -> int
        :param A: A matrix in linear model x_t = A @ x_prev + B @ u_prev ->
            4x4 NumPy array
        :param B: B matrix in linear model x_t = A @ x_prev + B @ u_prev ->
            4x2 NumPy array

        :return u_star: optimal control input [delta theta dot, delta v] --->
            [radians/s, m/s] 2x1 NumPy array with optimal control input
        """

        # Set error based on current and desired state
        x_error = xd-x

        N =time
        # Create a list of N + 1 elements, P is time varrying cost matrix based on both state and control effort
        P = [None] * (N + 1)
        # At the final time we dont have a control input so P is just our state cost
        P[N] = self.Q

        for i in range(N, 0, -1):
            # Discrete-time Algebraic Riccati Equation (ARE) to calculate the optimal
            # P matrix. This is a dynamic programing method, meaning we start
            # from the final time and work backwards to find our desired matrix
            P[i-1] = self.Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                self.R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

        # Calculate the optimal feedback gain K using the P matrix
        K = -np.linalg.pinv(self.R + B.T @ P[1] @ B) @ B.T @ P[1] @ A
        # Find optimal control input
        u_star = K @ x_error
        # Create varriable to check for clipping
        u_unclipped = u_star
        # Clip control based on min and max control inputs
        u_star[1,0] = np.clip(u_star[1,0],-1-x[3,0], 1-x[3,0]) # Clipped between -1 and 1 m/s
        u_star[0,0] = np.clip(u_star[0,0],np.sin(-1)*(u_star[1,0]+x[3,0])/self.L,np.sin(1)*(u_star[1,0]+x[3,0])/self.L) # clipped based on steering angle between -0.7 and 0.7 rads
        # Tell user if inputs were clipped
        if u_unclipped[1,0]!=u_star[1,0]:
            print('linear velocity clipped')
        if u_unclipped[0,0] != u_star[0,0]:
            print('angular velocity clipped')
        return u_star
