"""
This file contains a PID controller for the RASTIC Limo robots. 
We send the robot two controls using this file, a steering angle
and a linear velocity. The steering angle is determined by the
cross-product between a unit vector from the robot in the direction 
of its heading and the vector from the robot to its target.
The linear velocity is determined by the dot-product of these 
vectors.
"""
import numpy as np
import math

class PID:
    #The PID object
    def __init__(self,steer_kp = 1.,steer_ki=0.0045,steer_kd=0.0017,vel_kp = 1.,vel_ki=0.0045,vel_kd=0.0017,dt = .01):
        #The init values are tunable, I settled on values that worked okay for me. Further tuning may be reasonable

        # The control gains for the steering input
        self.steer_kp = steer_kp
        self.steer_ki = steer_ki
        self.steer_kd = steer_kd

        # The control gains for the velocity input
        self.vel_kp = vel_kp
        self.vel_ki = vel_ki
        self.vel_kd = vel_kd

        # How many seconds before a new control is published
        self.dt = dt


    def steerControl(self,x,x_d,prev_e,prev_int):
        # Finds the steering control

        # First find unit vector form robot in direction of heading
        unit_theta = np.array([
            [math.cos(x[2,0])],
            [math.sin(x[2,0])]
        ])

        # Find vector from robot to desired point
        ref = x_d[:2]-x[:2]

        # Steering error is the cross-product
        e = np.cross(unit_theta.T,ref.T)[0]

        # Update the integral of the error
        e_int = prev_int+e*self.dt
        # This ensures it does not blow up 
        e_int = max(min(e_int,0.5),-0.5)

        # Approximate the derivative of error
        e_der = (e-prev_e)/self.dt

        #Find the steering control
        u_steer = self.steer_kp*e+self.steer_ki*e_int+self.steer_kd*e_der

        # The steering angle ranges from -0.7 to 0.7 radians
        # Clip to ensure we fall within that range
        u_steer = max(min(u_steer,0.7),-0.7)

        # Return control value and updated error and error integral values
        # to prepare for the next step
        return u_steer, e, e_int

    def speedControl(self,x,x_d,prev_e,prev_int):
        # Finds the steering control

        # First find unit vector form robot in direction of heading
        unit_theta = np.array([
            [math.cos(x[2,0])],
            [math.sin(x[2,0])]
        ])

        # Find vector from robot to desired point
        ref = x_d[:2]-x[:2]

        # Velocity error is the dot product
        e = np.dot(unit_theta.T,ref)[0,0]

        # Update the integral of the error
        e_int = prev_int+e*self.dt
        # This ensures it does not blow up 
        e_int = max(min(e_int,0.5),-0.5)

        # Approximate the derivative of error
        e_der = (e-prev_e)/self.dt

        # Find the VELOCITY control
        u_vel = self.vel_kp*e+self.vel_ki*e_int+self.vel_kd*e_der

        # The velocity ranges from -1.0 to 1.0 meters/second
        # Clip to ensure we fall within that range
        u_vel = max(min(u_vel,0.7),-0.7)

        # Return control value and updated error and error integral values
        # to prepare for the next step
        return u_vel, e, e_int
