#!/usr/bin/env python3
# coding=UTF-8
"""
This file contains the ROS setup for AgileX Limos in Rastic. This code uses ROS 1. 
At time of writing (June 2024), the Rastic computer uses the NOETIC ROS Distrobution.
This code was adapted by Carter Berlind to be generalized and act as an educational resource.
Significant changes to this file were made upon its adaptation
Original code recieved from Sienna Chien, original authors not listed.

Description: ROS infrastructure for communicating with AgileX Limo
"""
import math
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDrive


class LIMO():
    def __init__(
            self, 
            node_name:str,
            rate:int = 10
            ) -> None:
        """
        This class will serve as an obeject containing the node responsible for 
        communicating with the robot and containg up-to-date information on the robot's state.

        :param node_name: this will name the node that communicates with the robot, e.g. "limo770"
            -> str
        :param rate: this is target publication frequency in Hz
            -> int
        """

        self.node_name = node_name
        # I place the initial position at the origin but once this node 
        # is run, this we be imdiately updated. 
        # This code does not use all of these,
        # but if you want to experiment with different control methods 
        self.position_z = 0
        self.position_x = 0
        self.position_yaw = 0
        self.velocity = 0
        self.acceleration = 0
        self.Receivedata = 0
        self.position_ip_z = 0
        self.position_ip_x = 0

        #Creates node, subscribes to motioncapture pose and publishes control info to limo
        rospy.init_node("listen_pos", anonymous=True)
        self.sub = rospy.Subscriber('/vrpn_client_node/'+self.node_name+'/pose', PoseStamped, self.callback,queue_size=1)
        self.pub = rospy.Publisher('vel_steer_'+self.node_name,AckermannDrive,queue_size=1) #topic name = CAV_Data
        rospy.Rate(rate)

    def callback(
            self, 
            msg:PoseStamped
            ) -> None:
        """
        This function is run everytime there is new information in the topic the node subscribes to.
        In this case, it updates the position information fomr the robot based on the motion capture.

        :param msg: This is the message recieved from the mocap with an updated position
            -> PoseStamped
        """

        # Update the 2D pose of the robot. 
        # Note that the Mocap is oriented such that the z-x plane is the floor
        self.position_z = msg.pose.position.z
        self.position_x = msg.pose.position.x

        # Update the orientation information
        # The Mocap send us orientation information in quaternions
        # This means we need to run a conversion if we want Euler Angles (Pitch,Yaw,Roll)
        # Because our robot is modeled as planar, we only care about the yaw
        [_,yaw,_] = self.quaternion_to_euler(msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w)
        self.position_yaw =  yaw

        #This will only be 1 if you recieve mocap data, you use this to troupbleshoot
        self.Receivedata=1

    def steeringAngleToSteeringCommand(
            self,
            ref_angle:float
            ) -> float:
        """
        In general, we like to say that our control input ranges ranges from -1 to 1.
        However, the steering inputs range from -0.7 radians to 0.7 radians.
        This converts a relative control input to an actual steering angle.

        :param refAngle: relative control input for steering
            -> float

        :return y: actual control input for steering
            ---> float
        """
        x = ref_angle
        y = 0.7*x
        return y

    def quaternion_to_euler(
            self, 
            x:float, 
            y:float, 
            z:float, 
            w:float
            ) -> list:
        """
        The motion capture tells us the orientation in quaternions. 
        Quaternions are a powerful tool for computation of angles in 3D
        However, our robots our planar so we only really want the yaw.
        This means we need to run a convesion which moves us from quaternions to euler angles.
        For more information on quaternions, see: https://www.allaboutcircuits.com/technical-articles/dont-get-lost-in-deep-space-understanding-quaternions/
        
        :param x: quaternion x imaginary part 
            -> float
        :param y: quaternion y imaginary part
            -> float
        :param z: quaternion z imaginary part
            -> float
        :param w: quaternion real part
            -> float

        :return x: Rotation about the x axis in radians
            ---> float
        :return y: Rotation about the y axis in radians
            ---> float
        :return z: Rotation about the z axis in radians
            ---> float 
        """

        # Find the roation about the x axis
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        x = math.atan2(t0, t1)

        # Find the roation about the y axis
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        y = math.asin(t2)

        # Find the roation about the z axis
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        z = math.atan2(t3, t4)

        # Note that because the floor is the z-x plane, the yaw is the
        # roation about the y axis
        return [x, y, z] # in radians

    def control(self,
            v_ref:float, 
            steer_ref:float
            ) -> AckermannDrive:
        """
        This creates a message that can be sent to the robot over ROS.
        The dynamic model for the robot, i.e. the mathmatical relationship
        that tells us how the robot moves based on control inputs and velocity,
        is known as Ackerman steering. You can find this and similar models 
        by looking into kinematic bicycles and dubins vehicles.

        :param v_ref: The desired linear velocity that you want the robot to move at in m/s
            -> float
        :param steer_ref: The desired steering angle in radians
            -> float

        :return drive_msg: The command that we send to the robot
            ---> AckermanDrive
        """

        drive_msg = AckermannDrive()
        drive_msg.speed = v_ref
        drive_msg.steering_angle = self.steeringAngleToSteeringCommand(steer_ref)
        return drive_msg
