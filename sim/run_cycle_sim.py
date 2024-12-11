"""
This script can be used to follow a previously found trajectory
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import json
import test_sim
import RUN_LIMO_sim
from RUN_LIMO_sim import *
from hytoperm import Experiment, Domain, extendKeywordArgs
from LIMO_PID_sim import PID


def testTraj(tracker: Tracker):
    # For a given trajectory that has been solved for this will test it
    points, _ = loadPoints(1, 6)

    # this sets an initial point for use in simulation.
    # on real limos, this is not needed
    tracker.x = points[:, 2:3]

    # this calls the tracking controller to follow the trajectorytracks the
    tracker.trackTrajectoryPID(points[:, 2:], stab_time=10)


def angleCorrection(thetas):
    # This function ensures that each angle is represented with a value minimum distance form its neighbor.
    # e.g. if your angle is 3pi/4, and you want to converge to pi,
    # this makes sure that the angle listed is not -pi.
    out = np.zeros(thetas.shape)
    out[0, 0] = thetas[0, 0]
    for i in range(thetas.shape[1]-1):
        signal = 0
        out[0, i+1] = thetas[0, i+1]
        while signal != 1:
            if abs(out[0, i+1]-2*np.pi-out[0, i]) < abs(out[0, i+1]-out[0, i]):
                out[0, i+1] -= 2*np.pi
            elif abs(out[0, i+1]+2*np.pi-out[0, i]) < abs(out[0, i+1]-out[0, i]):
                out[0, i+1] += 2*np.pi
            else:
                signal = 1
    return out


def getThetas(points):
    # This function retrieves the orientation of the robot along the trajectory
    # You will need this if using the LQR controller
    thetas = np.zeros((1, points.shape[1]))
    for i in range(points.shape[1]):
        if i == 0:
            # For the first value, the angle is that between the first and second point
            thetas[0, i] = math.atan2(
                points[1, 1]-points[1, 0], points[0, 1]-points[0, 0])
        elif i != points.shape[1]-1:
            # For all but the first and last points, the angle is that between the previous and next point
            thetas[0, i] = math.atan2(
                points[1, i+1]-points[1, i-1], points[0, i+1]-points[0, i-1])
        else:
            # For the final point, the angle is that between the penultimate and final points.
            thetas[0, i] = math.atan2(
                points[1, i]-points[1, i-1], points[0, i]-points[0, i-1])
    return thetas


def getVels(controls, hybrid_dynamics):
    # Retrieves the control input velocities from the original planner
    # This will again be used for the LQR  controller

    vels = np.zeros((1, controls.shape[1]))
    for i in range(vels.shape[1]):
        vel = controls[:, i:i+1] + hybrid_dynamics
        # We only care about the magnitude, the direction is approximated in the getThetas function
        vels[0, i] = np.linalg.norm(vel)
    return vels


def loadPoints(num):
    # Retrieves the points from the JSON files they are stored in
    # The num input is the trial number. The trajectory will be stored in the directory
    # with this trial number.

    # We start with loading the trajectory from individual segments
    points_dict = {}
    uts_dict = {}
    hds_dict = {}
    vels_dict = {}

    # Iterate through the number number of segments
    count = 0
    while True:

        try:
            # save the trajectory points
            f = open(f'trial{num}/cycleInfo{num}_{count}_points.json', 'r')
            points_dict[count] = json.loads(f.readline())

            # save the trajectory controls
            f = open(f'trial{num}/cycleInfo{num}_{count}_cntrls.json', 'r')
            uts_dict[count] = json.loads(f.readline())

            # Get the hybrid dynamics of the region
            f = open(f'trial{num}/cycleInfo{num}_{count}_dynams.json', 'r')
            hds_dict[count] = np.array(json.loads(f.readline()))
            # reshape the hybrid dynamics to a 2D vector
            hds_dict[count].reshape((2, 1))
            # Use the controls and dynamics to get the desired velocity
            vels_dict[count] = getVels(
                np.array(uts_dict[count]), hds_dict[count])

            count += 1

        except:
            break

    tot = count

    # compose points and velocities into a single array
    pts = np.array(points_dict[0])
    vels = np.array(vels_dict[0])
    for count in range(tot-1):
        pts = np.hstack((pts, np.array(points_dict[count+1])))
        vels = np.hstack((vels, np.array(vels_dict[count+1])))

    return pts, None

    # Find the thetas based on the points
    thetas = getThetas(pts)

    # This is helpful for the LQR, it allows makes sure that the angles
    # are represented in the easiest way to converge
    # e.g. if your angle is 3pi/4, and you want to converge to pi,
    # this makes sure that the angle listed is not -pi.
    thetas = angleCorrection(thetas)

    # Planning is done in a 1x1 world. scaling up the world slows
    # down the solving process, for this reason, we scale it up by 5 here
    # to be more realistic for the limos dynamics. The offset assumes the origin is in
    # the middle of the arena (roughly true in rastic)
    points = np.vstack((5*pts-2.5, thetas))

    # I have issues with the number of points and velocities being different.
    # This ensures they are the same
    if points.shape[1] != vels.shape[1]:
        min_shape = min(points.shape[1], vels.shape[1])
        points = points[:, :min_shape]
        vels = vels[:, :min_shape]
    # we also scale up the velocities to match the world scaling
    points = np.vstack((points, 5*vels))

    # From here we get the total trajectories not split into segments
    # We assume no hybrid dynamics. If you desire to use them, the previous points array
    # will need to be used
    hd = np.array([[0.0], [0.0]])

    # Retrieve the trajectory
    f = open(f'trial{num}/cycleInfo{num}_total_points.json', 'r')
    pts_2 = np.array(json.loads(f.readline()))
    f = open(f'trial{num}/cycleInfo{num}_total_cntrls.json', 'r')
    uts_2 = np.array(json.loads(f.readline()))

    # find the angles
    thetas_2 = getThetas(pts_2)
    thetas_2 = angleCorrection(thetas_2)

    # find the velocities
    vels_2 = getVels(uts_2, hd)
    points_2 = np.vstack((5*pts_2-2.5, thetas_2))

    # make sure the number of points and velocities are the same
    if points_2.shape[1] != vels_2.shape[1]:
        min_shape = min(points_2.shape[1], vels_2.shape[1])
        points_2 = points_2[:, :min_shape]
        vels_2 = vels_2[:, :min_shape]
    points_2 = np.vstack((points_2, 5*vels_2))

    # returned the stitched and unstitched trajectories
    return points, points_2

class ProblemSetup:
    def __init__(self, **kwargs):

        # World building
        rastic = Domain(xrange=[-4.5,4.5], yrange=[-3,3])                       # The default domain: our RASTIC environment at BU
        self._domain: Domain = kwargs.get('domain', rastic)                     # The domain specifications of the experiment
        self._nsets: int = kwargs.get('n_sets', 15)                             # The number of sets in the partition
        self._fraction: float = kwargs.get('fraction', 0.2)                     # The fraction of the sets that should be target sets
        self._trial: int = kwargs.get('trial', 1)                               # The trial number
        self._limo_name: str = "limo770"                                        # The name of the limo

        # Controller specifications
        self._dt: float = kwargs.get('dt', 0.01)                                # The time step
        self._PID: PID = kwargs.get('pid', PID())                               # The PID controller
        self._pid_tolerance: float = kwargs.get('pid_tolerance', 0.1)           # The tolerance for the PID controller (convergence radius to waypoints)    
        

    def getKwargs(self) -> dict:
        return {
            'domain': self._domain,
            'n_sets': self._nsets,
            'fraction': self._fraction,
            'trial': self._trial,
            'limo_name': self._limo_name,
            'dt': self._dt,
            'pid': self._PID,
            'pid_tolerance': self._pid_tolerance,
        }
    
if __name__ == "__main__":
    
    # Specify everything about the experiment
    ps = ProblemSetup()
    
    #### IF YOU WANT TO MAKE CHANGES TO SETUP CHANGE ps HERE ####
    #### e.g., ps._dt = 0.1

    # Try to load points from specified trial. If it doesn't exist, we will generate a new trial
    try:
        points, _ = loadPoints(ps._trial)
        ex = Experiment.deserialize(f"trial{ps._trial}/Experiment.pickle")
    except:
        world = test_sim.World(**ps.getKwargs())
        world.solve()
        world.export()
        points, _ = loadPoints(world.trial)
        ex = world.ex

    # When creating the tracker specify which limo you are using
    tracker = Tracker(**ps.getKwargs())

    # this calls the tracking controller to follow the trajectory
    data = tracker.trackTrajectoryPID(points[:, :])
    plotCyclesim(data, ex=ex)
