This file contains the physical implementation for the Limos.

Currently, I have not been able to implement this code on the Limos,
but I will include details on how the code is intended to work.

For information on Limo set up see https://github.com/SiennaChien/RASTIC-MoCap-Limo-Tutorial/blob/main/README.md
or reach out to Sienna Chien for a walk through siennac@bu.edu

At the time of writing, this procedure is correct with some exceptions.
The last three digits of the IP for the computer we use are 140,
not 139. The listener file I was attempting to use was the robot.py 
file in the src of the model_validation_ws found on the Limos.

To use this code, extra steps are needed compared to the sim.
Virtual environments often do not play nicely with ROS (in my experience) so I split up 
running this part of the code into multiple parts, one in the venv and one out of it.

I recommend having two tabs open in this file, one with the venv active and one without.
In the active one run the test_phys.py script to generate a trajectory.

Before you run the trajectory on the Limos, take the following steps:

1) open the run_cycle_phys.py script

2) scroll to the bottom and in the following lines, make sure the Limo number matches the one you are using

```
if __name__ == "__main__":
    #place the number of the Limo that you are using here
    tracker = Tracker("limo777")
    testTraj(tracker)
```

3) in the testTraj function at the top of the file, find the following line

```
points,_ = loadPoints(1,6)
```

4) Make sure that the first number passed is the trial number you want to run, and the second
   matches the number of trajectory segments (note that due to Python indexing, if the largest index is 5
   then the number of segments is 6)

5) once this is adjusted run the run_cycle_phys.py file to have the Limo follow the cycle

The breakdown of the code is this:

The run_cycle_phys.py loads the trajectory and calls a tracking controller from RUN_LIMO_phys.py.
The tracking controller will use either a PID or LQR defined in LIMO_PID_phys.py or
LIMO_LQR_phys.py respectively. The ROS node setup is found is found in LIMO_ROS_SETUP.py
but the node is called in the tracker object in RUN_LIMO_phys.py.

The issue I encountered was this, when I checked on board the robot, it would not see the 
control info publishing to the correct topic but on board, I could see that they were publishing.
Note that I ensured to export the master URI and ROS IPs for both the LIMO and RASTIC computer.

I do not know the source of the issue but I recommend asking folks who are managing to use the Limos successfully.

Instructions to turn off the RRBT visualization are the same as in the sim file.


