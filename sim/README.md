This folder contains the necessary code to run a simulated version of
the hytoperm implementation. The main file needed to run the sim is
the test.py file. 

In this folder, the RUN_LIMO_sim.py file contains the tracking controller.

The LIMO_LQR_sim.py and LIMO_PID_sim.py contain classes which
store the construction of the LQR and PID respectively,
i.e. the tunable parameters and functions which solve for individual controls.

nonLinModel.py contains a dynamic model that exactly matches a discrete-time version
of Ackerman steering motion model. This is NOT a linearized model and can NOT be used
directly with the LQR. The LQR file contains a linearization of this model.