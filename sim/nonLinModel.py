import numpy as np
import math

def nonlinearModelStep(x_t:np.ndarray,u:np.ndarray,dt:float)->np.ndarray:
    beta = math.atan2(math.tan(u[0,0]),2)
    x_t[3,0]=0.
    x1_dot = u[1,0]*math.cos(beta+x_t[2,0])
    x2_dot = u[1,0]*math.sin(beta+x_t[2,0])
    theta_dot=u[1,0]*math.tan(u[0,0])*math.cos(beta)/0.2
    v = u[1,0]
    x_dot = np.array([
        [x1_dot*dt],
        [x2_dot*dt],
        [theta_dot*dt],
        [v]
        ])
    return x_t+x_dot

def unicycleModelStep(x_t:np.ndarray,u:np.ndarray,dt:float)->np.ndarray:
    x_t[3,0]=0
    x1_dot = u[1,0]*math.cos(x_t[2,0])
    x2_dot = u[1,0]*math.sin(x_t[2,0])
    theta_dot = u[0,0]
    v = u[1,0]
    x_dot = np.array([
        [x1_dot*dt],
        [x2_dot*dt],
        [theta_dot*dt],
        [v]
        ])
    return x_t+x_dot
