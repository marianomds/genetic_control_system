#!/usr/bin/env python

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zpk2tf 

# Default time vector parameters
START_TIME = 0
STOP_TIME = 5
STEP_TIME = 0.01

# Default limit values for controller parameters
K_MAX = 100
ZERO_MIN = -20

# Default threshold values for algorithm convergence
OVERSHOOT_MAX = 20 # in percentage
RISE_TIME_MAX = 3 # in seconds

def overshoot(signal):
    return (signal.max() - signal[-1])*100

def rise_time(signal):
    result = next((varvec[0] for varvec in enumerate(signal) if varvec[1] > signal[-1]), signal.size)
    return result * STEP_TIME

def evolution(Gp, Time):

    ov = 999
    rt = 999
    while ((ov > OVERSHOOT_MAX) or (rt > RISE_TIME_MAX)):

        K = np.random.uniform(0, K_MAX)
        Z = np.random.uniform(ZERO_MIN, 0)

        # Create PI controller
        (Gc_num,Gc_den) = zpk2tf([Z],[0],K) # PI controller, 2 parameters: location of 1 zero, value of K, (+ 1 pole always in origin)
        Gc = ctrl.tf(Gc_num,Gc_den)

        # Closed loop system
        M = ctrl.feedback(Gc*Gp,1)

        # Closed loop step response
        (y,t) = ctrl.step(M, Time)

        # Evaluate overshoot and rise time
        ov = overshoot(y)
        rt = rise_time(y)

    # Print controller information
    print('\nController:')
    print('k: %f' % K)
    print('zero: %f' % Z)
    print(Gc)

    # Print closed loop transfer function
    print('Closed loop transfer:')
    print(M)

    return Gc, M


if __name__ == "__main__":

    # Create Plant to be controlled
    (Gp_num,Gp_den) = zpk2tf([],[-1, -2, -8],1)
    Gp = ctrl.tf(Gp_num,Gp_den)

    # Print transfer function of plant to be controlled
    print('Plant:')
    print(Gp)

    # Create time vector to be used
    T = np.arange(START_TIME, STOP_TIME, STEP_TIME)

    # Generate input and plant step response
    (y1,t1) = ctrl.step(ctrl.tf([1, 0],[1, 0]), T) # with ([1],[1]) step throws an error
    (y2,t2) = ctrl.step(Gp, T)

    # Run evolution algorithm
    Gc, M = evolution(Gp, T)

    # Closed loop step response
    (y3,t3) = ctrl.step(M, T)

    # Evaluate and print overshoot and rise time
    print('Overshoot: %f %%' % overshoot(y3))
    print('Rise time: %1.2f sec' % rise_time(y3))

    # Plot result
    plt.plot(t1, y1, t2, y2, t3, y3)
    plt.draw()

    # Block until the plot window is closed
    plt.show()
