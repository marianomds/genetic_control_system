#!/usr/bin/env python

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zpk2tf 

# Time vector parameters
START_TIME = 0
STOP_TIME = 20
STEP_TIME = 0.01

# Input parameters
FINAL_VALUE = 2
FINAL_TIME = 10 # MUST BE: FINAL_TIME < STOP_TIME

# Limit values for controller parameters
K_MAX = 100
ZERO_MIN = -20

# Threshold values for algorithm convergence
OVERSHOOT_MAX = 3 # in percentage
RISE_TIME_MAX = 11 # MUST BE: RISE_TIME_MAX > FINAL_TIME

def create_input():
    input_signal = np.arange(START_TIME, FINAL_TIME, STEP_TIME)*(FINAL_VALUE/FINAL_TIME)
    input_signal = np.append(input_signal, np.ones(int((STOP_TIME - FINAL_TIME)/STEP_TIME))*FINAL_VALUE)
    return input_signal

def overshoot(signal):
    return (signal.max() - signal[-1])*100

def rise_time(signal):
    result = next((varvec[0] for varvec in enumerate(signal) if varvec[1] > signal[-1]), signal.size)
    return result * STEP_TIME

def mse(signal1, signal2):
    return np.mean((signal1 - signal2)**2)

def evolution(Gp, Time, Input):

    ov = 999
    rt = 999
    while ((ov > OVERSHOOT_MAX) or (rt > RISE_TIME_MAX)):

        K = np.random.uniform(0, K_MAX)
        Z = np.random.uniform(ZERO_MIN, 0)

        # Create PI controller
        (Gc_num,Gc_den) = zpk2tf([Z],[0],K) # PI controller, 2 parameters: location of 1 zero, value of K, (+ 1 pole always in origin)
        Gc = ctrl.tf(Gc_num,Gc_den)

        # Evaluate closed loop stability
        gm, pm, Wcg, Wcp = ctrl.margin(Gc*Gp)
        if gm <= 1: # Only consider closed loop stable (gm > 0dB)
            continue

        # Closed loop system
        M = ctrl.feedback(Gc*Gp,1)

        # Closed loop step response
        y, t, xout = ctrl.lsim(M, Input, Time)

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

    # Generate input signal
    input_signal = create_input()

    # Generate plant response to input signal
    y1, t1, xout = ctrl.lsim(Gp, input_signal, T)

    # Run evolution algorithm
    Gc, M = evolution(Gp, T, input_signal)

    # Closed loop response to input signal
    y2, t2, xout = ctrl.lsim(M, input_signal, T)

    # Evaluate and print overshoot and rise time
    print('Overshoot: %f %%' % overshoot(y2))
    print('Rise time: %1.2f sec' % rise_time(y2))
    print('MSE: %f' % mse(y2, input_signal))

    # Plot result
    plt.plot(T, input_signal, t1, y1, t2, y2)
    plt.draw()

    # Block until the plot window is closed
    plt.show()
