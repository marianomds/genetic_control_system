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
FINAL_TIME = 5 # MUST BE: FINAL_TIME < STOP_TIME
IN_TYPE = 'SIGMOID' # Options: 'STEP', 'RAMP', 'SIGMOID'

# Limit values for controller parameters
ZERO_MIN = -20

# Metric to be optimized
OPTIMIZE = 'MSE' # Options: 'OV', 'RT, 'MSE'

# Threshold values for algorithm convergence
OVERSHOOT_TH = 3 # in percentage
RISE_TIME_TH = 11
MSE_TH = 0.01

def sigmoid (x):
    return 1/(1 + np.exp(-x))

# Function for creating input signal to the system
def create_input():
    if IN_TYPE == 'STEP':
        input_signal = np.zeros(int(FINAL_TIME/STEP_TIME))
    elif IN_TYPE == 'RAMP':
        input_signal = np.arange(START_TIME, FINAL_TIME, STEP_TIME)*(FINAL_VALUE/FINAL_TIME)
    elif IN_TYPE == 'SIGMOID':
        x_tanh = np.arange(START_TIME, FINAL_TIME, STEP_TIME)
        input_signal = sigmoid(x_tanh*(12/FINAL_TIME) - 6) * FINAL_VALUE
    else:
        print('Incorrect input type.')
        quit()

    input_signal = np.append(input_signal, np.ones(int((STOP_TIME - FINAL_TIME)/STEP_TIME))*FINAL_VALUE)
    
    return input_signal

def overshoot(signal):
    return (signal.max() - signal[-1])*100

def rise_time(signal):
    result = next((varvec[0] for varvec in enumerate(signal) if varvec[1] > signal[-1]), signal.size)
    return result * STEP_TIME

def mse(signal1, signal2): # Mean squared error
    return np.mean((signal1 - signal2)**2)

def evaluate(x, y):
    # Return fitness evaluation depending on metric to optimize
    if OPTIMIZE == 'OV':
        return overshoot(y)
    elif OPTIMIZE == 'RT':
        return rise_time(y)
    elif OPTIMIZE == 'MSE':
        return mse(x, y)
    else:
        print('Incorrect optimization metric.')
        quit()

def evolution(Gp, Time, Input):

    # Variable for storing best fitness
    fitness_best = 999 # Initial large enough value to ensure entering while loop

    # Number of not closed loop stable tries discarded 
    margin_discarded = 0

    # Select fitness threshold depending on metric to optimize
    if OPTIMIZE == 'OV':
        fitness_th = OVERSHOOT_TH
    elif OPTIMIZE == 'RT':
        fitness_th = RISE_TIME_TH
    elif OPTIMIZE == 'MSE':
        fitness_th = MSE_TH
    else:
        print('Incorrect optimization metric.')
        quit()
    
    # Array for storing history of best fitness individuals
    fitness_best_vec = np.array([])

    # Number of loops
    loop_n = 0

    # Keep entering while loop until fitness threshold is reached
    while (fitness_best > fitness_th):

        loop_n += 1

        # Try random parameter values
        Z1 = np.random.uniform(ZERO_MIN, 0)
        Z2 = np.random.uniform(ZERO_MIN, 0)

        # Create test PI controller (used to calculate the maximum K for closed loop stable)
        (Gc_num,Gc_den) = zpk2tf([Z1, Z2],[0],1) # PID controller, 3 parameters: location of 2 zeros, value of K, (+ 1 pole always in origin)
        Gc = ctrl.tf(Gc_num,Gc_den)

        # Evaluate closed loop stability
        gm, pm, Wcg, Wcp = ctrl.margin(Gc*Gp)

        # Dischard solutions with no gain margin
        if gm == None:
            margin_discarded += 1
            continue

        # If K < gm => closed loop stable (gm > 0dB)
        K = np.random.uniform(0, gm)

        # Create PI controller for closed loop stable system
        (Gc_num,Gc_den) = zpk2tf([Z1, Z2],[0],K) # PID controller, 3 parameters: location of 2 zeros, value of K, (+ 1 pole always in origin)
        Gc = ctrl.tf(Gc_num,Gc_den)

        # Closed loop system
        M = ctrl.feedback(Gc*Gp,1)

        # Closed loop step response
        y, t, xout = ctrl.lsim(M, Input, Time)

        # Evaluate fitness
        fitness = evaluate(Input,y)

        # If better fitness value is found, save (best) parameters
        if fitness < fitness_best:
            fitness_best = fitness
            K_best = K
            Z1_best = Z1
            Z2_best = Z2

        # Save history of best fitness in a vector for plotting
        fitness_best_vec = np.append(fitness_best_vec, fitness_best)

        # Real time plot of history of best fitness
        plt.plot(fitness_best_vec)
        plt.pause(0.001)

    # Create best PI controller
    (Gc_num_best,Gc_den_best) = zpk2tf([Z1_best, Z2_best],[0],K_best) # PI controller, 2 parameters: location of 1 zero, value of K, (+ 1 pole always in origin)
    Gc_best = ctrl.tf(Gc_num_best,Gc_den_best)

    # Best closed loop system
    M_best = ctrl.feedback(Gc_best*Gp,1)

    print('Total number of evaluations: %d' % loop_n)
    print('Number of discarded (not closed loop stable): %d' % margin_discarded)

    # Print controller information
    print('\nController:')
    print('k: %f' % K_best)
    print('zero 1: %f' % Z1_best)
    print('zero 2: %f' % Z2_best)
    print(Gc_best)

    # Print closed loop transfer function
    print('Closed loop transfer:')
    print(M_best)

    return Gc_best, M_best


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
    plt.figure()
    plt.plot(T, input_signal, t1, y1, t2, y2)
    plt.draw()

    # Block until the plot window is closed
    plt.show()

