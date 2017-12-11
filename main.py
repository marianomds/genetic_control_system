#!/usr/bin/env python

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zpk2tf 
from copy import deepcopy

# Time vector parameters
START_TIME = 0
STOP_TIME = 20
STEP_TIME = 0.01

# Plant parameters
PLANT_ZEROS = [] # empty: no zeros
PLANT_POLES = [-1, -2, -8]
PLANT_K = 1

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

# Genetic algorithm parameters
POPULATION_SIZE = 10
MAX_GEN = 10 # maximum number of generations
CROSS_OVER_P = 0.5 # probability of crossing over


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
    return (signal.max() - signal[-1])*100 # in percent

def rise_time(signal):
    result = next((varvec[0] for varvec in enumerate(signal) if varvec[1] > signal[-1]), signal.size)
    if IN_TYPE == 'STEP':
        return (result * STEP_TIME) - FINAL_TIME
    else:
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

def cross_over(population):
 
    # Initial values for the state machine
    parent_ind = 1
    state = 1

    # State Machine
    while(state <= 3):

        if state == 1: # Selection of the first parent
            if parent_ind >= POPULATION_SIZE: # As long as the index is smaller than the population size it will keep on selecting parents
                state = 4
            elif np.random.uniform(0, 1) > CROSS_OVER_P:
                parent_ind += 1
            else: # There is a probability CROSS_OVER_P for the indexed individual to be selected as a parent
                P1 = parent_ind - 1
                parent_ind += 1
                state = 2
      
        elif state == 2: # Selection of the second parent
            if parent_ind > POPULATION_SIZE:
                state = 4
            elif np.random.uniform(0, 1) > CROSS_OVER_P:
                parent_ind += 1
            else:
                P2 = parent_ind - 1
                parent_ind += 1
                state = 3

        elif state == 3: # Crossing over
            Ch1 = deepcopy(population[P1])
            Ch2 = deepcopy(population[P2])

            code = np.round(np.random.random(3))
            while(0 == code.sum() or 3 == code.sum()): # If all zeros or ones in code 
                code = np.round(np.random.random(3))

            if code[0] == 1:
                Ch1.Z1 = population[P2].Z1
                Ch2.Z1 = population[P1].Z1

            if code[1] == 1:
                Ch1.Z2 = population[P2].Z2
                Ch2.Z2 = population[P1].Z2

            if code[2] == 1:
                Ch1.K = population[P2].K
                Ch2.K = population[P1].K

            # Append the 2 new children to the end of the population list
            population.append(Ch1)
            population.append(Ch2)

            state = 1

    return population

def selection(population, Gp, Time, Input):

    # Compute the fitness of all the children of the new population
    for ind in range(POPULATION_SIZE, len(population)):
        population[ind].fitness_calc(Gp, Time, Input)

    # Order the population by best (lower) fitness
    population = sorted(population, key=lambda individual: individual.fitness)

    # Truncate the population so that only the POPULATION_SIZE best individuals remain
    del population[POPULATION_SIZE:]

    return population

class individual():

    # Initialization funcion
    def __init__(self, Gp, Time, Input):
        while(True):
            # Try random parameter values
            self.Z1 = np.random.uniform(ZERO_MIN, 0)
            self.Z2 = np.random.uniform(ZERO_MIN, 0)

            # Create test PI controller (used to calculate the maximum K for closed loop stable)
            (self.Gc_num,self.Gc_den) = zpk2tf([self.Z1, self.Z2],[0],1) # PID controller, 3 parameters: location of 2 zeros, value of K, (+ 1 pole always in origin)
            self.Gc = ctrl.tf(self.Gc_num,self.Gc_den)

            # Evaluate closed loop stability
            gm, pm, Wcg, Wcp = ctrl.margin(self.Gc*Gp)

            # Dischard solutions with no gain margin
            if gm == None:
                continue

            # If K < gm => closed loop stable (gm > 0dB)
            self.K = np.random.uniform(0, gm)

            # Create PI controller for closed loop stable system
            (self.Gc_num,self.Gc_den) = zpk2tf([self.Z1, self.Z2],[0],self.K) # PID controller, 3 parameters: location of 2 zeros, value of K, (+ 1 pole always in origin)
            self.Gc = ctrl.tf(self.Gc_num,self.Gc_den)

            # Closed loop system
            self.M = ctrl.feedback(self.Gc*Gp,1)

            # Closed loop step response
            y, t, xout = ctrl.lsim(self.M, Input, Time)

            # Evaluate fitness
            self.fitness = evaluate(Input,y)

            break

    def fitness_calc(self, Gp, Time, Input):

            # Create PI controller
            (self.Gc_num,self.Gc_den) = zpk2tf([self.Z1, self.Z2],[0],self.K) # PID controller, 3 parameters: location of 2 zeros, value of K, (+ 1 pole always in origin)
            self.Gc = ctrl.tf(self.Gc_num,self.Gc_den)

            # Evaluate closed loop stability
            gm, pm, Wcg, Wcp = ctrl.margin(self.Gc*Gp)

            # Dischard solutions with no gain margin
            if gm == None or gm <= 1:
                self.fitness = 999
                return

            # Closed loop system
            self.M = ctrl.feedback(self.Gc*Gp,1)

            # Closed loop step response
            y, t, xout = ctrl.lsim(self.M, Input, Time)

            # Evaluate fitness
            self.fitness = evaluate(Input,y)


def evolution(Gp, Time, Input):

    # Variable for storing best fitness
#    fitness_best = 999 # Initial large enough value to ensure entering while loop


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
    
    # Population: list of individual objects
    population = []

    # Array for storing history of best fitness individuals
    fitness_best_vec = np.array([])

    # Create random population
    for count in range(POPULATION_SIZE):
        x = individual(Gp, Time, Input)
        population.append(x)

    # Create random best individual
    best = individual(Gp, Time, Input)

    # Number of loops
    loop_n = 0

    # Keep entering while loop until fitness threshold is reached
    while (loop_n < MAX_GEN and best.fitness > fitness_th):

        loop_n += 1

        print('old')
        for ind in range(len(population)):
            print(population[ind].fitness)

        population = cross_over(population)
        # mutation(population)
        population = selection(population, Gp, Time, Input)

        print('new')
        for ind in range(len(population)):
            print(population[ind].fitness)

#        # If better fitness value is found, save (best) parameters
#        if fitness < fitness_best:
#            fitness_best = fitness
#            K_best = K
#            Z1_best = Z1
#            Z2_best = Z2

        # Save history of best fitness in a vector for plotting
        fitness_best_vec = np.append(fitness_best_vec, best.fitness)

        # Real time plot of history of best fitness
        plt.subplot(1,2,1)
        plt.plot(fitness_best_vec)
        plt.pause(0.001)

    # Create best PI controller
    (Gc_num_best,Gc_den_best) = zpk2tf([best.Z1, best.Z2],[0],best.K) # PI controller, 2 parameters: location of 1 zero, value of K, (+ 1 pole always in origin)
    Gc_best = ctrl.tf(Gc_num_best,Gc_den_best)

    # Best closed loop system
    M_best = ctrl.feedback(Gc_best*Gp,1)

    print('Total number of generations: %d' % loop_n)

    # Print controller information
    print('\nController:')
    print('k: %f' % best.K)
    print('zero 1: %f' % best.Z1)
    print('zero 2: %f' % best.Z2)
    print(Gc_best)

    # Print closed loop transfer function
    print('Closed loop transfer:')
    print(M_best)

    return Gc_best, M_best


if __name__ == "__main__":

    # Create Plant to be controlled
    (Gp_num,Gp_den) = zpk2tf(PLANT_ZEROS,PLANT_POLES,PLANT_K)
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
    plt.subplot(1,2,2)
    plt.plot(T, input_signal, t1, y1, t2, y2)

    # Block until the plot window is closed
    plt.show()

