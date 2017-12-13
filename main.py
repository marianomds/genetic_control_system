#!/usr/bin/env python

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zpk2tf 
from copy import deepcopy
from math import log10
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# Time vector parameters
START_TIME = 0
STOP_TIME = 20
STEP_TIME = 0.01

# Plant parameters
PLANT_ZEROS = [] # empty: no zeros
PLANT_POLES = [-1, -5 + 3j, -5 - 3j, -10]
PLANT_K = 1

# Input parameters
FINAL_VALUE = 1
FINAL_TIME = 5 # MUST BE: FINAL_TIME < STOP_TIME
IN_TYPE = 'STEP' # Options: 'STEP', 'RAMP', 'SIGMOID'

# Limit values for controller parameters
ZERO_MIN = -30
ZERO_MAX = 0
K_MAX = 1000

# Metric to be optimized
OPTIMIZE = 'MSE' # Options: 'OV', 'RT, 'MSE'

# Threshold values for algorithm convergence
OVERSHOOT_TH = 3 # in percentage
RISE_TIME_TH = 11
MSE_TH = 0.00001

# Genetic algorithm parameters
POPULATION_SIZE = 20
POPULATION_DECREASE = 1 # number of individuals to kill in each generation
MAX_GEN = 20 # maximum number of generations
CROSS_OVER_P = 0.5 # probability of crossing over
MUTATION_COEFF = .1 # mutation value

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
    if signal.max() <= FINAL_VALUE:
        return 100
    return (signal.max() - FINAL_VALUE)*100 # in percent

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

def mutation(population,fitness_th):

    # Duplicate first (best) individual. This new one will not be mutated.
    X = deepcopy(population[0])
    population.insert(0,X)

    mutation_val_vec = [0]

    for ind in range(1,len(population)):

        # Adaptive mutation: as long as the individual's fitness gets closer the the threshold, the mutation value decreases
        mutation_val_vec.append(-log10( fitness_th/population[ind].fitness )/10 + MUTATION_COEFF)

        mutation_val = mutation_val_vec[ind]

        population[ind].Z1 *= (1 + np.random.uniform(-mutation_val,mutation_val) )
        if population[ind].Z1 < ZERO_MIN:
            population[ind].Z1 = ZERO_MIN
        elif population[ind].Z1 > ZERO_MAX:
            population[ind].Z1 = ZERO_MAX
        population[ind].Z2 *= (1 + np.random.uniform(-mutation_val,mutation_val) )
        if population[ind].Z2 < ZERO_MIN:
            population[ind].Z2 = ZERO_MIN
        elif population[ind].Z2 > ZERO_MAX:
            population[ind].Z2 = ZERO_MAX
        population[ind].K *= (1 + np.random.uniform(-mutation_val,mutation_val) )
        if population[ind].K > K_MAX:
            population[ind].K = K_MAX
        if population[ind].K < 0:
            population[ind].K = 0

    mutation_average = sum(mutation_val_vec) / len(mutation_val_vec)

    return population, mutation_average

def selection(population, Gp, Time, Input):

    # Compute the fitness of all the population
    for ind in range(len(population)):
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
            self.gm, self.pm, self.Wcg, self.Wcp = ctrl.margin(self.Gc*Gp)

            # Dischard solutions with no gain margin
            if self.Wcg == None or (self.Wcp != None and self.Wcg >= self.Wcp):
                continue

            if self.gm == None: # None = inf
                self.gm = K_MAX

            # If K < gm => closed loop stable (gm > 0dB)
            self.K = np.random.uniform(0, self.gm)

            # Create PI controller for closed loop stable system
            (self.Gc_num,self.Gc_den) = zpk2tf([self.Z1, self.Z2],[0],self.K) # PID controller, 3 parameters: location of 2 zeros, value of K, (+ 1 pole always in origin)
            self.Gc = ctrl.tf(self.Gc_num,self.Gc_den)

            # Closed loop system
            self.M = ctrl.feedback(self.Gc*Gp,1)

            # Closed loop step response
            self.y, self.t, self.xout = ctrl.lsim(self.M, Input, Time)

            # Evaluate fitness
            self.fitness = evaluate(Input,self.y)

            break

    def fitness_calc(self, Gp, Time, Input):

            # Create PI controller
            (self.Gc_num,self.Gc_den) = zpk2tf([self.Z1, self.Z2],[0],self.K) # PID controller, 3 parameters: location of 2 zeros, value of K, (+ 1 pole always in origin)
            self.Gc = ctrl.tf(self.Gc_num,self.Gc_den)

            # Evaluate closed loop stability
            self.gm, self.pm, self.Wcg, self.Wcp = ctrl.margin(self.Gc*Gp)

            # Dischard solutions with no gain margin
            if self.gm == None or self.gm <= 1:
                self.fitness = 999
                return

            # Closed loop system
            self.M = ctrl.feedback(self.Gc*Gp,1)

            # Closed loop step response
            self.y, self.t, self.xout = ctrl.lsim(self.M, Input, Time)

            # Evaluate fitness
            self.fitness = evaluate(Input,self.y)


def evolution(Gp, Time, Input):

    global POPULATION_SIZE

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
    fitness_best_vec = np.array([999])
    fitness_ave_vec = np.array([999])
    mutation_ave_vec = np.array([0])

    # Create random population
    for count in range(POPULATION_SIZE):
        x = individual(Gp, Time, Input)
        population.append(x)

    # Number of loops
    loop_n = 0

    # Keep entering while loop until fitness threshold is reached
    while (loop_n < MAX_GEN and population[0].fitness > fitness_th): # population[0] is the best individual (since population is sorted in the selection function)

        # Generation count
        loop_n += 1

        # Generate new generation
        population = cross_over(population)
        population, mutation_average = mutation(population,fitness_th)
        population = selection(population, Gp, Time, Input)

        fitness_ave = 0
        for ind in range(0,POPULATION_SIZE):
            fitness_ave += population[ind].fitness
        fitness_ave /= POPULATION_SIZE

        fitness_ave_vec = np.append(fitness_ave_vec, fitness_ave)

        # Save history of best fitness in a vector for plotting
        if population[0].fitness > fitness_best_vec[-1]: # New best individual should not be worse than previous
            print('Evolution problem!')
            quit()

        fitness_best_vec = np.append(fitness_best_vec, population[0].fitness)

        mutation_ave_vec = np.append(mutation_ave_vec, mutation_average)

        # Real time plot of history of best fitness
        plt.subplot(1,2,1)
        plt.plot(fitness_best_vec[1:])
        plt.plot(fitness_ave_vec[1:])
        plt.plot(mutation_ave_vec[1:])
        plt.pause(0.001)

        # Recalculate population size
        POPULATION_SIZE -= POPULATION_DECREASE

    # Create best PI controller
    (Gc_num_best,Gc_den_best) = zpk2tf([population[0].Z1, population[0].Z2],[0],population[0].K) # PI controller, 2 parameters: location of 1 zero, value of K, (+ 1 pole always in origin)
    Gc_best = ctrl.tf(Gc_num_best,Gc_den_best)

    # Best closed loop system
    M_best = ctrl.feedback(Gc_best*Gp,1)

    print('\nTotal number of generations: %d' % loop_n)

    # Print controller information
    print('\nController:\n')
    print('k: %f' % population[0].K)
    print('zero 1: %f' % population[0].Z1)
    print('zero 2: %f' % population[0].Z2)
    print(Gc_best)

    # Print closed loop transfer function
    print('Closed loop transfer:')
    print(M_best)

    return Gc_best, M_best


if __name__ == "__main__":

    if MAX_GEN > POPULATION_SIZE/POPULATION_DECREASE:
        print('MAX_GEN should be less or equal than POPULATION_SIZE/POPULATION_DECREASE')
        quit()

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

