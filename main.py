#!/usr/bin/env python

import control as ctrl
from pzmap2 import pzmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zpk2tf 
from copy import deepcopy
from math import log10
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from sqlobject import *
from defs import *

# Declare connection to database
connection = connectionForURI("mysql://root:root@localhost/GeneticPID")
sqlhub.processConnection = connection

def sigmoid (x):
    return 1/(1 + np.exp(-x))

# Function for creating input signal to the system
def create_input():
    if IN_TYPE == 'STEP':
        input_signal = np.zeros(int(FINAL_TIME/STEP_TIME))
    elif IN_TYPE == 'RAMP':
        input_signal = np.arange(START_TIME, FINAL_TIME, STEP_TIME)*(FINAL_VALUE/FINAL_TIME)
    elif IN_TYPE == 'SIGMOID':
        x_sigmoid = np.arange(START_TIME, FINAL_TIME, STEP_TIME)
        input_signal = sigmoid(x_sigmoid*(12/FINAL_TIME) - 6) * FINAL_VALUE
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
    if OPTIMIZE == 'OV_MSE':
        return (overshoot(y) + 100*mse(x, y))
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

            code = np.round(np.random.random(5))
            while(0 == code.sum() or 5 == code.sum()): # If all zeros or ones in code 
                code = np.round(np.random.random(5))

            if code[0] == 1:
                Ch1.DP1 = population[P2].DP1
                Ch2.DP1 = population[P1].DP1

            if code[1] == 1:
                Ch1.WN1 = population[P2].WN1
                Ch2.WN1 = population[P1].WN1

            if code[2] == 1:
                Ch1.DP2 = population[P2].DP2
                Ch2.DP2 = population[P1].DP2

            if code[3] == 1:
                Ch1.WN2 = population[P2].WN2
                Ch2.WN2 = population[P1].WN2

            if code[4] == 1:
                Ch1.K = population[P2].K
                Ch2.K = population[P1].K

            # Append the 2 new children to the end of the population list
            population.append(Ch1)
            population.append(Ch2)

            state = 1

    return population

def mutation(population,fitness_th):

    # Duplicate best 3 individuals. These will not be mutated.
    X1 = deepcopy(population[0])
    X2 = deepcopy(population[1])
    X3 = deepcopy(population[2])
    population.insert(0,X3)
    population.insert(0,X2)
    population.insert(0,X1)

    mutation_val_vec = []

    for ind in range(3,len(population)):

        # Adaptive mutation: as long as the individual's fitness gets closer the the threshold, the mutation value decreases
        mutation_val_vec.append(-log10( fitness_th/population[ind].fitness )/10 + MUTATION_COEFF)

        mutation_val = mutation_val_vec[ind-3]

        population[ind].DP1 *= (1 + np.random.uniform(-mutation_val,mutation_val) )
        if population[ind].DP1 > DAMPING_MAX:
            population[ind].DP1 = DAMPING_MAX
        elif population[ind].DP1 < 0:
            population[ind].DP1 = 0
        population[ind].WN1 *= (1 + np.random.uniform(-mutation_val,mutation_val) )
        if population[ind].WN1 > WN_MAX:
            population[ind].WN1 = WN_MAX
        elif population[ind].WN1 < 0:
            population[ind].WN1 = 0

        population[ind].DP2 *= (1 + np.random.uniform(-mutation_val,mutation_val) )
        if population[ind].DP2 > DAMPING_MAX:
            population[ind].DP2 = DAMPING_MAX
        elif population[ind].DP2 < 0:
            population[ind].DP2 = 0
        population[ind].WN2 *= (1 + np.random.uniform(-mutation_val,mutation_val) )
        if population[ind].WN2 > WN_MAX:
            population[ind].WN2 = WN_MAX
        elif population[ind].WN2 < 0:
            population[ind].WN2 = 0

        population[ind].K *= (1 + np.random.uniform(-mutation_val,mutation_val) )
        if population[ind].K > K_MAX:
            population[ind].K = K_MAX
        if population[ind].K < 0:
            population[ind].K = 0

    mutation_average = (sum(mutation_val_vec) / len(mutation_val_vec))*100

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
            self.DP1 = np.random.uniform(0, DAMPING_MAX)
            self.WN1 = np.random.uniform(0, WN_MAX)
            self.DP2 = np.random.uniform(0, DAMPING_MAX)
            self.WN2 = np.random.uniform(0, WN_MAX)

            (self.Z1,self.Z2) = np.roots([1, 2*self.DP1*self.WN1, self.WN1**2])
            (self.Z3,self.Z4) = np.roots([1, 2*self.DP2*self.WN2, self.WN2**2])

            # Create test PI controller (used to calculate the maximum K for closed loop stable)
            (self.Gc_num,self.Gc_den) = zpk2tf([self.Z1, self.Z2, self.Z3, self.Z4],[0],1) # Controller with one pole in origin and 2 pair of zeros
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
            (self.Gc_num,self.Gc_den) = zpk2tf([self.Z1, self.Z2, self.Z3, self.Z4],[0],self.K) # Controller with one pole in origin and 2 pair of zeros
            self.Gc = ctrl.tf(self.Gc_num,self.Gc_den)

            # Closed loop system
            self.M = ctrl.feedback(self.Gc*Gp,1)

            # Closed loop step response
            self.y, self.t, self.xout = ctrl.lsim(self.M, Input, Time)

            # Evaluate fitness
            self.fitness = evaluate(Input,self.y)

            break

    def fitness_calc(self, Gp, Time, Input):

            (self.Z1,self.Z2) = np.roots([1, 2*self.DP1*self.WN1, self.WN1**2])
            (self.Z3,self.Z4) = np.roots([1, 2*self.DP2*self.WN2, self.WN2**2])

            # Create PI controller
            (self.Gc_num,self.Gc_den) = zpk2tf([self.Z1, self.Z2, self.Z3, self.Z4],[0],self.K) # Controller with one pole in origin and 2 pair of zeros
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
    if OPTIMIZE == 'OV_MSE':
        fitness_th = OVERSHOOT_MSE_TH
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
    print("Creating population: ", end = "", flush = True)
    for count in range(POPULATION_SIZE):
        print(count, end = " ", flush = True)
        x = individual(Gp, Time, Input)
        population.append(x)

    # Number of loops
    loop_n = 0

    # Open plot window
    fig = plt.figure()

    # Maximize plot window
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    # Keep entering while loop until fitness threshold is reached
    while (loop_n < MAX_GEN and population[0].fitness > fitness_th): # population[0] is the best individual (since population is sorted in the selection function)

        # Recalculate population size
        POPULATION_SIZE = round(POPULATION_SIZE_MAX - loop_n*POPULATION_DECREASE)

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
        
        plt.clf() # clear plot

        plt.subplot(2,2,3)
        pzmap(Gp, Plot=True, title='Plant P-Z Map')

        fit = fig.add_subplot(2,2,1)
        plt.title('Evolution')
        mut = fit.twinx()
        fit.set_xlabel("Generations                                                                              ")
        fit.set_ylabel("Fitness")
        mut.set_ylabel("Mutation [%]")
        p1, = fit.semilogy(fitness_best_vec[1:], color = 'k', label = 'Best fitness')
        p2, = fit.semilogy(fitness_ave_vec[1:], color = 'b', label = 'Average fitness')
        p3, = fit.plot((0, loop_n), (fitness_th, fitness_th), color = 'g', label = 'Fitness threshold')
        p4, = mut.plot(mutation_ave_vec[1:], color = 'r', label = 'Average mutation %')
        plt.ylim(0,100)
        lns = [p1, p2, p3, p4]
        fit.legend(handles=lns, loc='upper right')
        plt.pause(0.01)


    (population[0].Z1,population[0].Z2) = np.roots([1, 2*population[0].DP1*population[0].WN1, population[0].WN1**2])
    (population[0].Z3,population[0].Z4) = np.roots([1, 2*population[0].DP2*population[0].WN2, population[0].WN2**2])

    # Create best controller
    (Gc_num_best,Gc_den_best) = zpk2tf([population[0].Z1, population[0].Z2, population[0].Z3, population[0].Z4],[0],population[0].K) # Controller with one pole in origin and 2 pair of zeros
    Gc_best = ctrl.tf(Gc_num_best,Gc_den_best)

    # Best closed loop system
    M_best = ctrl.feedback(Gc_best*Gp,1)

    print('\n\nTotal number of generations: %d' % loop_n)

    # Print controller information
    print('\nCONTROLLER:\n')
    print('K: %f' % population[0].K)
    print('Damping Ratio 1: %f' % population[0].DP1)
    print('Natural Frequency 1: %f' % population[0].WN1)
    print('Damping Ratio 2: %f' % population[0].DP2)
    print('Natural Frequency 2: %f' % population[0].WN2)
    print(Gc_best)

    # Print closed loop transfer function
    print('CLOSED LOOP:')
    print(M_best)

    return Gc_best, population[0].K, population[0].DP1, population[0].WN1, population[0].DP2, population[0].WN2, M_best

class input_type(SQLObject):
    in_type = StringCol(length=10, varchar=True)

class input_signal_(SQLObject):
    final_value = FloatCol()
    final_time = FloatCol()
    input_type = ForeignKey('input_type', default=None)

class plant(SQLObject):
    k = FloatCol()
    zeros = StringCol(length=200, varchar=True) # Undefined lenght floar vector. Store it as string.
    poles = StringCol(length=200, varchar=True) # Undefined lenght floar vector. Store it as string.

class optimization(SQLObject):
    optimization_metric = StringCol(length=10, varchar=True)
    overshoot_mse_threshold = FloatCol()
    rise_time_threshold = FloatCol()
    mse_threshold = FloatCol()

class evolution_type(SQLObject):
    damping_max = FloatCol()
    wn_max = FloatCol()
    k_max = FloatCol()
    population_size = IntCol()
    population_decrease = FloatCol()
    maximum_generations = IntCol()
    cross_over_prob = FloatCol()
    mutation_min = FloatCol()
    optimization = ForeignKey('optimization', default=None)

class simulation_time(SQLObject):
    start_time = FloatCol()
    stop_time = FloatCol()
    step_time = FloatCol()

class pid(SQLObject):
    k = FloatCol()
    dp1 = FloatCol()
    wn1 = FloatCol()
    dp2 = FloatCol()
    wn2 = FloatCol()
    overshoot = FloatCol()
    rise_time = FloatCol()
    mse = FloatCol()
    input_signal_ = ForeignKey('input_signal_', default=None)
    plant = ForeignKey('plant', default=None)
    evolution_type = ForeignKey('evolution_type', default=None)
    simulation_time = ForeignKey('simulation_time', default=None)

def drop_create_tables():
    pid.dropTable(ifExists=True)  # First table to be deleted, since it has de foreign keys
    input_signal_.dropTable(ifExists=True)
    plant.dropTable(ifExists=True)
    evolution_type.dropTable(ifExists=True)
    simulation_time.dropTable(ifExists=True)
    input_type.dropTable(ifExists=True)
    optimization.dropTable(ifExists=True)

    optimization.createTable()
    input_type.createTable()
    simulation_time.createTable()
    evolution_type.createTable()
    plant.createTable()
    input_signal_.createTable()
    pid.createTable() # Last table to be created, since it has de foreign keys
    return

def fill_tables(k_best, dp1_best, wn1_best, dp2_best, wn2_best, ov, rt, mse):

    # Search if the current set of values exist already in the table.
    query = "SELECT id FROM input_type WHERE in_type = N'%s'" % (IN_TYPE)
    input_type_res = connection.queryAll(query)
    if len(input_type_res) == 0:
        input_type(in_type = IN_TYPE)
        input_type_res = connection.queryAll(query)
    
    # Search if the current set of values exist already in the table.
    query = "SELECT id FROM input_signal_ WHERE input_type_id = %d AND final_value = %f AND final_time = %f" % (input_type_res[0][0], FINAL_VALUE, FINAL_TIME)
    input_signal_res = connection.queryAll(query)
    if len(input_signal_res) == 0:
        input_signal_(input_type = input_type_res[0][0], final_value = FINAL_VALUE, final_time = FINAL_TIME)
        input_signal_res = connection.queryAll(query)

    # Serialize Plant zeros and ploes lists (converting to string is gives a shorter solution than json)
    plant_zeros_serial = ''.join(str(e) for e in PLANT_ZEROS)
    plant_poles_serial = ''.join(str(e) for e in PLANT_POLES)

    # Search if the current set of values exist already in the table.
    query = "SELECT id FROM plant WHERE k = %f AND zeros = N'%s' AND poles = N'%s'" % (PLANT_K,plant_zeros_serial,plant_poles_serial)
    plant_res = connection.queryAll(query)
    if len(plant_res) == 0:
        plant(k = PLANT_K, zeros = plant_zeros_serial, poles = plant_poles_serial)
        plant_res = connection.queryAll(query)

    # Search if the current set of values exist already in the table.
    query = "SELECT id FROM optimization WHERE optimization_metric = N'%s' AND overshoot_mse_threshold = %f AND rise_time_threshold = %f AND mse_threshold = %f" % (OPTIMIZE, OVERSHOOT_MSE_TH, RISE_TIME_TH, MSE_TH)
    optimization_res = connection.queryAll(query)
    if len(optimization_res) == 0:
        optimization(optimization_metric = OPTIMIZE, overshoot_mse_threshold = OVERSHOOT_MSE_TH, rise_time_threshold = RISE_TIME_TH, mse_threshold = MSE_TH)
        optimization_res = connection.queryAll(query)

    # Search if the current set of values exist already in the table.
    query = "SELECT id FROM evolution_type WHERE optimization_id = %d AND damping_max = %f AND wn_max = %f AND k_max = %f AND  population_size = %d AND population_decrease = %f AND maximum_generations = %d AND cross_over_prob = %f AND mutation_min = %f" % (optimization_res[0][0], DAMPING_MAX, WN_MAX, K_MAX, POPULATION_SIZE_MAX, POPULATION_DECREASE, MAX_GEN, CROSS_OVER_P, MUTATION_COEFF)
    evolution_type_res = connection.queryAll(query)
    if len(evolution_type_res) == 0:
        evolution_type(optimization = optimization_res[0][0], damping_max = DAMPING_MAX, wn_max = WN_MAX, k_max = K_MAX, population_size = POPULATION_SIZE_MAX, population_decrease = POPULATION_DECREASE, maximum_generations = MAX_GEN, cross_over_prob = CROSS_OVER_P, mutation_min = MUTATION_COEFF)
        evolution_type_res = connection.queryAll(query)

    # Search if the current set of values exist already in the table.
    query = "SELECT id FROM simulation_time WHERE start_time = %f AND stop_time = %f AND step_time = %f" % (START_TIME, STOP_TIME, STEP_TIME)
    simulation_time_res = connection.queryAll(query)
    if len(simulation_time_res) == 0:
        simulation_time(start_time = START_TIME, stop_time = STOP_TIME, step_time = STEP_TIME)
        simulation_time_res = connection.queryAll(query)

    pid(k = k_best, dp1 = dp1_best, wn1 = wn1_best, dp2 = dp2_best, wn2 = wn2_best, overshoot = float(ov), rise_time = rt, mse = float(mse), input_signal_ = input_signal_res[0][0], plant = plant_res[0][0], evolution_type = evolution_type_res[0][0], simulation_time = simulation_time_res[0][0])

    return

if __name__ == "__main__":

    if POPULATION_SIZE_MAX <= (POPULATION_DECREASE*MAX_GEN + 3):
        print('POPULATION_SIZE_MAX should be bigger than POPULATION_DECREASE*MAX_GEN + 3')
        quit()

    ans = input("Delete database tables and create new ones? (Y/N)")
    if (ans == 'y') or (ans == 'Y'):
        drop_create_tables()
    elif (ans != 'n') and (ans != 'N'):
        print('Input Error')
        quit()

    # Create Plant to be controlled
    (Gp_num,Gp_den) = zpk2tf(PLANT_ZEROS,PLANT_POLES,PLANT_K)
    Gp = ctrl.tf(Gp_num,Gp_den)

    # Print transfer function of plant to be controlled
    print('\nPLANT:')
    print(Gp)

    # Create time vector to be used
    T = np.arange(START_TIME, STOP_TIME, STEP_TIME)

    # Generate input signal
    input_signal = create_input()

    # Generate plant response to input signal
    y1, t1, xout = ctrl.lsim(Gp, input_signal, T)

    # Run evolution algorithm
    Gc, K_best, DP1_best, WN1_best, DP2_best, WN2_best, M = evolution(Gp, T, input_signal)

    # Closed loop response to input signal
    y2, t2, xout = ctrl.lsim(M, input_signal, T)

    # Evaluate and print overshoot and rise time
    print('Overshoot: %f %%' % overshoot(y2))
    print('Rise time: %1.2f sec' % rise_time(y2))
    print('MSE: %f' % mse(y2, input_signal))

    # Fill database tables with current simulation data
    fill_tables(K_best, DP1_best, WN1_best, DP2_best, WN2_best, overshoot(y2), rise_time(y2), mse(y2, input_signal))

    # Plot result
    plt.subplot(2,2,2)
    plt.title('Temporal response')
    plt.xlabel("Time                                                                                      ")
    p1, = plt.plot(T, input_signal, label = 'Input')
    p2, = plt.plot(t1, y1, label = 'Plant response')
    p3, = plt.plot(t2, y2, label = 'Closed-loop response')
    lns = [p1, p2, p3]
    plt.legend(handles=lns, loc='best')
    
    plt.subplot(2,2,4)
    pzmap(Gc, Plot=True, title='Controller P-Z Map')


    # Block until the plot window is closed
    plt.show()




