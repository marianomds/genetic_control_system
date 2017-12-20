#!/usr/bin/env python

# Time vector parameters
START_TIME = 0
STOP_TIME = 20
STEP_TIME = 0.1

# Plant parameters
PLANT_ZEROS = [-7 + 2j, -7 - 2j] # empty: no zeros
PLANT_POLES = [-3 + 8j, -3 - 8j, -5 + 3j, -5 - 3j, -8 + 3j, -8 - 3j, -7, -10]
PLANT_K = 100

# Input parameters
FINAL_VALUE = 1
FINAL_TIME = 5 # MUST BE: FINAL_TIME < STOP_TIME
IN_TYPE = 'STEP' # Options: 'STEP', 'RAMP', 'SIGMOID'

# Limit values for controller parameters
DAMPING_MAX = 10 # Maximum damping ratio
WN_MAX = 50 # Maximum natural frequency
K_MAX = 100000

# Metric to be optimized
OPTIMIZE = 'OV_MSE' # Options: 'OV_MSE', 'RT, 'MSE'

# Threshold values for algorithm convergence
OVERSHOOT_MSE_TH = 0.03
RISE_TIME_TH = 11
MSE_TH = 0.001

# Genetic algorithm parameters
POPULATION_SIZE_MAX = 10
POPULATION_SIZE = POPULATION_SIZE_MAX # initial population = max population
POPULATION_DECREASE = 0.3 # number of individuals to kill in each generation
MAX_GEN = 20 # maximum number of generations
CROSS_OVER_P = 0.5 # probability of crossing over
MUTATION_COEFF = .01 # minimum mutation value

