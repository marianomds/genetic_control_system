#!/usr/bin/env python

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zpk2tf 

# Create time vector to be used
start_time = 0
stop_time = 5
step_time = 0.01
T = np.arange(start_time, stop_time, step_time)

# Create Plant to be controlled
(Gp_num,Gp_den) = zpk2tf([],[-1, -2, -8],1)
Gp = ctrl.tf(Gp_num,Gp_den)

#Create system for testing
(Gc_num,Gc_den) = zpk2tf([-1, -2],[0],45.9)
Gc = ctrl.tf(Gc_num,Gc_den)
M = ctrl.feedback(Gc*Gp,1)


if __name__ == "__main__":

    # Print transfer function of Plant to be controlled:
    print('Plant:')
    print(Gp)

    # Print transfer function of controller:
    print('Controller:')
    print(Gc)

    # Print closed loop transfer function:
    print('Closed loop transfer:')
    print(M)

    # Generate step response
    (y1, T1) = ctrl.step(ctrl.tf([1, 0],[1, 0]), T) # with ([1],[1]) step throws an error
    (y2, T2) = ctrl.step(Gp, T)
    (y3, T3) = ctrl.step(M, T)

    print('Overshoot: %f %%' % ((y3.max() - y3[-1])*100))

    rt = next(x[0] for x in enumerate(y3) if x[1] > y3[-1])
    print('Rise time: %1.2f sec' % (rt*step_time))

    # Plot result
    plt.plot(T1, y1, T2, y2, T3, y3)
    plt.show()