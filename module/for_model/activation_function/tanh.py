import os
import numpy as np

DEBUGGER = os.getenv("DEBUGGER")

def tanh(x):
    if DEBUGGER=="True":
        print("\tenter tanh")

    exponient = np.e**(2*x)
    y = (exponient-1)/(exponient+1)

    if DEBUGGER=="True":
        print("\texit tanh")
    return y

def div_tanh(x_tanh):
    """ compute the derivative of tanh using tanh in stead of x
    """
    if DEBUGGER=="True":
        print("\tenter div_tanh")
        
    y = 1 - x_tanh**2
    
    if DEBUGGER=="True":
        print("\texit div_tanh")
    return y
