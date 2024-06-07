import os

DEBUGGER = os.getenv("DEBUGGER")

def linear(x):
    if DEBUGGER=="True":
        print("\tenter identity")

    y = x

    if DEBUGGER=="True":
        print("\texit identity")
    return y

def div_linear(place_holder=None):
    """ compute the derivative of tanh using tanh in stead of x
    
    Var:
        place_holder:
            useless; only for consistency

    """
    if DEBUGGER=="True":
        print("\tenter div_identity")
        
    y = 1
    
    if DEBUGGER=="True":
        print("\texit div_identity")
    return y