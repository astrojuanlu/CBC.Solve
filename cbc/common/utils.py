"This module provides a set of common utility functions."

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import PeriodicBC
from ufl import grad as ufl_grad

def is_periodic(bcs):
    "Check if boundary conditions are periodic"
    return all(isinstance(bc, PeriodicBC) for bc in bcs)

def missing_function(function):
    "Write an informative error message when function has not been overloaded"
    error("The function %s() has not been specified. Please provide a specification of this function.")

def grad(v):
    "Gradient operator fix for transpose in UFL definition"
    if v.rank() == 1:
        return ufl_grad(v).T
    else:
        return ufl_grad(v)
