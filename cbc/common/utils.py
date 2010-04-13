"This module provides a set of common utility functions."

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from math import ceil
from numpy import linspace
from dolfin import PeriodicBC, warning

def is_periodic(bcs):
    "Check if boundary conditions are periodic"
    return all(isinstance(bc, PeriodicBC) for bc in bcs)

def missing_function(function):
    "Write an informative error message when function has not been overloaded"
    error("The function %s() has not been specified. Please provide a specification of this function.")

def timestep_range(problem, mesh):
    """Return a sensible default time step and time step range based
    on an approximate CFL condition."""
    
    # Get problem parameters
    T = problem.end_time()
    ds = problem.time_step()
        
    # Set time step based on mesh if not specified
    if ds is None:
        ds = 0.25*mesh.hmin()

    # Compute range
    n = ceil(T / ds)
    t_range = linspace(0, T, n + 1)[1:]
    dt = t_range[0]

    # Warn about changing time step
    if ds != dt:
        warning("Changing time step from %g to %g" % (ds, dt))

    return dt, t_range
