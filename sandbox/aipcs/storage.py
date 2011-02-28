"""This module implements storing/retrieving primal and dual solutions
to and from file. It is used by the primal solver to store solutions,
by the dual solver to read the primal solution, and in the computation
of error indicators to read both the primal and dual solutions.

Note that the primal velocity is stored as a P2 function but always
returned as a down-sampled P1 function, while the dual velocity is
always a P2 function.
"""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-28

from numpy import append
from dolfin import *

def create_primal_series(parameters):
    "Create time series for primal solution"
    info("Creating primal time series.")
    if parameters["global_storage"]:
        u_F = TimeSeries("bin/u_F")
        p_F = TimeSeries("bin/p_F")
    else:
        u_F = TimeSeries("%s/bin/u_F" % parameters["output_directory"])
        p_F = TimeSeries("%s/bin/p_F" % parameters["output_directory"])

    return (u_F, p_F)

def create_dual_series(parameters):
    "Create time series for dual solution"
    info("Creating dual time series.")
    if parameters["global_storage"]:
        return TimeSeries("bin/Z")
    else:
        return TimeSeries("%s/bin/Z" % parameters["output_directory"])

def read_primal_data(U, t, Omega, series, parameters):
    "Read primal variables at given time"

    info("Reading primal data at t = %g" % t)

    # Get primal variables
    U_F, P_F = U

    # Create vectors for primal dof values on local meshes
    local_vals_u_F = Vector()
    local_vals_p_F = Vector()

    # Retrieve primal data
    series[0].retrieve(U_F.vector(), t)
    series[1].retrieve(P_F.vector(), t)

def read_dual_data(Z, t, series):
    "Read dual solution at given time"
    info("Reading dual data at t = %g" % t)
    series.retrieve(Z.vector(), t)

def read_timestep_range(T, series):
    "Read time step range"

    # Get nodal points for primal time series
    t = series[0].vector_times()

    # Check that time series is not empty and that it covers the interval
    if len(t) == 0:
        error("Missing primal data (empty).")
    elif t[0] > DOLFIN_EPS:
        error("Illegal initial value %.16e for primal data, expecting 0.0." % t[0])
    elif (t[-1] - T) / T < -100.0 * DOLFIN_EPS:
        error("Illegal final time %.16e for primal data, expecting (at least) %.16e" % (t[-1], T))

    return t

def write_primal_data(U, t, series):
    "Write primal data at given time"
    [series[i].store(U[i].vector(), t) for i in range(2)]

def write_dual_data(Z, t, series):
    "Write dual solution at given time"
    series.store(Z.vector(), t)
