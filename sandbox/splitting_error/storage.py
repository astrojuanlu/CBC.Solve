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

# Last changed: 2010-09-16

from numpy import append
from dolfin import *

def create_primal_series():
    "Create time series for primal solution"
    u_F = TimeSeries("bin/u_F")
    p_F = TimeSeries("bin/p_F")
    return (u_F, p_F)

def create_dual_series():
    "Create time series for dual solution"
    return TimeSeries("bin/Z")

# fix
def read_primal_data(U, t, Omega, Omega_F, Omega_S, series):
    "Read primal variables at given time"

    info("Reading primal data at t = %g" % t)

    # Get primal variables
    U_F, P_F, U_S, P_S, U_M = U

    # Create vectors for primal dof values on local meshes
    local_vals_u_F = Vector()
    local_vals_p_F = Vector()
    local_vals_U_S = Vector()
    local_vals_P_S = Vector()
    local_vals_U_M = Vector()

    # Retrieve primal data
    series[0].retrieve(local_vals_u_F, t)
    series[1].retrieve(local_vals_p_F, t)
    series[2].retrieve(local_vals_U_S, t)
    series[3].retrieve(local_vals_P_S, t)
    series[4].retrieve(local_vals_U_M, t)

    # Get vertex mappings from local meshes to global mesh
    vmap_F = Omega_F.data().mesh_function("global vertex indices").values()
    vmap_S = Omega_S.data().mesh_function("global vertex indices").values()

    # Get the number of vertices and edges
    Omega_F.init(1)
    Nv   = Omega.num_vertices()
    Nv_F = Omega_F.num_vertices()
    Ne_F = Omega_F.num_edges()

    # Compute mapping to global dofs
    global_dofs_U_F = append(vmap_F, vmap_F + Nv)
    global_dofs_P_F = vmap_F
    global_dofs_U_S = append(vmap_S, vmap_S + Nv)
    global_dofs_P_S = append(vmap_S, vmap_S + Nv)
    global_dofs_U_M = append(vmap_F, vmap_F + Nv)

    # Get rid of P2 dofs for u_F and create a P1 function
    local_vals_u_F = append(local_vals_u_F[:Nv_F], local_vals_u_F[Nv_F + Ne_F: 2*Nv_F + Ne_F])

    # Set degrees of freedom for primal functions
    U_F.vector()[global_dofs_U_F] = local_vals_u_F
    P_F.vector()[global_dofs_P_F] = local_vals_p_F
    U_S.vector()[global_dofs_U_S] = local_vals_U_S
    P_S.vector()[global_dofs_P_S] = local_vals_P_S
    U_M.vector()[global_dofs_U_M] = local_vals_U_M

def read_dual_data(Z, t, series):
    "Read dual solution at given time"
    info("Reading dual data at t = %g" % t)
    series.retrieve(Z.vector(), t)

def read_timestep_range(T, series):
    "Read time step range"

    # Get nodal points for primal time series
    t = series[0].vector_times()

    # Check that time series is not empty and covers the interval
    if not (len(t) > 1 and t[0] == 0.0 and t[-1] == T):
        print "Nodal points for primal time series:", t
        raise RuntimeError, "Missing primal data"

    return t

def write_primal_data(U, t, series):
    "Write primal data at given time"
    [series[i].store(U[i].vector(), t) for i in range(2)]

def write_dual_data(Z, t, series):
    "Write dual solution at given time"
    series.store(Z.vector(), t)
