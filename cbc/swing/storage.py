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

# Last changed: 2012-05-03

from numpy import append
from dolfin import *

def create_primal_series(parameters):
    "Create time series for primal solution"
    info("Creating primal time series.")
    if parameters["global_storage"]:
        u_F = TimeSeries("bin/u_F")
        p_F = TimeSeries("bin/p_F")
        U_S = TimeSeries("bin/U_S")
        P_S = TimeSeries("bin/P_S")
        U_M = TimeSeries("bin/U_M")
    else:
        u_F = TimeSeries("%s/bin/u_F" % parameters["output_directory"])
        p_F = TimeSeries("%s/bin/p_F" % parameters["output_directory"])
        U_S = TimeSeries("%s/bin/U_S" % parameters["output_directory"])
        P_S = TimeSeries("%s/bin/P_S" % parameters["output_directory"])
        U_M = TimeSeries("%s/bin/U_M" % parameters["output_directory"])

    return (u_F, p_F, U_S, P_S, U_M)

def create_dual_series(parameters):
    "Create time series for dual solution"
    info("Creating dual time series.")
    if parameters["global_storage"]:
        return TimeSeries("bin/Z")
    else:
        return TimeSeries("%s/bin/Z" % parameters["output_directory"])

def get_globaldof_mappings(Omega,Omega_F,Omega_S, parameters):
    """Get submesh to globalmesh mappings"""
    # Get mappings from local meshes to global mesh
    v_F = Omega_F.data().mesh_function("parent_vertex_indices").array()
    v_S = Omega_S.data().mesh_function("parent_vertex_indices").array()
    e_F = Omega_F.data().mesh_function("parent_edge_indices").array()
    e_S = Omega_S.data().mesh_function("parent_edge_indices").array()

    # Get the number of vertices and edges
    Nv = Omega.num_vertices()
    Ne = Omega.num_edges()

    # Compute mapping to global dofs
    global_dofs_U_F = append(append(v_F, Nv + e_F), append((Nv + Ne) + v_F, (Nv + Ne + Nv) + e_F))
    global_dofs_P_F = v_F
    if parameters["structure_element_degree"] == 1:
        global_dofs_U_S = append(v_S, Nv + v_S)
        global_dofs_P_S = global_dofs_U_S
    else:
        global_dofs_U_S = append(append(v_S, Nv + e_S), append((Nv + Ne) + v_S, (Nv + Ne + Nv) + e_S))
        global_dofs_P_S = global_dofs_U_S
    global_dofs_U_M = append(v_F, Nv + v_F)
    return (global_dofs_U_F, global_dofs_P_F,global_dofs_U_S,global_dofs_P_S,global_dofs_U_M)    

def read_primal_data(U, t, Omega, Omega_F, Omega_S, series, parameters):
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
    # Get mappings
    (global_dofs_U_F, global_dofs_P_F,global_dofs_U_S,global_dofs_P_S,global_dofs_U_M) = \
                      get_globaldof_mappings(Omega,Omega_F,Omega_S, parameters)
    
    # Set degrees of freedom for primal functions
    U_F.vector()[global_dofs_U_F] = local_vals_u_F
    P_F.vector()[global_dofs_P_F] = local_vals_p_F
    U_S.vector()[global_dofs_U_S] = local_vals_U_S
    P_S.vector()[global_dofs_P_S] = local_vals_P_S
    U_M.vector()[global_dofs_U_M] = local_vals_U_M

def read_dual_data(Z, t, series):
    "Read dual solution at given time"
    info("Reading dual data at t = %g" % t)
    series.retrieve(Z.vector(), t, False)

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
    [series[i].store(U[i].vector(), t) for i in range(5)]

def write_dual_data(Z, t, series):
    "Write dual solution at given time"
    series.store(Z.vector(), t)
