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

# Last changed: 2010-09-13

from numpy import append
from dolfin import *

# Time series for primal variables
_u_F_data = None
_p_F_data = None
_U_S_data = None
_P_S_data = None
_U_M_data = None

# Time series for dual variables
_Z_data = None

def init_primal_data(Omega):
    "Return primal variables on the full domain initialized to zero"

    # Open time series for primal solution
    global _u_F_data, _p_F_data, _U_S_data, _P_S_data, _U_M_data
    _u_F_data = TimeSeries("bin/u_F")
    _p_F_data = TimeSeries("bin/p_F")
    _U_S_data = TimeSeries("bin/U_S")
    _P_S_data = TimeSeries("bin/P_S")
    _U_M_data = TimeSeries("bin/U_M")

    # Create function spaces
    V_F = VectorFunctionSpace(Omega, "CG", 1)
    Q_F = FunctionSpace(Omega, "CG", 1)
    V_S = VectorFunctionSpace(Omega, "CG", 1)
    Q_S = VectorFunctionSpace(Omega, "CG", 1)
    V_M = VectorFunctionSpace(Omega, "CG", 1)

    # Create primal functions
    U_F = Function(V_F)
    P_F = Function(Q_F)
    U_S = Function(V_S)
    P_S = Function(Q_S)
    U_M = Function(V_M)

    return U_F, P_F, U_S, P_S, U_M

def init_dual_space(Omega):
    "Return dual function space on the full domain"

    # Create function spaces
    V_F = VectorFunctionSpace(Omega, "CG", 2)
    Q_F = FunctionSpace(Omega, "CG", 1)
    V_S = VectorFunctionSpace(Omega, "CG", 1)
    Q_S = VectorFunctionSpace(Omega, "CG", 1)
    V_M = VectorFunctionSpace(Omega, "CG", 1)
    Q_M = VectorFunctionSpace(Omega, "CG", 1)

    # Create mixed function space
    W = MixedFunctionSpace([V_F, Q_F, V_S, Q_S, V_M, Q_M])

    return W

def init_dual_data(Omega):
    "Return dual variables on the full domain initialized to zero"

    # Open time series for dual solution
    global _Z_data
    _Z_data = TimeSeries("bin/Z")

    # Create dual function
    W = init_dual_space(Omega)
    Z = Function(W)

    return Z, Z.split(False)

def read_primal_data(U_F, P_F, U_S, P_S, U_M, t,
                     Omega, Omega_F, Omega_S):
    "Read primal variables at given time"

    info("Reading primal data at t = %g" % t)

    # Create vectors for primal dof values on local meshes
    local_vals_u_F = Vector()
    local_vals_p_F = Vector()
    local_vals_U_S = Vector()
    local_vals_P_S = Vector()
    local_vals_U_M = Vector()

    # Retrieve primal data
    global _u_F_data, _p_F_data, _U_S_data, _P_S_data, _U_M_data
    _u_F_data.retrieve(local_vals_u_F, t)
    _p_F_data.retrieve(local_vals_p_F, t)
    _U_S_data.retrieve(local_vals_U_S, t)
    _P_S_data.retrieve(local_vals_P_S, t)
    _U_M_data.retrieve(local_vals_U_M, t)

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

def read_dual_data(Z, t):
    "Read dual solution at given time"

    info("Reading dual data at t = %g" % t)

    # Retrieve dual data
    global _Z_data
    _Z_data.retrieve(Z.vector(), t)

def read_timestep_range(problem):
    "Read time step range"

    # Get nodal points for primal time series
    global _u_F_data
    t = _u_F_data.vector_times()

    # Check that time series is not empty and covers the interval
    T = problem.end_time()
    if not (len(t) > 1 and t[0] == 0.0 and t[-1] == T):
        print "Nodal points for primal time series:", t
        raise RuntimeError, "Missing primal data"

    return t

def write_primal_data(u_F, p_F, U_S, P_S, U_M, t, parameters):
    "Write primal data at given time"

    # Check if we should store solution
    if not parameters["save_series"]: return

    # Check if we should initialize the series
    global _u_F_data, _p_F_data, _U_S_data, _P_S_data, _U_M_data

    # Save to series
    _u_F_data.store(u_F.vector(), t)
    _p_F_data.store(p_F.vector(), t)
    _U_S_data.store(U_S.vector(), t)
    _P_S_data.store(P_S.vector(), t)
    _U_M_data.store(U_M.vector(), t)

def write_dual_data(Z, t, parameters):
    "Write dual solution at given time"

    # Check if we should store solution
    if not parameters["save_series"]: return

    # Check if we should initialize the series
    global _Z_data
    if _Z_data is None:
        _Z_data = TimeSeries("bin/Z")

    # Save to series
    _Z_data.store(Z.vector(), t)
