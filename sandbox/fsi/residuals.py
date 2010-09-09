"""This module implements all residuals used for adaptivity. Note that
time residuals need to be initialized before they are first used to
enable reuse of forms between time steps."""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from operators import Sigma_F as _Sigma_F
from operators import Sigma_S as _Sigma_S
from operators import Sigma_M as _Sigma_M
from operators import F, J
from storage import *

def evaluate_space_residuals(problem):
    "Evaluate residuals in space"

    # Get problem parameters
    Omega   = problem.mesh()
    rho_F   = problem.fluid_density()
    mu_F    = problem.fluid_viscosity()
    rho_S   = problem.structure_density()
    mu_S    = problem.structure_mu()
    lmbda_S = problem.structure_lmbda()
    alpha_M = problem.mesh_alpha()
    mu_M    = problem.mesh_mu()
    lmbda_M = problem.mesh_lmbda()

    # Define projection spaces
    V = VectorFunctionSpace(Omega, "DG", 0)
    Q = FunctionSpace(Omega, "DG", 0)

    # Define test/trial functions
    v = TestFunction(V)
    q = TestFunction(Q)

    # Initialize primal functions
    U_F0, P_F0, U_S0, P_S0, U_M0 = init_primal_data(Omega)
    U_F1, P_F1, U_S1, P_S1, U_M1 = init_primal_data(Omega)

    # Initialize dual functions
    Z0, (Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0) = init_dual_data(Omega)
    Z1, (Z_F1, Y_F1, Z_S1, Y_S1, Z_M1, Y_M1) = init_dual_data(Omega)

    # Define time step (value set in each time step)
    kn = Constant(0.0)

    # Define normals
    N = FacetNormal(Omega)
    N_F = N
    N_S = N

    # Define midpoint values
    U_F = 0.5 * (U_F0 + U_F1)
    P_F = 0.5 * (P_F0 + P_F1)
    U_S = 0.5 * (U_S0 + U_S1)
    P_S = 0.5 * (P_S0 + P_S1)
    U_M = 0.5 * (U_M0 + U_M1)

    # Define time derivatives
    dt_U_F = (1/kn) * (U_F1 - U_F0)
    dt_U_M = (1/kn) * (U_M1 - U_M0)
    Dt_U_F = rho_F * J(U_M) * (dt_U_F + dot(grad(U_F), dot(inv(F(U_M)), U_F - dt_U_M)))
    Dt_U_S = (1/kn) * (U_S1 - U_S0)
    Dt_P_S = rho_S * (1/kn) * (P_S1 - P_S0)
    Dt_U_M = alpha_M * (1/kn) * (U_M1 - U_M0)

    # Define stresses
    Sigma_F = J(U_M)*dot(_Sigma_F(U_F, P_F, U_M, mu_F), inv(F(U_M)).T)
    Sigma_S = _Sigma_S(U_S, mu_S, lmbda_S)
    Sigma_M = _Sigma_M(U_M, mu_M, lmbda_M)

    # Fluid residuals
    Rh_F1 = inner(v, Dt_U_F)*dx - inner(v, div(Sigma_F))*dx
    Rh_F2 = inner(avg(v), jump(dot(Sigma_F, N_F)))*dS
    Rh_F3 = inner(q, div(J(U_M)*dot(inv(F(U_M)), U_F)))*dx

    # Structure residuals
    Rh_S1 = inner(v, Dt_P_S)*dx - inner(v, div(Sigma_S))*dx
    Rh_S2 = inner(avg(v), jump(dot(Sigma_S, N_S)))*dS
    Rh_S3 = inner(v, dot(Sigma_S - Sigma_F, N_S))('+')*dS(1)
    Rh_S4 = inner(v, Dt_U_S - P_S)*dx

    # Mesh residuals
    Rh_M1 = inner(v, Dt_U_M)*dx - inner(v, div(Sigma_M))*dx
    Rh_M2 = inner(avg(v), jump(dot(Sigma_M, N_F)))*dS
    Rh_M4 = inner(v, U_M - U_S)('+')*dS(1)

def evaluate_time_residuals():
    "Evaluate residuals in time"

    pass

def evaluate_computational_residuals():
    "Evaluate computational residuals"

    pass

def init_time_residuals():
    "Initialize time residuals"

    pass
