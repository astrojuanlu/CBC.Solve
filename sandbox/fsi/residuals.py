"This module implements residuals used for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-13

from dolfin import *
from operators import Sigma_F as _Sigma_F
from operators import Sigma_S as _Sigma_S
from operators import Sigma_M as _Sigma_M
from operators import F, J

def weak_residuals():
    "Return weak residuals"
    return None

def strong_residuals(U_F0, P_F0, U_S0, P_S0, U_M0,
                     U_F1, P_F1, U_S1, P_S1, U_M1,
                     Z_F,  Y_F,  Z_S,  Y_S,  Z_M,  Y_M,
                     ZZ_F, YY_F, ZZ_S, YY_S, ZZ_M, YY_M,
                     w, kn, problem):
    "Return strong residuals (integrated by parts)"

    # Get problem parameters
    Omega   = problem.mesh()
    Omega_F = problem.fluid_mesh()
    Omega_S = problem.structure_mesh()
    rho_F   = problem.fluid_density()
    mu_F    = problem.fluid_viscosity()
    rho_S   = problem.structure_density()
    mu_S    = problem.structure_mu()
    lmbda_S = problem.structure_lmbda()
    alpha_M = problem.mesh_alpha()
    mu_M    = problem.mesh_mu()
    lmbda_M = problem.mesh_lmbda()

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

    # Fluid residual contributions
    R_F0 = w*inner(ZZ_F - Z_F, Dt_U_F - div(Sigma_F))*dx
    R_F1 = avg(w)*inner(ZZ_F('+') - Z_F('+'), jump(dot(Sigma_F, N_F)))*dS
    R_F2 = w*inner(YY_F - Y_F, div(J(U_M)*dot(inv(F(U_M)), U_F)))*dx

    # Structure residual contributions
    R_S0 = w*inner(ZZ_S - Z_S, Dt_P_S - div(Sigma_S))*dx
    R_S1 = avg(w)*inner(ZZ_S('+') - Z_S('+'), jump(dot(Sigma_S, N_S)))*dS
    R_S2 = avg(w)*inner(ZZ_S - Z_S, dot(Sigma_S - Sigma_F, N_S))('+')*dS(1)
    R_S3 = w*inner(YY_S - Y_S, Dt_U_S - P_S)*dx

    # Mesh residual contributions
    R_M0 = w*inner(ZZ_M - Z_M, Dt_U_M - div(Sigma_M))*dx
    R_M1 = avg(w)*inner(ZZ_M('+') - Z_M('+'), jump(dot(Sigma_M, N_F)))*dS
    R_M2 = avg(w)*inner(YY_M - Y_M, U_M - U_S)('+')*dS(1)

    return (R_F0, R_F1, R_F2), (R_S0, R_S1, R_S2, R_S3), (R_M0, R_M1, R_M2)
