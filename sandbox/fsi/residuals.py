"""This module implements all residuals used for adaptivity. Note that
time residuals need to be initialized before they are first used to
enable reuse of forms between time steps."""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from operators import *

def evaluate_space_residuals():
    "Evaluate residuals in space"

    # Define midpoint values
    U_F = 0.5 * (U_F0 + U_F1)
    U_S = 0.5 * (U_S0 + U_S1)
    U_M = 0.5 * (U_M0 + U_M1)

    # Define time derivatives
    Dt_U_F = Dt(U_F0, U_F1, U_M0, U_M1, rho_F, kn)
    Dt_U_S = (1/kn) * (U_S1 - U_S0)
    Dt_P_S = rho_S * (1/kn) * (P_S1 - P_S0)
    Dt_U_M = alpha_M * (1/kn) * (U_M1 - U_M0)

    # Define stresses
    Sigma_F = J(U_M)*dot(Sigma_F(U_F, P_F, U_M, mu_F), inv(F(U_M)).T)
    Sigma_S = Sigma_S(U_S, mu_S, lmbda_S)
    Sigma_M = Sigma_M(U_M, mu_M, lmbda_M)

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
