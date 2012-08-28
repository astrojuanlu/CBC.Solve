"This module implements residuals used for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-05-04

from dolfin import *

from cbc.twist import PiolaTransform
from operators import Sigma_F as _Sigma_F
from operators import Sigma_S as _Sigma_S
from operators import Sigma_M as _Sigma_M
from operators import F, J, I

def inner_product(v, w):
    "Return inner product for mixed fluid/structure space"

    # Define cell integrals
    dx_F = dx(0)
    dx_S = dx(1)
    dx_M = dx_F

    # Extract variables
    v1_F, q1_F, s1_F, v1_S, q1_S, v1_M, q1_M = v
    v2_F, q2_F, s2_F, v2_S, q2_S, v2_M, q2_M = w

    # Inner product on subdomains, requiring ident_zeros
    m1 = (inner(v1_F, v2_F) + q1_F*q2_F + inner(s1_F, s2_F))*dx_F + \
         (inner(v1_S, v2_S) + inner(q1_S, q2_S))*dx_S + \
         (inner(v1_M, v2_M) + inner(q1_M, q2_M))*dx_M

    # Inner product on the whole domain
    m2 = (inner(v1_F, v2_F) + q1_F*q2_F + inner(s1_F, s2_F))*dx + \
         (inner(v1_S, v2_S) + inner(q1_S, q2_S))*dx + \
         (inner(v1_M, v2_M) + inner(q1_M, q2_M))*dx

    return m2

def weak_residuals(U0, U1, U, w, kn, problem):
    "Return weak residuals"

    # Extract variables
    U_F0, P_F0, U_S0, P_S0, U_M0 = U0
    U_F1, P_F1, U_S1, P_S1, U_M1 = U1
    U_F,  P_F,  U_S,  P_S,  U_M  = U
    v_F, q_F, s_F, v_S, q_S, v_M, q_M = w

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

    # MER: added these
    B       = problem.structure_body_force()
    G_0     = problem.structure_boundary_traction_extra()
    F_M     = problem.mesh_right_hand_side()

    # Define normals
    N = FacetNormal(Omega)
    N_F = N
    N_S = -N

    # Define cell integrals
    dx_F = dx(0)
    dx_S = dx(1)

    # Define facet integrals
    dS_F  = dS(0)
    dS_S  = dS(1)
    d_FSI = dS(2)

    # Define time derivatives
    dt_U_F = (1/kn) * (U_F1 - U_F0)
    dt_U_M = (1/kn) * (U_M1 - U_M0)
    
    Dt_U_F = rho_F * J(U_M) * (dt_U_F + dot(grad(U_F), dot(inv(F(U_M)), U_F - dt_U_M)))
    Dt_U_S = (1/kn) * (U_S1 - U_S0)
    Dt_P_S = rho_S * (1/kn) * (P_S1 - P_S0)
    Dt_U_M = alpha_M * (1/kn) * (U_M1 - U_M0)

    # Define stresses
    Sigma_F = PiolaTransform(_Sigma_F(U_F, P_F, U_M, mu_F), U_M)
    Sigma_S = _Sigma_S(U_S, mu_S, lmbda_S)
    Sigma_M = _Sigma_M(U_M, mu_M, lmbda_M)

    # Fluid residual
    Id = I(U_F)
    
    R_F = inner(v_F, Dt_U_F)*dx_F + inner(grad(v_F), Sigma_F)*dx_F \
        - inner(v_F, mu_F*J(U_M)*dot(dot(inv(F(U_M)).T, grad(U_F).T), dot(inv(F(U_M)).T, N_F)))*ds \
        + inner(v_F, J(U_M)*P_F*dot(Id, dot(inv(F(U_M)).T, N_F)))*ds \
        + inner(q_F, div(J(U_M)*dot(inv(F(U_M)), U_F)))*dx_F

    # Structure residual
    # MER: What about the body force? (Added, makes a difference, good)
    # MER And the extra stress? (Added, sign dubious, b/c evaluates to
    # zero since it is orthogonal to test case dual solution)
    R_S = inner(v_S, Dt_P_S)*dx_S + inner(grad(v_S), Sigma_S)*dx_S \
        - inner(v_S('-'), dot(Sigma_F('+'), N_S('+')))*d_FSI \
        + inner(q_S, Dt_U_S - P_S)*dx_S \
        - inner(v_S, B)*dx_S \
        + inner(v_S('-'), G_0('-'))*d_FSI \

    # Mesh residual contributions
    R_M = inner(v_M, Dt_U_M)*dx_F + inner(sym(grad(v_M)), Sigma_M)*dx_F \
        + inner(q_M, U_M - U_S)('+')*d_FSI \
        - inner(v_M, F_M)*dx_F

    return R_F, R_S, R_M

def strong_residuals(U0, U1, U, Z, EZ, w, kn, problem):
    "Return strong residuals (integrated by parts)"

    # Extract variables
    U_F0, P_F0, U_S0, P_S0, U_M0        = U0
    U_F1, P_F1, U_S1, P_S1, U_M1        = U1
    U_F,  P_F,  U_S,  P_S,  U_M         = U
    Z_F,  Y_F, X_F, Z_S, Y_S, Z_M, Y_M  = Z
    EZ_F, EY_F, EX_F ,EZ_S, EY_S, EZ_M, EY_M  = EZ

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

    # FIXME: Check sign of N_S, should it be -N?

    # Define inner products
    dx_F = dx(0)
    dx_S = dx(1)

    # Define "facet" products
    dS_F  = dS(0)
    dS_S  = dS(1)
    d_FSI = dS(2)

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
    Sigma_F = PiolaTransform(_Sigma_F(U_F, P_F, U_M, mu_F), U_M)
    Sigma_S = _Sigma_S(U_S, mu_S, lmbda_S)
    Sigma_M = _Sigma_M(U_M, mu_M, lmbda_M)

    # Fluid residual contributions
    # FIXME: Add dual Lagrange multiplier EX_F for the fluid
    R_F0 = w*inner(EZ_F - Z_F, Dt_U_F - div(Sigma_F))*dx_F
    R_F1 = avg(w)*inner(EZ_F('+') - Z_F('+'), jump(Sigma_F, N_F))*dS_F
    R_F2 = w*inner(EZ_F - Z_F, dot(Sigma_F, N_F))*ds
    R_F3 = w*inner(EY_F - Y_F, div(J(U_M)*dot(inv(F(U_M)), U_F)))*dx_F

    # Structure residual contributions (note the minus sign on N_F('+'))
    R_S0 = w*inner(EZ_S - Z_S, Dt_P_S - div(Sigma_S))*dx_S
    R_S1 = avg(w)*inner(EZ_S('-') - Z_S('-'), jump(Sigma_S, N_S))*dS_S
    R_S2 = w('-')*inner(EZ_S('-') - Z_S('-'), dot(Sigma_S('-') - Sigma_F('+'), -N_F('+')))*d_FSI
    R_S3 = w*inner(EY_S - Y_S, Dt_U_S - P_S)*dx_S

    # Mesh residual contributions
    R_M0 = w*inner(EZ_M - Z_M, Dt_U_M - div(Sigma_M))*dx_F
    R_M1 = avg(w)*inner(EZ_M('+') - Z_M('+'), jump(Sigma_M, N_F))*dS_F
    R_M2 = w('+')*inner(EY_M - Y_M, U_M - U_S)('+')*d_FSI # this should be zero

    return (R_F0, R_F1, R_F2, R_F3), (R_S0, R_S1, R_S2, R_S3), (R_M0, R_M1, R_M2)
