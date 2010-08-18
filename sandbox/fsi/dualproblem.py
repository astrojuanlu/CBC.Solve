"This module specifies the variational forms for the dual FSI problem."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-18

from dolfin import *
from operators import *

def dual_forms(Omega_F, Omega_S, k, problem,
               v_F,  q_F,  v_S,  q_S,  v_M,  q_M,
               Z_F,  Y_F,  Z_S,  Y_S,  Z_M,  Y_M,
               Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0,
               U_F0, P_F0, U_S0, P_S0, U_M0,
               U_F1, P_F1, U_S1, P_S1, U_M1):
    "Return bilinear and linear forms for one time step."

    info("Creating dual forms")

    # Get problem parameters
    rho_F   = problem.fluid_density()
    mu_F    = problem.fluid_viscosity()
    rho_S   = problem.structure_density()
    mu_S    = problem.structure_mu()
    lmbda_S = problem.structure_lmbda()
    mu_M    = problem.mesh_mu()
    lmbda_M = problem.mesh_lmbda()

    # Define normals
    N_S = FacetNormal(Omega_S)
    N   = N_S('+')
    N_F = FacetNormal(Omega_F)

    # Define identity matrix (2D)
    I = Identity(2)

    # Operators for A_SS 
    Fu = F(U_S1)
    Eu = Fu*Fu.T - I
    Ev = grad(v_S)*Fu.T + Fu*grad(v_S).T
    Sv = grad(v_S)*(2*mu_S*Eu + lmbda_S*tr(Eu)*I) + Fu*(2*mu_S*Ev + lmbda_S*tr(Ev)*I)

    #====DUAL FORMS=================================================================================================================================

    A_FF01 = -(1/k)*inner((Z_F0 - Z_F), rho_F*J(U_M1)*v_F)*dx(0)
    A_FF02 =  inner(Z_F, rho_F*J(U_M1)*dot(dot(grad(v_F),inv(F(U_M1))), (U_F1 - (U_M0 - U_M1)*(1/k))))*dx(0)
    A_FF03 =  inner(Z_F, rho_F*J(U_M1)*dot(grad(U_F1) , dot(inv(F(U_M1)), v_F)))*dx(0)
    A_FF04 =  inner(grad(Z_F), J(U_M1)*mu_F*dot(grad(v_F) , dot(inv(F(U_M1)), inv(F(U_M1)).T)))*dx(0)
    A_FF05 =  inner(grad(Z_F), J(U_M1)*mu_F*dot(inv(F(U_M1)).T, dot(grad(v_F).T, inv(F(U_M1)).T)))*dx(0)
    A_FF06 = -inner(grad(Z_F), J(U_M1)*q_F*inv(F(U_M1)).T)*dx(0)
    A_FF07 =  inner(Y_F, div(J(U_M1)*dot(inv(F(U_M1)),v_F)))*dx(0)

    G_FF   = -inner(Z_F, dot(J(U_M1)*mu_F*dot(inv(F(U_M1)).T , dot(grad(v_F).T, inv(F(U_M1)).T)), N_F))*ds
    
    A_SF01 = -inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(grad(v_F('+')), inv(F(U_M1))('+')), dot(inv(F(U_M1)).T('+'), N)))*dS(1)    
    A_SF02 = -inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(inv(F(U_M1)).T('+'), grad(v_F('+')).T), dot(inv(F(U_M1)).T('+'), N)))*dS(1)
    A_SF03 =  inner(Z_S('+'), J(U_M1)('+')*q_F('+')*dot(I('+'), dot(inv(F(U_M1)).T('+'), N)))*dS(1) 

    A_SS = - (1/k)*inner(Z_S0 - Z_S, rho_S*q_S)*dx(1) + inner(grad(Z_S), Sv)*dx(1) \
           - (1/k)*inner(Y_S0 - Y_S, v_S)*dx(1) - inner(Y_S, q_S)*dx(1)

    A_MS = - inner(Y_M('+'), q_S('+'))*dS(1)

    A_FM01 =  (1/k)*inner(Z_F, rho_F*DJ(U_M1, v_M)*(U_F0 - U_F1))*dx(0)
    A_FM02 =  inner(Z_F, rho_F*DJ(U_M1, v_M)*dot(grad(U_F1), dot(inv(F(U_M1)), (U_M1 - U_M0)*(1/k))))*dx(0)
    A_FM03 = -inner(Z_F,  rho_F*J(U_M1)*dot((dot(grad(U_F1), dot(inv(F(U_M1)), dot(grad(v_M),inv(F(U_M1)))))),(U_F1 - (U_M0 - U_M1)/k)))*dx(0)
    A_FM04 =  (1/k)*inner((Z_F0 - Z_F), rho_F*J(U_M1)*dot(grad(U_F1), dot(inv(F(U_M1)), v_M )))*dx(0)
    A_FM05 =  inner(grad(Z_F), DJ(U_M1, v_M)*dot(Sigma_F(U_F1, P_F1, U_M1, mu_F), inv(F(U_M1)).T))*dx(0)
    A_FM06 = -inner(grad(Z_F), J(U_M1)*dot(mu_F*(dot(grad(U_F1), dot(inv(F(U_M1)), dot(grad(v_M), inv(F(U_M1)))))), inv(F(U_M1)).T))*dx(0)
    A_FM07 = -inner(grad(Z_F), J(U_M1)*dot(mu_F*(dot(inv(F(U_M1)).T, dot(grad(v_M).T, dot(inv(F(U_M1)).T, grad(U_F1).T )))), inv(F(U_M1)).T))*dx(0)
    A_FM08 = -inner(grad(Z_F), J(U_M1)*dot(mu_F*(dot(grad(U_F1), dot(inv(F(U_M1)), dot(inv(F(U_M1)).T, grad(v_M).T )))), inv(F(U_M1)).T))*dx(0)
    A_FM09 = -inner(grad(Z_F), J(U_M1)*dot(mu_F*(dot(inv(F(U_M1)).T, dot(grad(U_F1).T, dot(inv(F(U_M1)).T, grad(v_M).T )))), inv(F(U_M1)).T))*dx(0)
    A_FM10 =  inner(grad(Z_F), J(U_M1)*dot(dot( P_F1*I,inv(F(U_M1)).T),  dot(grad(v_M).T, inv(F(U_M1)).T)))*dx(0)
    A_FM11 =  inner(Y_F, div(DJ(U_M1,v_M)*dot(inv(F(U_M1)), U_F1)))*dx(0)
    A_FM12 = -inner(Y_F, div(J(U_M1)*dot(dot(inv(F(U_M1)), grad(v_M)), dot(inv(F(U_M1)), U_F1))))*dx(0)

    G_FM1 = -inner(Z_F, DJ(U_M1, v_M)*mu_F*dot(dot(inv(F(U_M1)).T,grad(U_F1).T), dot(inv(F(U_M1)).T, N_F)))*ds
    G_FM2 =  inner(Z_F, J(U_M1)*mu_F*dot(dot(inv(F(U_M1)).T, dot(grad(v_M).T, inv(F(U_M1)).T)), dot(grad(U_F1).T, dot(inv(F(U_M1)).T, N_F ))))*ds
    G_FM3 =  inner(Z_F, J(U_M1)*mu_F*dot(dot(inv(F(U_M1)).T, dot(grad(U_F1).T, inv(F(U_M1)).T)), dot(grad(v_M).T , dot(inv(F(U_M1)).T, N_F))))*ds

    A_SM01 = -inner(Z_S('+'), tr(dot(grad(v_M('+')),inv(F(U_M1)('+'))))*dot(dot(Sigma_F(U_F1, P_F1, U_M1, mu_F)('+'),inv(F(U_M1)('+')).T), N))*dS(1)
    A_SM02 =  inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(grad(U_F1('+')), dot(inv(F(U_M1))('+'),grad(v_M('+')))), dot(inv(F(U_M1))('+'), dot(inv(F(U_M1)).T('+'), N))))*dS(1)
    A_SM03 =  inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(grad(U_F1('+')).T, dot(inv(F(U_M1)).T('+'), grad(v_M('+')).T)), dot(inv(F(U_M1)).T('+'), dot(inv(F(U_M1)).T('+'),N))))*dS(1)
    A_SM04 =  inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(grad(U_F1('+')), inv(F(U_M1))('+')),dot(inv(F(U_M1)).T('+'), dot(grad(v_M('+')).T, dot(inv(F(U_M1)).T('+'),N)))))*dS(1)
    A_SM05 =  inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(inv(F(U_M1)).T('+'), grad(U_M1('+')).T),dot(inv(F(U_M1)).T('+'), dot(grad(v_M('+')).T, dot(inv(F(U_M1)).T('+'),N)))))*dS(1)
    A_SM06 = -inner(Z_S('+'), J(U_M1)('+')*dot(dot(P_F1('+')*I('+'),inv(F(U_M1)).T('+')), dot(grad(v_M('+')).T, dot(inv(F(U_M1)).T('+'), N))))*dS(1)

    A_MM01 = -(1/k)*inner(v_M, Z_M0 - Z_M)*dx(0) + inner(sym(grad(Z_M)), Sigma_M(v_M, mu_M, lmbda_M))*dx(0)
    A_MM02 = inner(Z_M('+'),v_M('+'))*dS(1)
    A_MM03 = inner(Y_M('+'),q_M('+'))*dS(1)
    
    #==================================================================================================================================================

    # Collect forms to system
    A_FF = A_FF01 + A_FF02 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07 #+ G_FF
    A_SF = A_SF01 + A_SF02 + A_SF03
    A_MM = A_MM01 + A_MM02 + A_MM03
    G_FM = G_FM1 + G_FM2 + G_FM3
    A_FM =  A_FM01 + A_FM02 + A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08 + A_FM09 + A_FM10 + A_FM11 + A_FM12 + G_FM
    A_SM = A_SM01 + A_SM02 + A_SM03 + A_SM04 + A_SM05 + A_SM06
    A_system = A_FF + A_FM + A_SS + A_SF + A_SM + A_MM + A_MS

    # FIXME: Goal functional should not be defined here
    # Define goal funtionals
    n_F = FacetNormal(Omega_F)
    area = 0.2*0.5
    T = 1.0
    goal_functional = (1/T)*(1.0/area)*v_S[0]*dx(1)

    # Define the dual rhs and lhs
    A = lhs(A_system)
    L = rhs(A_system) + goal_functional

    info("Dual forms created")

    return A, L
