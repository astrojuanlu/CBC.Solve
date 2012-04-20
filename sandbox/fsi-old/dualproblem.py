"This module specifies the variational forms for the dual FSI problem."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-04-10

from dolfin import *
from operators import *

def create_dual_forms(Omega_F, Omega_S, k, problem,
                      v_F,  q_F,  v_S,  q_S,  v_M,  q_M,
                      Z_F,  Y_F,  Z_S,  Y_S,  Z_M,  Y_M,
                      Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0,
                      U_F0, P_F0, U_S0, P_S0, U_M0,
                      U_F1, P_F1, U_S1, P_S1, U_M1):
    "Return bilinear and linear forms for one time step."

    info_blue("Creating dual forms")

    # Get problem parameters
    rho_F   = problem.fluid_density()
    mu_F    = problem.fluid_viscosity()
    rho_S   = problem.structure_density()
    mu_S    = problem.structure_mu()
    lmbda_S = problem.structure_lmbda()
    mu_M    = problem.mesh_mu()
    lmbda_M = problem.mesh_lmbda()
    alpha_M = problem.mesh_alpha()

    # Define normals
    N_S = FacetNormal(Omega_S)
    N   = N_S('+')
    N_F = FacetNormal(Omega_F)

    # Define inner products
    dx_F = dx(0)
    dx_S = dx(1)
    dx_M = dx_F
    d_FSI = dS(2)
    ds_F = ds(0)

    # Operators for A_SS
    Fu = F(U_S1)
    Eu = Fu*Fu.T - I
    Ev = grad(v_S)*Fu.T + Fu*grad(v_S).T
    Sv = grad(v_S)*(2*mu_S*Eu + lmbda_S*tr(Eu)*I) + Fu*(2*mu_S*Ev + lmbda_S*tr(Ev)*I)

	# Stabilization for boundary terms
    eps = Constant(1e-8)

    # Dual forms
    A_FF01 = -(1/k)*inner((Z_F0 - Z_F), rho_F*J(U_M1)*v_F)*dx_F
    A_FF02 =  inner(Z_F, rho_F*J(U_M1)*dot(dot(grad(v_F), inv(F(U_M1))), (U_F1 - (U_M0 - U_M1)*(1/k))))*dx_F
    A_FF03 =  inner(Z_F, rho_F*J(U_M1)*dot(grad(U_F1), dot(inv(F(U_M1)), v_F)))*dx(0)
    A_FF04 =  inner(grad(Z_F), J(U_M1)*mu_F*dot(grad(v_F), dot(inv(F(U_M1)), inv(F(U_M1)).T)))*dx_F
    A_FF05 =  inner(grad(Z_F), J(U_M1)*mu_F*dot(inv(F(U_M1)).T, dot(grad(v_F).T, inv(F(U_M1)).T)))*dx_F
    A_FF06 = -inner(grad(Z_F), J(U_M1)*q_F*inv(F(U_M1)).T)*dx_F
    A_FF07 =  inner(Y_F, div(J(U_M1)*dot(inv(F(U_M1)), v_F)))*dx_F

    G_FF   = -inner(Z_F, dot(J(U_M1)*mu_F*dot(inv(F(U_M1)).T, dot(grad(v_F).T, inv(F(U_M1)).T)), N_F))*ds_F

    A_SF01 = -inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(grad(v_F('+')), inv(F(U_M1))('+')), dot(inv(F(U_M1)).T('+'), N)))*d_FSI
    A_SF02 = -inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(inv(F(U_M1)).T('+'), grad(v_F('+')).T), dot(inv(F(U_M1)).T('+'), N)))*d_FSI
    A_SF03 =  inner(Z_S('+'), J(U_M1)('+')*q_F('+')*dot(I('+'), dot(inv(F(U_M1)).T('+'), N)))*d_FSI

    A_SS   = - (1/k)*inner(Z_S0 - Z_S, rho_S*q_S)*dx_S + inner(grad(Z_S), Sv)*dx_S \
             - (1/k)*inner(Y_S0 - Y_S, v_S)*dx_S - inner(Y_S, q_S)*dx_S

    A_MS   = - inner(Y_M('+'), v_S('+'))*d_FSI

    A_FM01 =  (1/k)*inner(Z_F, rho_F*J(U_M1)*tr(dot(grad(v_M), inv(F(U_M1))))*(U_F0 - U_F1))*dx_F
    A_FM02 =  inner(Z_F, rho_F*J(U_M1)*tr(dot(grad(v_M), inv(F(U_M1))))*dot(grad(U_F1), dot(inv(F(U_M1)), (U_M1 - U_M0)*(1/k) + U_F1)))*dx_F
    A_FM03 = -inner(Z_F,  rho_F*J(U_M1)*dot((dot(grad(U_F1), dot(inv(F(U_M1)), dot(grad(v_M), inv(F(U_M1)))))), (U_F1 - (U_M0 - U_M1)/k)))*dx_F
    A_FM04 =  (1/k)*inner((Z_F0 - Z_F), rho_F*J(U_M1)*dot(grad(U_F1), dot(inv(F(U_M1)), v_M )))*dx_F
    A_FM05 =  inner(grad(Z_F), J(U_M1)*tr(dot(grad(v_M), inv(F(U_M1))))*dot(Sigma_F(U_F1, P_F1, U_M1, mu_F), inv(F(U_M1)).T))*dx_F
    A_FM06 = -inner(grad(Z_F), J(U_M1)*dot(mu_F*(dot(grad(U_F1), dot(inv(F(U_M1)), dot(grad(v_M), inv(F(U_M1)))))), inv(F(U_M1)).T))*dx_F
    A_FM07 = -inner(grad(Z_F), J(U_M1)*dot(mu_F*(dot(inv(F(U_M1)).T, dot(grad(v_M).T, dot(inv(F(U_M1)).T, grad(U_F1).T )))), inv(F(U_M1)).T))*dx_F
    A_FM08 = -inner(grad(Z_F), J(U_M1)*dot(dot(Sigma_F(U_F1, P_F1, U_M1, mu_F), inv(F(U_M1)).T), dot(grad(v_M).T, inv(F(U_M1)).T)))*dx_F
    A_FM09 =  inner(Y_F, div(J(U_M1)*tr(dot(grad(v_M), inv(F(U_M1))))*dot(inv(F(U_M1)), U_F1)))*dx_F
    A_FM10 = -inner(Y_F, div(J(U_M1)*dot(dot(inv(F(U_M1)), grad(v_M)), dot(inv(F(U_M1)), U_F1))))*dx_F

    G_FM1 = -inner(Z_F, J(U_M1)*tr(dot(grad(v_M), inv(F(U_M1))))*mu_F*dot(dot(inv(F(U_M1)).T, grad(U_F1).T), dot(inv(F(U_M1)).T, N_F)))*ds_F
    G_FM2 =  inner(Z_F, J(U_M1)*mu_F*dot(dot(inv(F(U_M1)).T, dot(grad(v_M).T, inv(F(U_M1)).T)), dot(grad(U_F1).T, dot(inv(F(U_M1)).T, N_F ))))*ds_F
    G_FM3 =  inner(Z_F, J(U_M1)*mu_F*dot(dot(inv(F(U_M1)).T, dot(grad(U_F1).T, inv(F(U_M1)).T)), dot(grad(v_M).T, dot(inv(F(U_M1)).T, N_F))))*ds_F

    A_SM01 = -inner(Z_S('+'), J(U_M1)('+')*tr(dot(grad(v_M('+')), inv(F(U_M1)('+'))))*dot(dot(Sigma_F(U_F1, P_F1, U_M1, mu_F)('+'), inv(F(U_M1)('+')).T), N))*d_FSI
    A_SM02 =  inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(grad(U_F1('+')), dot(inv(F(U_M1))('+'), grad(v_M('+')))), dot(inv(F(U_M1))('+'), dot(inv(F(U_M1)).T('+'), N))))*d_FSI
    A_SM03 =  inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(grad(U_F1('+')).T, dot(inv(F(U_M1)).T('+'), grad(v_M('+')).T)), dot(inv(F(U_M1)).T('+'), dot(inv(F(U_M1)).T('+'), N))))*d_FSI
    A_SM04 =  inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(grad(U_F1('+')), inv(F(U_M1))('+')), dot(inv(F(U_M1)).T('+'), dot(grad(v_M('+')).T, dot(inv(F(U_M1)).T('+'), N)))))*d_FSI
    A_SM05 =  inner(Z_S('+'), J(U_M1)('+')*mu_F*dot(dot(inv(F(U_M1)).T('+'), grad(U_M1('+')).T), dot(inv(F(U_M1)).T('+'), dot(grad(v_M('+')).T, dot(inv(F(U_M1)).T('+'), N)))))*d_FSI
    A_SM06 = -inner(Z_S('+'), J(U_M1)('+')*dot(dot(P_F1('+')*I('+'), inv(F(U_M1)).T('+')), dot(grad(v_M('+')).T, dot(inv(F(U_M1)).T('+'), N))))*d_FSI

    A_MM01 = -(alpha_M/k)*inner(Z_M0 - Z_M, v_M)*dx_F + inner(sym(grad(Z_M)), Sigma_M(v_M, mu_M, lmbda_M))*dx_F
    A_MM02 = inner(Z_M('+'), q_M('+'))*d_FSI + eps*inner(Y_M, q_M)*dx_F
    A_MM03 = inner(Y_M('+'), v_M('+'))*d_FSI

    # Collect forms
    A_FF = A_FF01 + A_FF02 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07 + G_FF
    A_SF = A_SF01 + A_SF02 + A_SF03
    G_FM = G_FM1  + G_FM2  + G_FM3
    A_FM = A_FM01 + A_FM02 + A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08 + A_FM09 + A_FM10 + G_FM
    A_SM = A_SM01 + A_SM02 + A_SM03 + A_SM04 + A_SM05 + A_SM06
    A_MM = A_MM01 + A_MM02 + A_MM03
    A_system = A_FF + A_FM + A_SS + A_SF + A_SM + A_MM + A_MS

    # Define goal funtional
    goal_functional = problem.evaluate_functional(v_F, q_F, v_S, q_S, v_M, dx_F, dx_S, dx_M)

    # Define the dual rhs and lhs
    A = lhs(A_system)
    L = rhs(A_system) + goal_functional

    info_blue("Dual forms created")

    return A, L

def create_dual_bcs(problem, W):
    "Create boundary conditions for dual problem"

    bcs = []

    # Boundary conditions for dual velocity
    for boundary in problem.fluid_velocity_dirichlet_boundaries():
        bcs += [DirichletBC(W.sub(0), (0, 0), boundary)]
    bcs += [DirichletBC(W.sub(0), (0, 0), problem.fsi_boundary, 1)]

    # Boundary conditions for dual pressure
    for boundary in problem.fluid_pressure_dirichlet_boundaries():
        bcs += [DirichletBC(W.sub(1), 0, boundary)]

    # Boundary conditions for dual structure displacement and velocity
    for boundary in problem.structure_dirichlet_boundaries():
        bcs += [DirichletBC(W.sub(2), (0, 0), boundary)]
        bcs += [DirichletBC(W.sub(3), (0, 0), boundary)]

    # Boundary conditions for dual mesh displacement
    bcs += [DirichletBC(W.sub(4), (0, 0), DomainBoundary())]

    # In addition to the above boundary conditions, we also need to
    # add homogeneous boundary conditions for Z_F and Z_M on the FSI
    # boundary. Note that the no-slip boundary condition for U_F does
    # not include the FSI boundary when interpreted as a boundary
    # condition for Z_F if it is defined in terms of 'on_boundary'
    # which has a different meaning for the full mesh.

    # Boundary condition for Z_F on FSI boundary
    bcs += [DirichletBC(W.sub(0), (0, 0), problem.fsi_boundary, 2)]

    # Boundary condition for Z_M on FSI boundary
    bcs += [DirichletBC(W.sub(4), (0, 0), problem.fsi_boundary, 2)]

    return bcs
