"This module specifies the variational forms for the dual FSI problem."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-03-07

from dolfin import *

def create_dual_forms(Omega, k, problem,
                      v_F,  q_F,
                      Z_F,  Y_F,
                      Z_F0, Y_F0,
                      U_F0, P_F0,
                      U_F1, P_F1):
    "Return bilinear and linear forms for one time step."

    info_blue("Creating dual forms")

    # Get problem parameters
    rho_F = problem.fluid_density()
    mu_F = problem.fluid_viscosity()

    # Define normal and identity matrix
    N_F = FacetNormal(Omega)
    I = Identity(U_F0.cell().d)

    # Dual forms
    A_FF01 = -(1/k)*inner((Z_F0 - Z_F), rho_F*v_F)*dx
    A_FF02 =  inner(Z_F, rho_F*dot(grad(v_F), U_F1))*dx
    A_FF03 =  inner(Z_F, rho_F*dot(grad(U_F1), v_F))*dx
    A_FF04 =  inner(grad(Z_F), mu_F*grad(v_F))*dx
    A_FF05 =  inner(grad(Z_F), mu_F*grad(v_F).T)*dx
    A_FF06 = -inner(grad(Z_F), q_F*I)*dx
    A_FF07 =  inner(Y_F, div(v_F))*dx
    G_FF   = -inner(Z_F, mu_F*dot(grad(v_F).T, N_F))*ds

    # Collect forms
    A_system = A_FF01 + A_FF02 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07 + G_FF

    # Define goal funtional
    M, cd, efd, ifd = problem.evaluate_functional(v_F, q_F)

    # Define the dual rhs and lhs
    A = lhs(A_system)
    L = rhs(A_system) + M

    info_blue("Dual forms created")

    return A, L, cd, efd, ifd

def create_dual_bcs(problem, W):
    "Create boundary conditions for dual problem"

    bcs = []

    # Boundary conditions for dual velocity
    for boundary in problem.fluid_velocity_dirichlet_boundaries():
        bcs += [DirichletBC(W.sub(0), (0, 0), boundary)]

    # Boundary conditions for dual pressure
    for boundary in problem.fluid_pressure_dirichlet_boundaries():
        bcs += [DirichletBC(W.sub(1), 0, boundary)]

    return bcs
