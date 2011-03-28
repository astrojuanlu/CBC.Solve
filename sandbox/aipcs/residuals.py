"This module implements residuals used for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-03-28

from dolfin import *

def _Sigma(u, p, mu):
    "Return fluid stress"
    I = Identity(u.cell().d)
    return mu*(grad(u) + grad(u).T) - p*I

def inner_product(v, w):
    "Return inner product for mixed velocity/pressure space"
    v1, q1 = v
    v2, q2 = w
    return (inner(v1, v2) + q1*q2)*dx

def weak_residual(U0, U1, U, w, kn, problem):
    "Return weak residuals"

    # Extract variables
    U0, P0 = U0
    U1, P1 = U1
    U, P = U
    v, q = w

    # Get problem parameters
    Omega = problem.mesh()
    rho = problem.fluid_density()
    mu = problem.fluid_viscosity()

    # Define normals
    N = FacetNormal(Omega)

    # Define time derivative
    Dt_U = rho*((U1 - U0)/kn + dot(grad(U), U))

    # Define stress
    Sigma = _Sigma(U, P, mu)

    # Momentum residual
    r0 = inner(v, Dt_U)*dx + inner(grad(v), Sigma)*dx \
       - inner(v, mu*dot(grad(U).T, N))*ds \
       + inner(v, P*N)*ds

    # Continuity residual
    r1 = inner(q, div(U))*dx

    return r0, r1

def strong_residual(U0, U1, U, Z, EZ, w, kn, problem):
    "Return strong residuals (integrated by parts)"

    # Extract variables
    U0, P0 = U0
    U1, P1 = U1
    U,  P  = U
    Z,  Y  = Z
    EZ, EY = EZ

    # Get problem parameters
    Omega = problem.mesh()
    rho = problem.fluid_density()
    mu = problem.fluid_viscosity()

    # Define normals
    N = FacetNormal(Omega)

    # Define midpoint values
    U = 0.5 * (U0 + U1)
    P = 0.5 * (P0 + P1)

    # Define time derivative
    Dt_U = rho * ((U1 - U0)/kn + dot(grad(U), U))

    # Define stress
    Sigma = _Sigma(U, P, mu)

    # Fluid residual contributions
    R0 = w*inner(EZ - Z, Dt_U - div(Sigma))*dx
    R1 = avg(w)*inner(EZ('+') - Z('+'), jump(Sigma, N))*dS
    R2 = w*inner(EZ - Z, dot(mu*grad(U), N))*ds
    R3 = w*inner(EY - Y, div(U))*dx

    return (R0, R1, R2, R3)
