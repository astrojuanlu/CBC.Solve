"This module specifies the variational forms for the dual fluid problem."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-08

from dolfin import *

def create_dual_forms(problem, Omega, k,
                      v, q, z, y, z0, 
                      primal_sol_0, 
                      primal_sol_1):

    info_blue("Creating dual forms")

    # Get problem parameters
    rho = problem.density()
    mu  = problem.viscosity()

    # Get primal data 
    uh0, ph0 = primal_sol_0
    uh1, ph1 = primal_sol_1

    # Define facet normal
    n = FacetNormal(Omega)

    # Define identity matrix in 2D
    I = Identity(2)
    
    # Define the symetric gradient
    def epsilon(v):
        return 0.5*(grad(v) + grad(v).T)

    # Define the fluid stress tensor
    def sigma(v, q):
        return  2*mu*epsilon(v) - q*I
    
    # FIXME: Talk to Anders/Mats about this cG(1) formulation 
    # Defined mid point value for the fluid 
    u_mid = 0.5*(uh0 + uh1)

    # Define the dual momemtum form
    dual_mom = -(1/k)*rho*inner((z0 - z), v)*dx \
               + rho*inner(z, dot(grad(u_mid), v))*dx \
               + rho*inner(z, dot(grad(v), u_mid))*dx \
               + inner(epsilon(z), sigma(v,q))*dx \
               - inner(z, dot(sigma(v,q),n ))*ds
     
    # Define the dual continuity form
    dual_cont = inner(y, div(v))*dx

    # Collect momentum and continuity forms
    dual = dual_mom + dual_cont

    # Define Riezs' representer (Gauss pulse)
    psi = Expression("exp(-(pow(25*(x[0] - 0.75), 2) + pow(25*(x[1] - 0.25), 2)) / 5.0)")

    # FIXME: Should be v*psi*dx ?
    # Define goal funtional 
    goal_functional = (v[0] + v[1])*psi*dx
    
    # Define the dual rhs and lhs
    A = lhs(dual) 
    L = rhs(dual) + goal_functional

    info_blue("Dual forms created")

    return A, L
