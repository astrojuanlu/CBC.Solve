"This module specifies the variational forms for the dual fluid problem."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-01-07

from dolfin import *

def create_dual_forms(Omega, k, problem,
                      v, q, z, y, z0, uh):

    info_blue("Creating dual forms")

    # Get problem parameters
    rho = problem.fluid_density()
    mu  = problem.fluid_viscosity()

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

    # Define the dual momemtum form
    # FIXME: Add Neumann terms if needed 
    dual_mom = -(1/k)*rho*inner((z0 - z), v)*dx \
               + rho*inner(z, dot(grad(uh), v))*dx \
               + rho*inner(z, dot(grad(v), uh))*dx \
               + inner(epsilon(z), sigma(v,q))*dx

    # Define the dual continuity form
    dual_cont = inner(y, div(v))*dx

    # Collect momentum and continuity forms
    dual = dual_mom + dual_cont

    # FIXME: Goal functional should not be defined here
    # Define goal funtional
    goal_functional = v[1]*dx + v[0]*dx

    # Define the dual rhs and lhs
    A = lhs(dual) 
    L = rhs(dual) + goal_functional

    info_blue("Dual forms created")

    return A, L
