"This module implements residuals used for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-08

from dolfin import *
from operators import *

def weak_residuals(U0, U1, U, Z, kn, problem):
    "Return weak residuals used for Ek. The primal data is evaluated in at U1"

    # Extract variables
    u0, p0 = U0
    u1, p1 = U1
    u, p   = U
    z, y   = Z
   
    # Get problem parameters
    Omega = problem.fluid_mesh()
    rho   = problem.density()
    mu    = problem.viscosity()
    
    # Define normals
    n = FacetNormal(Omega)

    # Define weak residual for the time error 
    wR  = (1/kn)*rho*inner(z, u1 - u0)*dx \
        + rho*inner(z, dot(grad(u), u))*dx \
        + inner(epsilon(z), sigma(u, p, mu))*dx \
        - inner(z, dot(sigma(u, p, mu), n))*ds \
        + inner(y, div(u))*dx
    
    return wR

def strong_residuals(U, U0, U1, Z, EZ, dg, kn, problem):
    "Return strong residuals (integrated by parts)"

    # Extract variables
    # Here U, U0, U1, Z and EZ are mid point values
    u, p   = U   
    u0, p0 = U0
    u1, p1 = U1 
    z, y   = Z   
    Ez, Ey = EZ  

    # Get problem parameters
    Omega = problem.fluid_mesh()
    rho   = problem.density()
    mu    = problem.viscosity()

    # Define normals
    n = FacetNormal(Omega)

    # Define element residuals for momentum eq.
    sR_mom_K = dg*(1/kn)*rho*inner(Ez - z, u1 - u0)*dx \
             + dg*rho*inner(Ez - z, dot(grad(u), u))*dx \

    # Define moment eq. residuals defined on facets, GRAD U formulation
    sR_mom_dK = avg(dg)*inner(Ez('+') - z('+'), jump(dot(2*mu*0.5*(grad(u) + grad(u).T), n)))*dS \
              + dg*inner(Ez - z, dot(sigma(u, p, mu) , n))*ds 

#     # Define moment eq. residuals defined on facets, SIGMA formulation
#     sR_mom_dK = avg(dg)*inner(Ez('+') - z('+'), jump(dot(sigma(u,p,mu), n)))*dS \
#               + dg*inner(Ez - z, dot(sigma(u, p, mu) , n))*ds 

    # Define continuity eq. element residuals
    sR_con_K =  dg*inner(Ey - y, div(u))*dx

    
    return (sR_mom_K, sR_mom_dK, sR_con_K)
