"This module implements residuals used for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-08

from dolfin import *
from operators import *

def weak_residuals(U0, U1, w, kn, problem):
    "Return weak residuals (used in Ek and Ec)"

    # Extract variables
    # Note: w can be a test function or the dual solution
    uh0, ph0 = U0
    uh1, ph1 = U1
    v, q     = w 

    # Define mid point value
    u_mid = 0.5*(uh0 + uh1)

    # Get problem parameters
    Omega = problem.fluid_mesh()
    rho   = problem.density()
    mu    = problem.viscosity()
    
    # Define normals
    n = FacetNormal(Omega)

    # FIXME: How should the mom. eq be evaluated?
    # FIXME: How should the cont. eq be evaluated?
  
    # Define weak residual
    wR = (1/kn)*rho*inner(v, uh1 - uh0)*dx \
        + rho*inner(v, dot(grad(u_mid), u_mid))*dx \
        + inner(epsilon(v), sigma(u_mid, ph0, mu))*dx \
        - inner(v, dot(sigma(u_mid, ph0, mu), n))*ds \
        + inner(q, div(uh0))*dx
    
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

    # FIXME: How should the mom. eq be evaluated?
    # FIXME: How should the cont. eq be evaluated?

    # Define element residuals for momentum eq.
    sR_mom_K =  dg*(1/kn)*rho*inner(Ez - z, u1 - u0)*dx \
             +  dg*rho*inner(Ez - z, dot(grad(u), u))*dx \
             -  dg*inner(Ez - z, div(sigma(u, p0, mu)))*dx

    # Define moment eq. residuals defined on facets (jumps and BCs)
    sR_mom_dK = avg(dg)*inner(Ez('+') - z('+'), jump(dot(sigma(u, p0, mu), n)))*dS \
              - dg*inner(Ez - z, dot(sigma(u, p0, mu), n))*ds 

    # Define continuity eq. element residuals
    sR_con_K =  dg*inner(Ey - y, div(u0))*dx

    
    return (sR_mom_K, sR_mom_dK, sR_con_K)
