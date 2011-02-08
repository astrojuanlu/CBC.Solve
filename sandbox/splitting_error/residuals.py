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

def strong_residuals(U0, U1, Z, EZ, dg, kn, problem):
    "Return strong residuals (integrated by parts)"

    # Extract variables
    uh0, ph0 = U0
    uh1, ph1 = U1
    z, y     = Z
    Ez, Ey   = EZ

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

    # Define strong residual 
    sR =  dg*(1/kn)*rho*inner(Ez - z, uh1 - uh0)*dx \
       +  dg*rho*inner(Ez - z, dot(grad(u_mid), u_mid))*dx \
       +  dg*inner(Ez - z, div(sigma(u_mid, ph0, mu)))*dx \
       +  avg(dg)*inner(Ez('+') - z('+'), jump(dot(sigma(u_mid, ph0, mu), n)))*dS

    
    return sR
