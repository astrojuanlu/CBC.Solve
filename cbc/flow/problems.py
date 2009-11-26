__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2009-11-12

__all__ = ["NavierStokes"]

from dolfin import Constant, error
from cbc.common import CBCProblem
from cbc.flow.solvers import NavierStokesSolver
from ufl import grad, Identity

class NavierStokes(CBCProblem):
    "Base class for all Navier-Stokes problems"

    def __init__(self):
        "Create Navier-Stokes problem"
        self.solver = NavierStokesSolver(self)

    def solve(self):
        "Solve and return computed solution (u, p)"
        return self.solver.solve()

    def step(self, dt):
        "Make a time step of size dt"
        return self.solver.step(dt)

    def update(self):
        "Propagate values to next time step"
        return self.solver.update()

    def cauchy_stress(self, u, p):
        epsilon = 0.5*(grad(u) + grad(u).T)
        nu = self.viscosity()
        sigma = 2.0*nu*epsilon - p*Identity(u.cell().d)
        return sigma

    def viscous_stress(self, u):
        epsilon = 0.5*(grad(u) + grad(u).T)
        nu = self.viscosity()
        fluid_stress_u = 2.0*nu*epsilon 
        return fluid_stress_u
        
    def pressure_stress(self, u, p):
        fluid_stress_p = - p*Identity(u.cell().d)
        return fluid_stress_p


    #--- Functions that must be overloaded by subclasses ---

    def mesh(self):
        "Return mesh"
        missing_function("mesh")

    #--- Functions that may optionally be overloaded by subclasses ---

    def viscosity(self):
        "Return viscosity"
        return 1.0

    def body_force(self, V):
        "Return body force f"
        f = Constant(V.mesh(), (0,)*V.mesh().geometry().dim())
        return f

    def mesh_velocity(self, V):
         "Return mesh velocity (for ALE formulations)"
         w = Constant(V.mesh(), (0,)*V.mesh().geometry().dim())
         return w

    def boundary_conditions(self, V, Q):
        "Return boundary conditions for velocity and pressure"
        return [], []

    def initial_conditions(self, V, Q):
        "Return initial conditions for velocity and pressure"
        u0 = Constant(V.mesh(), (0,)*V.mesh().geometry().dim())
        p0 = Constant(V.mesh(), 0)
        return u0, p0

    def end_time(self):
        "Return end time"
        return 1.0

    def time_step(self):
        "Return preferred time step"
        return None

    def max_velocity(self):
        "Return maximum velocity (used for selecting time step)"
        return 1.0

    def __str__(self):
        "Return a short description of the problem"
        return "Navier-Stokes problem"
