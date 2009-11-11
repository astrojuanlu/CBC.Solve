__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2009-11-06

__all__ = ["StaticNavierStokesProblem", "NavierStokesProblem"]

from dolfin import Constant, error
from cbc.common import CBCProblem
from cbc.flow.solvers import StaticNavierStokesSolver, NavierStokesSolver

class StaticNavierStokesProblem(CBCProblem):
    "Base class for all static Navier-Stokes problems"

    def solve(self):
        "Solve and return computed solution (u, p)"
        solver = StaticNavierStokesSolver()
        return solver.solve(self)

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

    def boundary_conditions(self, V, Q):
        "Return boundary conditions for velocity and pressure"
        return [], []

    def __str__(self):
        "Return a short description of the problem"
        return "Navier-Stokes problem (static)"

class NavierStokesProblem(StaticNavierStokesProblem):
    "Base class for all (dynamic) Navier-Stokes problems"

    def solve(self):
        "Solve and return computed solution (u, p)"
        solver = NavierStokesSolver()
        return solver.solve(self)

    #--- Functions that may optionally be overloaded by subclasses ---

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
        return "Navier-Stokes problem (dynamic)"
