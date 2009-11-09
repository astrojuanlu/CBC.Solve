__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2009-11-06

__all__ = ["NavierStokesProblemStep", "NavierStokesProblem"]

from dolfin import *
from cbc.common import CBCProblem
from cbc.flow.solvers import NavierStokesSolverStep, NavierStokesSolver

class NavierStokesProblemStep(CBCProblem):
    "Base class for all Navier-Stokes problems (single time-step)"

    def __init__(self):
        self.intial_conditions(VectorFunctionSpace(self.mesh(), "CG", 2), \
                                   FunctionSpace(self.mesh(), "CG", 1))

    def solve(self):
        "Solve and return computed solution (u, p)"
        solver = NavierStokesSolverStep()
        return solver.solve(self)

    #--- Functions that must be overloaded by subclasses ---

    def mesh(self):
        "Return mesh"
        missing_function("mesh")

    #--- Functions that may optionally be overloaded by subclasses ---

    def intial_conditions(self, V, Q):
        "Return initial conditions for velocity and pressure"
        u0 = Constant(V.mesh(), (0,)*V.mesh().geometry().dim())
        p0 = Constant(V.mesh(), 0)
        return u0, p0

    def time_step(self):
        "Return preferred time step"
        return None

    def _store_previous_solution(self, u0, p0):
        "Return initial conditions for velocity and pressure"
        self.u0 = u0
        self.p0 = p0

    def _get_previous_solution(self):
        return self.u0, self.p0    

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
        return "Navier-Stokes problem (single time-step)"

class NavierStokesProblem(NavierStokesProblemStep):
    "Base class for all (dynamic) Navier-Stokes problems"

    def solve(self):
        "Solve and return computed solution (u, p)"
        solver = NavierStokesSolver()
        return solver.solve(self)

    #--- Functions that may optionally be overloaded by subclasses ---

    def end_time(self):
        "Return end time"
        return 1.0

    def max_velocity(self):
        "Return maximum velocity (used for selecting time step)"
        return 1.0

    def __str__(self):
        "Return a short description of the problem"
        return "Navier-Stokes problem (dynamic)"
