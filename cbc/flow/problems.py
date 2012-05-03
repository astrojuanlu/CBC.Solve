__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-04-30

__all__ = ["NavierStokes"]

from dolfin import error, Constant, Parameters, info
from cbc.common import CBCProblem
from cbc.flow.solvers import NavierStokesSolver
from cbc.flow.saddlepointsolver import TaylorHoodSolver
from ufl import grad, Identity

class NavierStokes(CBCProblem):
    "Base class for all Navier-Stokes problems"

    def __init__(self, parameters=None, solver="ipcs"):
        "Create Navier-Stokes problem"

        self.parameters = Parameters("problem_parameters")

        # Create solver
        if solver == "taylor-hood":
            info("Using Taylor-Hood based Navier-Stokes solver")
            self.solver = TaylorHoodSolver(self)
        elif solver == "ipcs":
            info("Using IPCS based Navier-Stokes solver")
            self.solver = NavierStokesSolver(self)
        else:
            error("Unknown Navier--Stokes solver: %s" % solver)

        # Set up parameters
        self.parameters.add(self.solver.parameters)

    def solve(self):
        "Solve and return computed solution (u, p)"

        # Update solver parameters
        self.solver.parameters.update(self.parameters["solver_parameters"])

        # Call solver
        return self.solver.solve()

    def step(self, dt):
        "Make a time step of size dt"

        # Update solver parameters
        self.solver.parameters.update(self.parameters["solver_parameters"])

        # Call solver
        return self.solver.step(dt)

    def update(self, t):
        "Propagate values to next time step"
        return self.solver.update(t)

    def solution(self):
        "Return current solution values"
        return self.solver.solution()

    def solution_values(self):
        "Return solution values at t_{n-1} and t_n"
        return self.solver.solution_values()

    #--- Functions that must be overloaded by subclasses ---

    def mesh(self):
        "Return mesh"
        missing_function("mesh")

    #--- Functions that may optionally be overloaded by subclasses ---

    def density(self):
        "Return density"
        return 1.0

    def viscosity(self):
        "Return viscosity"
        return 1.0

    def body_force(self, V):
        "Return body force f"
        return []

    def boundary_traction(self, V):
        "Return boundary traction g = sigma(u, p) * n"
        return []

    def mesh_velocity(self, V):
         "Return mesh velocity (for ALE formulations)"
         w = Constant((0,)*V.mesh().geometry().dim())
         return w

    def boundary_conditions(self, V, Q):
        "Return boundary conditions for velocity and pressure"
        return [], []

    def velocity_dirichlet_values(self):
        "Return Dirichlet boundary values for the velocity"
        return []

    def velocity_dirichlet_boundaries(self):
        "Return Dirichlet boundaries for the velocity"
        return []

    def pressure_dirichlet_values(self):
        "Return Dirichlet boundary conditions for the velocity"
        return []

    def pressure_dirichlet_boundaries(self):
        "Return Dirichlet boundaries for the velocity"
        return []

    def velocity_initial_condition(self):
        "Return initial condition for velocity"
        return None

    def pressure_initial_condition(self):
        "Return initial condition for pressure"
        return 0

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
