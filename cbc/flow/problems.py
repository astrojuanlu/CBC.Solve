__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-29

__all__ = ["NavierStokes", "NavierStokesDual"]

from dolfin import error, Constant, Parameters
from cbc.common import CBCProblem
from cbc.flow.solvers import NavierStokesSolver, NavierStokesDualSolver
from ufl import grad, Identity

class NavierStokes(CBCProblem):
    "Base class for all Navier-Stokes problems"

    def __init__(self, parameters=None):
        "Create Navier-Stokes problem"

        # Create solver
        self.solver = NavierStokesSolver(self)

        # Set up parameters
        self.parameters = Parameters("problem_parameters")
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

    def cauchy_stress(self, u, p):
        epsilon = 0.5*(grad(u) + grad(u).T)
        mu = self.viscosity()
        sigma = 2.0*mu*epsilon - p*Identity(u.cell().d)
        return sigma

    def viscous_stress(self, u):
        epsilon = 0.5*(grad(u) + grad(u).T)
        mu = self.viscosity()
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

    def density(self):
        "Return density"
        return 1.0

    def viscosity(self):
        "Return viscosity"
        return 1.0

    def body_force(self, V):
        "Return body force f"
        f = Constant((0,)*V.mesh().geometry().dim())
        return f

    def mesh_velocity(self, V):
         "Return mesh velocity (for ALE formulations)"
         w = Constant((0,)*V.mesh().geometry().dim())
         return w

    def boundary_conditions(self, V, Q):
        "Return boundary conditions for velocity and pressure"
        return [], []

    def initial_conditions(self, V, Q):
        "Return initial conditions for velocity and pressure"
        u0 = Constant((0,)*V.mesh().geometry().dim())
        p0 = Constant(0)
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

class NavierStokesDual(NavierStokes):
    "Base class for all Navier-Stokes dual problems"

    def __init__(self, parameters=None):
        "Create Navier-Stokes dual problem"

        # Create solver
        self.solver = NavierStokesDualSolver(self)

        # Set up parameters
        self.parameters = Parameters("dual_problem_parameters")
        self.parameters.add(self.solver.parameters)

    def functional(self, u, p, V, Q, n):
        "Return goal functional"
        missing_function("functional")

    def boundary_markers(self):
        "Return exterior boundary markers"
        return None
