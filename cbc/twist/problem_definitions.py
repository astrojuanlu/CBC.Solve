__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from cbc.common import CBCProblem
from cbc.twist.solution_algorithms import StaticMomentumBalanceSolver, MomentumBalanceSolver

class StaticHyperelasticity(CBCProblem):
    """Base class for all static hyperelasticity problems"""

    def __init__(self):
        """Create the static hyperelasticity problem"""
        self.solver = StaticMomentumBalanceSolver(self)

    def solve(self):
        """Solve for and return the computed displacement field, u"""
        return self.solver.solve()

    def body_force(self):
        """Return body force, B"""
        return []

    def dirichlet_conditions(self):
        """Return Dirichlet boundary conditions for the displacment
        field"""
        return []

    def dirichlet_boundaries(self):
        """Return boundaries over which Dirichlet conditions act"""
        return []

    def neumann_conditions(self):
        """Return Neumann boundary conditions for the stress field"""
        return []

    def neumann_boundaries(self):
        """Return boundaries over which Neumann conditions act"""
        return []

    def material_model(self):
        pass

    def first_pk_stress(self, u):
        """Return the first Piola-Kirchhoff stress tensor, P, given a
        displacement field, u"""
        return self.material_model().FirstPiolaKirchhoffStress(u)

    def second_pk_stress(self, u):
        """Return the second Piola-Kirchhoff stress tensor, S, given a
        displacement field, u"""
        return self.material_model().SecondPiolaKirchhoffStress(u)

    def functional(self, u):
        """Return value of goal functional"""
        return None

    def reference(self):
        """Return reference value for the goal functional"""
        return None

    def __str__(self):
        """Return a short description of the problem"""
        return "Static hyperelasticity problem"

class Hyperelasticity(StaticHyperelasticity):
    """Base class for all quasistatic/dynamic hyperelasticity
    problems"""

    def __init__(self):
        """Create the hyperelasticity problem"""
        self.solver = MomentumBalanceSolver(self)

    def init(self, scalar, vector):
        """Initialize problem with function spaces"""
        pass

    def solve(self):
        """Solve for and return the computed displacement field, u"""
        return self.solver.solve()

    def step(self, dt):
        "Take a time step of size dt"
        return self.solver.step(dt)

    def update(self):
        "Propagate values to next time step"
        return self.solver.update()

    def end_time(self):
        """Return the end time of the computation"""
        pass

    def time_step(self):
        """Return the time step size"""
        pass

    def is_dynamic(self):
        """Return True if the inertia term is to be considered, or
        False if it is to be neglected (quasi-static)"""
        return False

    def reference_density(self, scalar):
        """Return the reference density of the material"""
        rho0 = Constant(1.0)
        return rho0

    def initial_conditions(self, vector):
        """Return initial conditions for displacement field, u0, and
        velocity field, v0""" 
        u0 = Constant((0,)*vector.mesh().geometry().dim())
        v0 = Constant((0,)*vector.mesh().geometry().dim())
        return u0, v0

    def dirichlet_conditions(self, vector):
        """Return Dirichlet boundary conditions for the displacment
        field"""
        return []

    def dirichlet_boundaries(self):
        """Return boundaries over which Dirichlet conditions act"""
        return []

    def neumann_conditions(self, vector):
        """Return Neumann boundary conditions for the stress field"""
        T = Constant((0,)*vector.mesh().geometry().dim())
        return [T]

    def neumann_boundaries(self):
        """Return boundaries over which Neumann conditions act"""
        return []
