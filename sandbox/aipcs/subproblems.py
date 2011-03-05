"""This module defines the three subproblems:

  FluidProblem     - the fluid problem (F)
  StructureProblem - the structure problem (S)
  MeshProblem      - the mesh problem (M)
"""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-03-06

__all__ = ["FluidProblem", "extract_solution"]

from dolfin import *

from cbc.flow import NavierStokes

# Define fluid problem
class FluidProblem(NavierStokes):

    def __init__(self, problem, parameters):

        # Store problem
        self.problem = problem

        # Store initial and current mesh
        self.Omega = problem.mesh()

        # Initialize base class
        NavierStokes.__init__(self)

        # Handle plotting and saving
        self.parameters["solver_parameters"]["plot_solution"] = parameters["plot_solution"]
        self.parameters["solver_parameters"]["save_solution"] = False

    def mesh(self):
        return self.Omega

    def viscosity(self):
        return self.problem.fluid_viscosity()

    def density(self):
        return self.problem.fluid_density()

    def mesh_velocity(self, V):
        self.w = Function(V)
        return self.w

    def velocity_dirichlet_values(self):
        return self.problem.fluid_velocity_dirichlet_values()

    def velocity_dirichlet_boundaries(self):
        return self.problem.fluid_velocity_dirichlet_boundaries()

    def pressure_dirichlet_values(self):
        return self.problem.fluid_pressure_dirichlet_values()

    def pressure_dirichlet_boundaries(self):
        return self.problem.fluid_pressure_dirichlet_boundaries()

    def velocity_initial_condition(self):
        return self.problem.fluid_velocity_initial_condition()

    def pressure_initial_condition(self):
        return self.problem.fluid_pressure_initial_condition()

    def end_time(self):
        return self.problem.end_time()

    def time_step(self):
        # Time step will be selected elsewhere
        return self.end_time()

    def __str__(self):
        return "The fluid problem (F)"

def extract_solution(F):
    "Extract solution from sub problems"

    # Extract solutions from subproblems
    u_F, p_F = F.solution()

    # Pack up solutions
    U = (u_F, p_F)

    return U
