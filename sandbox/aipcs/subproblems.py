"""This module defines the three subproblems:

  FluidProblem     - the fluid problem (F)
  StructureProblem - the structure problem (S)
  MeshProblem      - the mesh problem (M)
"""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-28

__all__ = ["FluidProblem", "extract_solution", "extract_num_dofs"]

from dolfin import *

from cbc.flow import NavierStokes
from cbc.twist import Hyperelasticity, StVenantKirchhoff, PiolaTransform
from operators import Sigma_F as _Sigma_F

# Define fluid problem
class FluidProblem(NavierStokes):

    def __init__(self, problem):

        # Store problem
        self.problem = problem

        # Store initial and current mesh
        self.Omega = problem.mesh()

        # Create functions for velocity and pressure on reference domain
        self.V = VectorFunctionSpace(self.Omega, "CG", 2)
        self.Q = FunctionSpace(self.Omega, "CG", 1)
        self.U_F0 = Function(self.V)
        self.U_F1 = Function(self.V)
        self.P_F0 = Function(self.Q)
        self.P_F1 = Function(self.Q)
        self.U_F = 0.5 * (self.U_F0 + self.U_F1)
        self.P_F = 0.5 * (self.P_F0 + self.P_F1)

        # Calculate number of dofs
        self.num_dofs = self.V.dim() + self.Q.dim()

        # Initialize base class
        NavierStokes.__init__(self)

        # Don't plot and save solution in subsolvers
        self.parameters["solver_parameters"]["plot_solution"] = False
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

def extract_num_dofs(F):
    "Extract the number of dofs"
    return F.num_dofs

def extract_solution(F):
    "Extract solution from sub problems"

    # Extract solutions from subproblems
    u_F, p_F = F.solution()

    # Pack up solutions
    U = (u_F, p_F)

    return U
