"""This module defines the three subproblems:

  FluidProblem     - the fluid problem (F)
  StructureProblem - the structure problem (S)
  MeshProblem      - the mesh problem (M)
"""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-27

__all__ = ["FluidProblem", "StructureProblem", "extract_solution",
           "extract_num_dofs"]

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
        self.Omega_F = problem.fluid_mesh()
        self.omega_F0 = Mesh(self.Omega_F)
        self.omega_F1 = Mesh(self.Omega_F)

        # Create functions for velocity and pressure on reference domain
        self.V = VectorFunctionSpace(self.Omega_F, "CG", 2)
        self.Q = FunctionSpace(self.Omega_F, "CG", 1)
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
        return self.omega_F1

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

    def update_extra(self):

        # Copy mesh coordinates
        self.omega_F0.coordinates()[:] = self.omega_F1.coordinates()[:]

    def __str__(self):
        return "The fluid problem (F)"

# Define structure problem
class StructureProblem(Hyperelasticity):

    def __init__(self, problem, parameters):

        # Store problem
        self.problem = problem

        # Define function spaces and functions for transfer of fluid stress
        structure_element_degree = parameters["structure_element_degree"]
        Omega_F = problem.fluid_mesh()
        Omega_S = problem.structure_mesh()
        self.V_F = VectorFunctionSpace(Omega_F, "CG", structure_element_degree)
        self.V_S = VectorFunctionSpace(Omega_S, "CG", structure_element_degree)
        self.test_F = TestFunction(self.V_F)
        self.trial_F = TrialFunction(self.V_F)
        self.G_F = Function(self.V_F)
        self.G_S = Function(self.V_S)
        self.N_F = FacetNormal(Omega_F)
        self.N_S = FacetNormal(Omega_S)

        # Calculate number of dofs
        self.num_dofs = 2 * self.V_S.dim()

        # Initialize base class
        Hyperelasticity.__init__(self)

        # Don't plot and save solution in subsolvers
        self.parameters["solver_parameters"]["plot_solution"] = False
        self.parameters["solver_parameters"]["save_solution"] = False
        self.parameters["solver_parameters"]["element_degree"] = parameters["structure_element_degree"]

    def mesh(self):
        return self.problem.structure_mesh()

    def reference_density(self):
        return self.problem.structure_density()

    def dirichlet_values(self):
        return self.problem.structure_dirichlet_values()

    def dirichlet_boundaries(self):
        return self.problem.structure_dirichlet_boundaries()

    def neumann_boundaries(self):
        return [self.problem.structure_neumann_boundaries()]

    def neumann_conditions(self):
        return [self.G_S]

    def material_model(self):
        mu    = self.problem.structure_mu()
        lmbda = self.problem.structure_lmbda()
        return StVenantKirchhoff([mu, lmbda])

    def update_fluid_stress(self, Sigma_F):

        # Project traction to a function on the boundary. This ensures
        # that the integral of G_S inside the structure solver equals
        # the integral of G_F since G_F and G_S are set equal on the
        # common boundary, dof by dof. Furthermore, the integral of
        # G_F against a test function is by the below projection equal
        # to the integral of the tracion Sigma_F N_F so this transfer
        # in fact does not involve an approximation.
        info("Assembling traction on fluid domain")
        new = True
        if new:
            d_FSI = ds(2)
            a_F = dot(self.test_F, self.trial_F)*d_FSI
            L_F = -dot(self.test_F, dot(Sigma_F, self.N_F))*d_FSI
            A_F = assemble(a_F, exterior_facet_domains=self.problem.fsi_boundary_F)
            B_F = assemble(L_F, exterior_facet_domains=self.problem.fsi_boundary_F)
        else:
            a_F = dot(self.test_F, self.trial_F)*ds
            L_F = -dot(self.test_F, dot(Sigma_F, self.N_F))*ds
            A_F = assemble(a_F)
            B_F = assemble(L_F)
        A_F.ident_zeros()
        solve(A_F, self.G_F.vector(), B_F)

        # Add contribution from fluid vector to structure
        info("Transferring values to structure domain")
        self.G_S.vector().zero()
        self.problem.add_f2s(self.G_S.vector(), self.G_F.vector())

        # Uncomment to debug transfer of stress
        #self.debug_stress_transfer(Sigma_F)

    def time_stepping(self):
        return "CG1"

    def time_step(self):
        # Time step will be selected elsewhere
        return self.end_time()

    def end_time(self):
        return self.problem.end_time()

    def debug_stress_transfer(self, Sigma_F):
        "Debug transfer of stress"

        d_FSI = ds(2)

        # Compute direct integral of normal traction
        form = dot(dot(Sigma_F, self.N_F), self.N_F)*d_FSI
        integral_0 = assemble(form, exterior_facet_domains=self.problem.fsi_boundary_F)

        # Compute integral of projected (and negated) normal traction
        form = dot(self.G_F, self.N_F)*d_FSI
        integral_1 = -assemble(form, exterior_facet_domains=self.problem.fsi_boundary_F)

        # Compute integral of transferred projection
        form = dot(self.G_S, self.N_S)*d_FSI
        integral_2 = assemble(form, exterior_facet_domains=self.problem.fsi_boundary_S)

        info("Debugging transfer of stress from fluid to structure.")
        info("The following three integrals should be the same")
        info("")
        info("  I0 = %.16g" % integral_0)
        info("  I1 = %.16g" % integral_1)
        info("  I2 = %.16g" % integral_2)
        info("")

    def __str__(self):
        return "The structure problem (S)"

def extract_num_dofs(F, S):
    "Extract the number of dofs"
    return F.num_dofs + S.num_dofs

def extract_solution(F, S):
    "Extract solution from sub problems"

    # Extract solutions from subproblems
    u_F, p_F = F.solution()
    U_S, P_S = S.solution()

    # Pack up solutions
    U = (u_F, p_F, U_S, P_S)

    return U
