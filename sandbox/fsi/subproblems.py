"""This module defines the three subproblems:

  FluidProblem     - the fluid problem (F)
  StructureProblem - the structure problem (S)
  MeshProblem      - the mesh problem (M)
"""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-30

__all__ = ["FluidProblem", "StructureProblem", "MeshProblem"]

from dolfin import *

from cbc.flow import NavierStokes
from cbc.twist import Hyperelasticity, StVenantKirchhoff
from cbc.twist import DeformationGradient, PiolaTransform

# Define fluid problem
class FluidProblem(NavierStokes):

    def __init__(self, problem):

        # Store problem
        self.problem = problem

        # Store initial and current mesh
        self.Omega_F = problem.initial_fluid_mesh()
        self.omega_F0 = Mesh(self.Omega_F)
        self.omega_F1 = Mesh(self.Omega_F)

        # Create functions for velocity and pressure on reference domain
        self.V = VectorFunctionSpace(self.Omega_F, "CG", 2)
        self.Q = FunctionSpace(self.Omega_F, "CG", 1)
        self.U_F = Function(self.V)
        self.P_F = Function(self.Q)

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

    def boundary_conditions(self, V, Q):
        return self.problem.fluid_boundary_conditions(V, Q)

    def end_time(self):
        return self.problem.end_time()

    def time_step(self):
        return self.problem.initial_time_step()

    def compute_fluid_stress(self, u_F, p_F, U_M):

        # Map u and p back to reference domain
        self.U_F.vector()[:] = u_F.vector()[:]
        self.P_F.vector()[:] = p_F.vector()[:]

        # Compute mesh deformation gradient
        F = DeformationGradient(U_M)
        F_inv = inv(F)
        F_inv_T = F_inv.T
        I = variable(Identity(U_M.cell().d))

        # Compute mapped stress sigma_F \circ Phi (here, grad "=" Grad)
        mu = self.viscosity()
        sigma_F = mu*(grad(self.U_F)*F_inv + F_inv_T*grad(self.U_F).T) - self.P_F*I

        # Map to physical stress
        Sigma_F = PiolaTransform(sigma_F, U_M)

        return Sigma_F

    def update_mesh_displacement(self, U_M, dt):

        # Get mesh coordinate data
        X  = self.Omega_F.coordinates()
        x0 = self.omega_F0.coordinates()
        x1 = self.omega_F1.coordinates()
        dofs = U_M.vector().array()
        dim = self.omega_F1.geometry().dim()
        N = self.omega_F1.num_vertices()

        # Update omega_F1
        for i in range(N):
            for j in range(dim):
                x1[i][j] = X[i][j] + dofs[j*N + i]

        # FIXME: Is this necessary? Should be taken care of above
        self.omega_F1.coordinates()[:] = x1

        # Smooth the mesh
        num_smoothings = self.problem.parameters["solver_parameters"]["num_smoothings"]
        self.omega_F1.smooth(num_smoothings)

        # Update mesh velocity
        wx = self.w.vector().array()
        for i in range(N):
            for j in range(dim):
                wx[j*N + i] = (x1[i][j] - x0[i][j]) / dt

        # FIXME: Is this necessary? Should be taken care of above
        self.w.vector()[:] = wx

        # Reassemble matrices
        self.solver.reassemble()

    def update_extra(self):
        # FIXME: The solver should call this function automatically
        # Copy mesh coordinates
        self.omega_F0.coordinates()[:] = self.omega_F1.coordinates()[:]

    def __str__(self):
        return "The fluid problem (F)"

# Define structure problem
class StructureProblem(Hyperelasticity):

    def __init__(self, problem):

        # Store problem
        self.problem = problem

        # Define function spaces and functions for transfer of fluid stress
        Omega_F = problem.initial_fluid_mesh()
        Omega_S = problem.structure_mesh()
        self.V_F = VectorFunctionSpace(Omega_F, "CG", 1)
        self.V_S = VectorFunctionSpace(Omega_S, "CG", 1)
        self.v1_F = TestFunction(self.V_F)
        self.v2_F = TrialFunction(self.V_F)
        self.G_F = Function(self.V_F)
        self.G_S = Function(self.V_S)
        self.N_F = FacetNormal(Omega_F)

        # Initialize base class
        Hyperelasticity.__init__(self)

        # Don't plot and save solution in subsolvers
        self.parameters["solver_parameters"]["plot_solution"] = False
        self.parameters["solver_parameters"]["save_solution"] = False

    def mesh(self):
        return self.problem.structure_mesh()

    def dirichlet_conditions(self):
        fix = Constant((0,0))
        return [fix]

    def dirichlet_boundaries(self):
        #FIXME: Figure out how to use the constants above in the
        #following boundary definitions
        bottom ="x[1] == 0.0"
        return [bottom]

    def update_fluid_stress(self, Sigma_F):

        # Project traction to piecewise linears on boundary
        info("Assembling traction on fluid domain")
        a_F = dot(self.v1_F, self.v2_F)*ds
        L_F = -dot(self.v1_F, dot(Sigma_F, self.N_F))*ds
        A_F = assemble(a_F)
        B_F = assemble(L_F)
        A_F.ident_zeros()
        solve(A_F, self.G_F.vector(), B_F)

        # Add contribution from fluid vector to structure
        info("Transferring values to structure domain")
        self.G_S.vector().zero()
        self.problem.add_f2s(self.G_S.vector(), self.G_F.vector())

    def neumann_conditions(self):
        return [self.G_S]

    def neumann_boundaries(self):
        # Return the entire structure boundary as the Neumann
        # boundary, knowing that the Dirichlet boundary will overwrite
        # it at the bottom
        return["on_boundary"]

    def reference_density(self):
        return 15.0

    def material_model(self):
        factor = 500
        mu    = 0.15 * factor
        lmbda = 0.25 * factor
        return StVenantKirchhoff([mu, lmbda])

    def time_stepping(self):
        return "CG1"

    def time_step(self):
        return self.problem.initial_time_step()

    def end_time(self):
        return self.problem.end_time()

    def __str__(self):
        return "The structure problem (S)"

# Define mesh problem (time-dependent linear elasticity)
class MeshProblem():

    def __init__(self, problem):

        # Store problem
        self.problem = problem

        # Get problem parameters
        mu, lmbda, alpha = problem.mesh_parameters()
        Omega_F = problem.initial_fluid_mesh()
        dt = problem.initial_time_step()

        # Define function spaces and functions
        V = VectorFunctionSpace(Omega_F, "CG", 1)
        v = TestFunction(V)
        u = TrialFunction(V)
        u0 = Function(V)
        u1 = Function(V)

        # Define boundary condition
        displacement = Function(V)
        bc = DirichletBC(V, displacement, DomainBoundary())

        # Define the stress tensor
        def sigma(v):
            return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(v.cell().d)

        # Define cG(1) scheme for time-stepping
        k = Constant(dt)
        a = alpha*inner(v, u)*dx + 0.5*k*inner(sym(grad(v)), sigma(u))*dx
        L = alpha*inner(v, u0)*dx - 0.5*k*inner(sym(grad(v)), sigma(u0))*dx

        # Store variables for time stepping
        self.u0 = u0
        self.u1 = u1
        self.a = a
        self.L = L
        self.k = k
        self.displacement = displacement
        self.bc = bc

    def step(self, dt):
        "Compute solution for new time step"

        # Update time step
        self.k.assign(dt)

        # Assemble linear system and apply boundary conditions
        A = assemble(self.a)
        b = assemble(self.L)
        self.bc.apply(A, b)

        # Compute solution
        solve(A, self.u1.vector(), b)

        return self.u1

    def update(self, t):
        self.u0.assign(self.u1)
        return self.u1

    def update_structure_displacement(self, U_S):
        self.displacement.vector().zero()
        self.problem.add_s2f(self.displacement.vector(), U_S.vector())

    def __str__(self):
        return "The mesh problem (M)"
