"""This module defines the three problems:

  FluidProblem     - the fluid problem (F)
  StructureProblem - the structure problem (S)
  MeshProblem      - the mesh problem (M)
"""

from cbc.flow import NavierStokes
from cbc.twist import Hyperelasticity, StVenantKirchhoff
from cbc.twist import DeformationGradient, PiolaTransform

# FIXME: Clean this up
from common import *

# Define fluid problem
class FluidProblem(NavierStokes):

    def __init__(self):
        NavierStokes.__init__(self)
        self.V = VectorFunctionSpace(Omega_F, "CG", 2)
        self.VP1 = VectorFunctionSpace(Omega_F, "CG", 1)
        self.Q = FunctionSpace(Omega_F, "CG", 1)
        self.U_F_temp = Function(self.V)
        self.U_F = Function(self.VP1)
        self.P_F = Function(self.Q)

    def mesh(self):
        return omega_F1

    def viscosity(self):
        return 0.002

    def density(self):
        return 1.0

    def mesh_velocity(self, V):
        self.w = Function(V)
        return self.w

    def boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0.0, 0.0)), noslip)

        # FIXME: Anders fix DirichletBC to take int or float instead of Constant

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(1.0), inflow)
        bcp1 = DirichletBC(Q, Constant(0.0), outflow)

        return [bcu], [bcp0, bcp1]

    def end_time(self):
        return T

    def time_step(self):
        return dt

    def compute_fluid_stress(self, u_F, p_F, U_M):

        # Map u and p back to reference domain
        self.U_F_temp.vector()[:] = u_F.vector()[:]
        self.P_F.vector()[:] = p_F.vector()[:]
        
        print "Projecting fluid-stress to P1 elements"
        # Project U_F to a P1 element
        self.U_F = project(self.U_F_temp, self.VP1)

        # Compute mesh deformation gradient
        F = DeformationGradient(U_M)
        F_inv = inv(F)
        F_inv_T = F_inv.T
        I = variable(Identity(2))
       
        # Compute mapped stress (sigma_F \circ Phi) (here, grad "=" Grad)
        mu = self.viscosity()
        sigma_F = mu*(grad(self.U_F)*F_inv + F_inv_T*grad(self.U_F).T) \
                  - self.P_F*I

        # Map to physical stress
        Sigma_F = PiolaTransform(sigma_F, U_M)

        return Sigma_F

    def update_mesh_displacement(self, U_M):

        # Update the mesh
        X  = Omega_F.coordinates()
        x0 = omega_F0.coordinates()
        x1 = omega_F1.coordinates()
        dofs = U_M.vector().array()
        dim = omega_F1.geometry().dim()
        N = omega_F1.num_vertices()
        for i in range(N):
            for j in range(dim):
                x1[i][j] = X[i][j] + dofs[j*N + i]

        # Update mesh
        omega_F1.coordinates()[:] = x1

        # Smooth the mesh
        omega_F1.smooth(mesh_smooth)

        # Update mesh velocity
        wx = self.w.vector().array()
        for i in range(N):
            for j in range(dim):
                wx[j*N + i] = (x1[i][j] - x0[i][j]) / dt

        self.w.vector()[:] = wx

        # Reassemble matrices
        self.solver.reassemble()

    def update_extra(self):

        # FIXME: The solver should call this function automatically

        # Copy mesh coordinates
        omega_F0.coordinates()[:] = omega_F1.coordinates()[:]

    def __str__(self):
        return "Pressure driven channel (2D) with an obstructure"

# Define struture problem
class StructureProblem(Hyperelasticity):

    def __init__(self):

        # Define functions and function spaces for transfer the fluid stress
        # FIXME: change name on function spaces
        self.V_F = VectorFunctionSpace(Omega_F, "CG", 1)
        self.v_F = TestFunction(self.V_F)
        self.N_F = FacetNormal(Omega_F)
        self.V_S = VectorFunctionSpace(Omega_S, "CG", 1)

        Hyperelasticity.__init__(self)

    def mesh(self):
        return Omega_S

    def dirichlet_conditions(self):
        fix = Constant((0,0))
        return [fix]

    def dirichlet_boundaries(self):
        #FIXME: Figure out how to use the constants above in the
        #following boundary definitions
        bottom ="x[1] == 0.0"
        return [bottom]

    def update_fluid_stress(self, Sigma_F):

        # Assemble traction on fluid domain
        print "Assembling traction on fluid domain"
        L_F = inner(self.v_F, dot(Sigma_F, self.N_F))*ds
        B_F = assemble(L_F)

        # Transfer values to structure domain
        print "Transferring values to structure domain"

        # Add contribution from fluid vector to structure
        B_S = Vector(self.V_S.dim())
        fsi_add_f2s(B_S, B_F)

        # In the structure solver the body force is defined on
        # the LHS...
        self.fluid_load.vector()[:] = - B_S.array()

    def neumann_conditions(self):
        self.fluid_load = Function(self.V_S)
        return [self.fluid_load]

    def neumann_boundaries(self):
        # Return the entire structure boundary as the Neumann
        # boundary, knowing that the Dirichlet boundary will overwrite
        # it at the bottom
        return["on_boundary"]

    def reference_density(self):
        return 1.0

    def material_model(self):
        mu    = 0.15
        lmbda = 0.25
        return StVenantKirchhoff([mu, lmbda])

    def time_stepping(self):
        return "CG1"

    def time_step(self):
        return dt

    def end_time(self):
        return T

    def __str__(self):
        return "The structure problem"

# Define mesh problem (time-dependent linear elasticity)
class MeshProblem():

    def __init__(self):

        # Define functions etc.
        V_M = VectorFunctionSpace(Omega_F, "CG", 1)
        v  = TestFunction(V_M)
        u = TrialFunction(V_M)
        u0 = Function(V_M)
        u1 = Function(V_M)
        displacement = Function(V_M)
        bcs = DirichletBC(V_M, displacement, compile_subdomains("on_boundary"))

        # Define the stress tensor
        def sigma(v):
            return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(v.cell().d)

        # Define mesh parameters
        mu = 3.8461
        lmbda = 5.76
        alpha = 1.0

        # Define form (cG1 scheme) (lhs/rhs do not work with sym_grad...)
        k = Constant(dt)
        a = alpha*inner(v, u)*dx + 0.5*k*inner(sym(grad(v)), sigma(u))*dx
        L = alpha*inner(v, u0)*dx - 0.5*k*inner(sym(grad(v)), sigma(u0))*dx
        A = assemble(a)

        # Store variables for time stepping (and saving data)
        self.u = u
        self.u0 = u0
        self.u1 = u1
        self.A = A
        self.L = L
        self.dt = dt
        self.displacement = displacement
        self.bcs = bcs
        self.V_M = V_M
        self.alpha = alpha
        self.mu = mu
        self.lmbda = lmbda

    # Compute mesh equation
    def step(self, dt):
        b = assemble(self.L)
        self.bcs.apply(self.A, b)
        solve(self.A, self.u1.vector(), b)
        return self.u1

    # Update structure displacement
    def update_structure_displacement(self, U_S):
        self.displacement.vector().zero()
        fsi_add_s2f(self.displacement.vector(), U_S.vector())

    # Update mesh solution
    def update(self, t):
        self.u0.assign(self.u1)
        return self.u1

    def __str__(self):
        return "The mesh problem"
