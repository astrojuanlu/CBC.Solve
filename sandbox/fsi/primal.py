# A simple FSI problem involving a hyperelastic obstruction in a
# Navier-Stokes flow field. Lessons learnt from this exercise will be
# used to construct an FSI class in the future.

from cbc.flow import *
from cbc.twist import *
from cbc.common.utils import *
from numpy import array, append
from common import *
from math import ceil

plot_solution = False

# Define fluid problem
class FluidProblem(NavierStokes):

    def __init__(self):
        NavierStokes.__init__(self)
        self.V = VectorFunctionSpace(Omega_F, "CG", 2)
        self.Q = FunctionSpace(Omega_F, "CG", 1)
        self.U_F = Function(self.V)
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
        self.U_F.vector()[:] = u_F.vector()[:]
        self.P_F.vector()[:] = p_F.vector()[:]

        # Compute mesh deformation gradient
        F = DeformationGradient(U_M)
        F_inv = inv(F)
        F_inv_T = F_inv.T

        # Compute mapped stress (sigma_F \circ Phi) (here, grad "=" Grad)
        mu = self.viscosity()
        sigma_F = mu*(grad(self.U_F)*F_inv + F_inv_T*grad(self.U_F).T \
                  - self.P_F*Identity(self.U_F.cell().d))

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
        omega_F1.smooth(50)

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
        bottom ="x[1] == 0.0 && x[0] >= 1.4 && x[0] <= 1.6"
        return [bottom]

    def update_fluid_stress(self, Sigma_F):

        # Assemble traction on fluid domain
        print "Assembling traction on fluid domain"
        L_F = dot(self.v_F, dot(Sigma_F, self.N_F))*ds
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

    def is_dynamic(self):
        return True

    def time_stepping(self):
        return "CG1"

    def time_step(self):
        return dt

    def end_time(self):
        return T

    def __str__(self):
        return "The structure problem"

# Define mesh problem
class MeshProblem(StaticHyperelasticity):

    def __init__(self):
        self.V_M = VectorFunctionSpace(Omega_F, "CG", 1)

        StaticHyperelasticity.__init__(self)

    def mesh(self):
        return Omega_F

    def dirichlet_conditions(self):
        self.displacement = Function(self.V_M)
        return [self.displacement]

    def dirichlet_boundaries(self):
        return ["on_boundary"]

    def material_model(self):
        E  = 10.0
        nu = 0.3
        mu   = E / (2.0*(1.0 + nu))
        lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))
        return LinearElastic([mu, lmbda])

    def update_structure_displacement(self, U_S):
        self.displacement.vector().zero()
        fsi_add_s2f(self.displacement.vector(), U_S.vector())
        self.displacement.vector().array()

    def __str__(self):
        return "The mesh problem"

# Define the three problems
F = FluidProblem()
S = StructureProblem()
M = MeshProblem()

# Solve mesh equation (will give zero vector first time which corresponds to
# identity map between the current domain and the reference domain)
U_M = M.solve()

# Create inital displacement vector
V0 = VectorFunctionSpace(Omega_S, "CG", 1)
v0 = Function(V0)
U_S_vector_old  = v0.vector()

# Create files for storing solution
file_u_F = File("u_F.pvd")
file_p_F = File("p_F.pvd")
file_U_S = File("U_S.pvd")
file_P_S = File("P_S.pvd")
file_U_M = File("U_M.pvd")

# Fix time step if needed. Note that this has to be done
# in oder to save the primal data at the correct time
n = ceil(T / dt)
t_range = linspace(0, T, n + 1)[1:]
dt = t_range[0]

# Time-stepping
t = 0
while t <= T:

 # Fixed point iteration on FSI problem
    r = 2*tol
    while r > tol:

        # Solve fluid equation
        u_F, p_F = F.step(dt)

        # Update fluid stress for structure problem
        Sigma_F = F.compute_fluid_stress(u_F, p_F, U_M)
        S.update_fluid_stress(Sigma_F)

        # Solve structure equation
        structure_sol = S.step(dt)
        U_S, P_S = structure_sol.split(True)

        # Update structure displacement for mesh problem
        M.update_structure_displacement(U_S)

        # Solve mesh equation
        U_M = M.solve()

        # Update mesh displcament and mesh velocity
        F.update_mesh_displacement(U_M)

        # Plot solutions
        if plot_solution:
           plot(u_F, title="Fluid velocity")
           plot(U_S, title="Structure displacement", mode="displacement")
           plot(U_M, title="Mesh displacement", mode="displacement")
           plot(F.w, title="Mesh velocity")

        # Compute residual
        U_S_vector_old.axpy(-1, U_S.vector())
        r = norm(U_S_vector_old)
        U_S_vector_old[:] = U_S.vector()[:]

        print "*******************************************"
        print "Solving the problem at t = ", str(t)
        print "With time step dt =" , str(dt)
        print ""
        print "norm(r)", str(r)
        print " "
        print "*******************************************"

        # Check convergence
        if r < tol:
            break

    # Move to next time step
    F.update(t)
    S.update()

    # FIXME: This should be done automatically by the solver
    F.update_extra()

    # Store solutions
    file_u_F << u_F
    file_p_F << p_F
    file_U_S << U_S
    file_P_S << P_S
    file_U_M << U_M

    # Store primal vectors
    primal_u_F.store(u_F.vector(), t)
    primal_p_F.store(p_F.vector(), t)
    primal_U_S.store(U_S.vector(), t)
    primal_P_S.store(P_S.vector(), t)
    primal_U_M.store(U_M.vector(), t)

    # Move on to the next time level
    t += dt

# Define convergence indicator for structure (integral over displacement in x1-direction)
displacement = assemble(U_S[0]*dx, mesh = U_S.function_space().mesh())
functional = u_F((4.0, 0.5))[0]

# Print convergence indicators
print "*******************************************"
print "Mesh size: %g "%  mesh.num_cells()
print "mesh h %g"  % mesh.hmin()
print "Time step kn: %g"% dt
print "End time T: %g"%T
print "TOL %g" % tol
print " "
print " "
print "Functional: %g" % functional
print "Displacement %g"% displacement
print "*******************************************"

