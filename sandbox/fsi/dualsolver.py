"This module implements the dual FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-17

from numpy import append
from dolfin import *
from subproblems import *
from dualproblem import dual_forms

class DualSolver:
    "Dual FSI solver"

    def __init__(self, problem, solver_parameters):
        "Create dual FSI solver"

        # Get solver parameters
        self.plot_solution = solver_parameters["plot_solution"]
        self.save_solution = solver_parameters["save_solution"]
        self.save_series = solver_parameters["save_series"]

        # Create files for saving to VTK
        if self.save_solution:
            Z_F_file = File("pvd/Z_F.pvd")
            Y_F_file = File("pvd/Y_F.pvd")
            Z_S_file = File("pvd/Z_S.pvd")
            Y_S_file = File("pvd/Y_S.pvd")
            Z_M_file = File("pvd/Z_M.pvd")
            Y_M_file = File("pvd/Y_M.pvd")

        # Create time series for storing solution
        if self.save_series:
            Z_F_series = TimeSeries("bin/Z_F")
            Y_F_series = TimeSeries("bin/Y_F")
            Z_S_series = TimeSeries("bin/Z_S")
            Y_S_series = TimeSeries("bin/Y_S")
            Z_M_series = TimeSeries("bin/Z_M")
            Y_M_series = TimeSeries("bin/Y_M")

        # Open time series for primal solution
        self.u_F_series = TimeSeries("bin/u_F")
        self.p_F_series = TimeSeries("bin/p_F")
        self.U_S_series = TimeSeries("bin/U_S")
        self.P_S_series = TimeSeries("bin/P_S")
        self.U_M_series = TimeSeries("bin/U_M")

        # Get nodal points for primal time series
        t = self.u_F_series.vector_times()
        T = problem.end_time()
        if not (len(t) > 1 and t[0] == 0.0 and t[-1] == T):
            print "Nodal points for primal time series:", times
            raise RuntimeError, "Missing primal data, unable to solve dual problem."
        self.timestep_range = t

        # Store problem
        self.problem = problem

    def solve(self):
        "Solve the dual FSI problem"

        # Get problem parameters
        T = self.problem.end_time()
        Omega = self.problem.mesh()
        Omega_F = self.problem.fluid_mesh()
        Omega_S = self.problem.structure_mesh()

        # Define function spaces defined on the whole domain
        V_F1 = VectorFunctionSpace(Omega, "CG", 1)
        V_F2 = VectorFunctionSpace(Omega, "CG", 2)
        Q_F  = FunctionSpace(Omega, "CG", 1)
        V_S  = VectorFunctionSpace(Omega, "CG", 1)
        Q_S  = VectorFunctionSpace(Omega, "CG", 1)
        V_M  = VectorFunctionSpace(Omega, "CG", 1)
        Q_M  = VectorFunctionSpace(Omega, "CG", 1)

        # Create mixed function space
        W = MixedFunctionSpace([V_F2, Q_F, V_S, Q_S, V_M, Q_M])

        # Create test and trial functions
        (v_F, q_F, v_S, q_S, v_M, q_M) = TestFunctions(W)
        (Z_F, Y_F, Z_S, Y_S, Z_M, Y_M) = TrialFunctions(W)

        # Create solution functions
        Z0 = Function(W)
        Z1 = Function(W)
        (Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0) = Z0.split()
        (Z_F1, Y_F1, Z_S1, Y_S1, Z_M1, Y_M1) = Z1.split()

        # Create primal functions
        U_F0 = Function(V_F1); U_F1 = Function(V_F1)
        P_F0 = Function(Q_F);  P_F1 = Function(Q_F)
        U_S0 = Function(V_S);  U_S1 = Function(V_S)
        P_S0 = Function(Q_S);  P_S1 = Function(Q_S)
        U_M0 = Function(V_M);  U_M1 = Function(V_M)

        # Create time step (value set in each time step)
        k = Constant(0.0)

        # Create variational forms for dual problem
        A, L = dual_forms(Omega_F, Omega_S, k, self.problem,
                          v_F,  q_F,  v_S,  q_S,  v_M,  q_M,
                          Z_F,  Y_F,  Z_S,  Y_S,  Z_M,  Y_M,
                          Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0,
                          U_F0, P_F0, U_S0, P_S0, U_M0,
                          U_F1, P_F1, U_S1, P_S1, U_M1)

        # Create dual boundary conditions
        bcs = self._create_boundary_conditions(W)

        # Time-stepping
        p = Progress("Time-stepping")
        for i in reversed(range(len(self.timestep_range) - 1)):

            # Get current time and time step
            t0 = self.timestep_range[i]
            t1 = self.timestep_range[i + 1]
            dt = t1 - t0
            k.assign(dt)

            # Read primal data
            self._read_primal_data(U_F0, P_F0, U_S0, P_S0, U_M0, t0)

            # Assemble matrix and vector
            matrix = assemble(A,
                              cell_domains = cell_domains,
                              interior_facet_domains = interior_facet_domains,
                              exterior_facet_domains = exterior_boundary)

            #
            vector = assemble(L, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains, exterior_facet_domains = exterior_boundary)

            # Apply bcs
            for bc in bcs:
                bc.apply(dual_matrix, dual_vector)

            # Remove inactive dofs
            dual_matrix.ident_zeros()

            # Compute dual solution
            solve(dual_matrix, Z.vector(), dual_vector)

            # Copy solution from previous interval
            # Dual varibles
            Z_F0.assign(Z_F)
            Y_F0.assign(Y_F)
            Z_S0.assign(Z_S)
            Y_S0.assign(Y_S)
            Z_M0.assign(Z_M)
            #Y_M0.assign(Y_M)

            # Primal varibles
            U_M0.assign(U_M)
            U_F0.assign(U_F)

            # Plot solutions
            if plot_solution:
                plot(Z_F, title="Dual fluid velocity")
                plot(Y_F, title="Dual fluid pressure")
                plot(Z_S, title="Dual displacement")
                plot(Y_S, title="Dual structure velocity")
                plot(Z_M, title="Dual mesh displacement")
                plot(Y_M, title="Dual mesh Lagrange Multiplier")

            # Store dual bin files. NOTE: we save dual solutions at t!
            if store_bin_files:
                dual_Z.store(Z.vector(), t)

            # Store vtu files
            if store_vtu_files:
                file_Z_F << Z_F
                file_Y_F << Y_F
                file_Z_S << Z_S
                file_Y_S << Y_S
                file_Z_M << Z_M
                file_Y_M << Y_M

            # Move to next time interval
            t += kn

    def _read_primal_data(self, U_F, P_F, U_S, P_S, U_M, t):
        """Read primal data at given time. This includes reading the data
        stored on file, transferring the data to the full domain, and
        downsampling the velocity from P2 to P1."""

        info("Reading primal data at t = %g" % t)

        # Create vectors for primal dof values on local meshes
        local_vals_u_F = Vector()
        local_vals_p_F = Vector()
        local_vals_U_S = Vector()
        local_vals_P_S = Vector()
        local_vals_U_M = Vector()

        # Retrieve primal data
        self.u_F_series.retrieve(local_vals_u_F, t)
        self.p_F_series.retrieve(local_vals_p_F, t)
        self.U_S_series.retrieve(local_vals_U_S, t)
        self.P_S_series.retrieve(local_vals_P_S, t)
        self.U_M_series.retrieve(local_vals_U_M, t)

        # Get meshes
        Omega   = self.problem.mesh()
        Omega_F = self.problem.fluid_mesh()
        Omega_S = self.problem.structure_mesh()

        # Get vertex mappings from local meshes to global mesh
        vmap_F = Omega_F.data().mesh_function("global vertex indices").values()
        vmap_S = Omega_S.data().mesh_function("global vertex indices").values()

        # Get the number of vertices and edges
        Omega_F.init(1)
        Nv   = Omega.num_vertices()
        Nv_F = Omega_F.num_vertices()
        Ne_F = Omega_F.num_edges()

        # Compute mapping to global dofs
        global_dofs_U_F = append(vmap_F, vmap_F + Nv)
        global_dofs_P_F = vmap_F
        global_dofs_U_S = append(vmap_S, vmap_S + Nv)
        global_dofs_P_S = append(vmap_S, vmap_S + Nv)
        global_dofs_U_M = append(vmap_F, vmap_F + Nv)

        # Get rid of P2 dofs for u_F and create a P1 function
        local_vals_u_F = append(local_vals_u_F[:Nv_F], local_vals_u_F[Nv_F + Ne_F: 2*Nv_F + Ne_F])

        # Set degrees of freedom for primal functions
        U_F.vector()[global_dofs_U_F] = local_vals_u_F
        P_F.vector()[global_dofs_P_F] = local_vals_p_F
        U_S.vector()[global_dofs_U_S] = local_vals_U_S
        P_S.vector()[global_dofs_P_S] = local_vals_P_S
        U_M.vector()[global_dofs_U_M] = local_vals_U_M

    def _create_boundary_conditions(self, W):
        "Create boundary conditions for dual problem"

        bcs = []

        # Boundary conditions for dual velocity
        for boundary in self.problem.fluid_velocity_dirichlet_boundaries():
            bcs += [DirichletBC(W.sub(0), (0, 0), boundary)]
        bcs += [DirichletBC(W.sub(0), (0, 0), self.problem.fsi_boundary, 1)]

        # Boundary conditions for dual pressure
        for boundary in self.problem.fluid_pressure_dirichlet_boundaries():
            bcs += [DirichletBC(W.sub(1), 0, boundary)]

        # Boundary conditions for dual structure displacement and velocity
        for boundary in self.problem.structure_dirichlet_boundaries():
            bcs += [DirichletBC(W.sub(2), (0, 0), boundary)]
            bcs += [DirichletBC(W.sub(3), (0, 0), boundary)]

        # Boundary conditions for dual mesh displacement
        bcs += [DirichletBC(W.sub(4), (0, 0), DomainBoundary())]

        return bcs
