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
            self.Z_F_file = File("pvd/Z_F.pvd")
            self.Y_F_file = File("pvd/Y_F.pvd")
            self.Z_S_file = File("pvd/Z_S.pvd")
            self.Y_S_file = File("pvd/Y_S.pvd")
            self.Z_M_file = File("pvd/Z_M.pvd")
            self.Y_M_file = File("pvd/Y_M.pvd")

        # Create time series for storing solution
        if self.save_series:
            self.Z_F_series = TimeSeries("bin/Z_F")
            self.Y_F_series = TimeSeries("bin/Y_F")
            self.Z_S_series = TimeSeries("bin/Z_S")
            self.Y_S_series = TimeSeries("bin/Y_S")
            self.Z_M_series = TimeSeries("bin/Z_M")
            self.Y_M_series = TimeSeries("bin/Y_M")

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

        # Extract sub functions (shallow copy)
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

            # FIXME: Missing exterior_facet_domains, need to figure
            # FIXME: out why they are needed

            # Assemble matrix
            matrix = assemble(A,
                              cell_domains=self.problem.cell_domains,
                              interior_facet_domains=self.problem.fsi_boundary)

            # Assemble vector
            vector = assemble(L,
                              cell_domains=self.problem.cell_domains,
                              interior_facet_domains=self.problem.fsi_boundary)

            # Apply boundary conditions
            for bc in bcs:
                bc.apply(matrix, vector)

            # Remove inactive dofs
            matrix.ident_zeros()

            # Solve linear system
            solve(matrix, Z0.vector(), vector)

            # Extract sub functions (deep copy)
            (Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0) = Z0.split(True)

            # Save and plot solution
            self._save_solution(Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0)
            self._save_series(Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0, t0)
            self._plot_solution(Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0)

            # Copy solution to previous interval (going backwards in time)
            Z1.assign(Z0)

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

    def _save_solution(self, Z_F, Y_F, Z_S, Y_S, Z_M, Y_M):
        "Save solution to VTK"

        # Check if we should save
        if not self.save_solution: return

        # Save to file
        self.Z_F_file << Z_F
        self.Y_F_file << Y_F
        self.Z_S_file << Z_S
        self.Y_S_file << Y_S
        self.Z_M_file << Z_M
        self.Y_M_file << Y_M

    def _save_series(self, Z_F, Y_F, Z_S, Y_S, Z_M, Y_M, t):
        "Save solution to time series"

        # Check if we should save
        if not self.save_series: return

        # Save to series
        self.Z_F_series.store(Z_F.vector(), t)
        self.Y_F_series.store(Y_F.vector(), t)
        self.Z_S_series.store(Z_S.vector(), t)
        self.Y_S_series.store(Y_S.vector(), t)
        self.Z_M_series.store(Z_M.vector(), t)
        self.Y_M_series.store(Y_M.vector(), t)

    def _plot_solution(self, Z_F, Y_F, Z_S, Y_S, Z_M, Y_M):
        "Save solution to time series"

        # Check if we should plot
        if not self.plot_solution: return

        # Plot solution
        plot(Z_F, title="Dual fluid velocity")
        plot(Y_F, title="Dual fluid pressure")
        plot(Z_S, title="Dual displacement")
        plot(Y_S, title="Dual displacement velocity")
        plot(Z_M, title="Dual mesh displacement")
        plot(Y_M, title="Dual mesh Lagrange multiplier")
