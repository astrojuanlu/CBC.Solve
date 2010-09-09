"This module implements the dual FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-09

from time import time
from dolfin import *
from storage import *
from dualproblem import dual_forms

# FIXME: alpha_M missing

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

        # Store problem and parameters
        self.problem = problem
        self.parameters = solver_parameters

    def solve(self):
        "Solve the dual FSI problem"

        # Record CPU time
        cpu_time = time()

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

        # Create dual functions
        Z0, (Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0) = init_dual_data(Omega)
        Z1, (Z_F1, Y_F1, Z_S1, Y_S1, Z_M1, Y_M1) = init_dual_data(Omega)

        # Create primal functions
        U_F0, P_F0, U_S0, P_S0, U_M0 = init_primal_data(Omega)
        U_F1, P_F1, U_S1, P_S1, U_M1 = init_primal_data(Omega)

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
        timestep_range = read_timestep_range(self.problem)
        for i in reversed(range(len(timestep_range) - 1)):

            # Get current time and time step
            t0 = timestep_range[i]
            t1 = timestep_range[i + 1]
            T  = self.problem.end_time()
            dt = t1 - t0
            k.assign(dt)

            # Display progress
            info("")
            info("-"*80)
            begin("* Starting new time step")
            info_blue("  * t = %g (T = %g, dt = %g)" % (t0, T, dt))

            # Read primal data
            #self._read_primal_data(U_F0, P_F0, U_S0, P_S0, U_M0, t0)
            read_primal_data(U_F0, P_F0, U_S0, P_S0, U_M0, t0,
                             Omega, Omega_F, Omega_S)

            # FIXME: Missing exterior_facet_domains, need to figure
            # FIXME: out why they are needed

            # Assemble matrix
            info("Assembling matrix")
            matrix = assemble(A,
                              cell_domains=self.problem.cell_domains,
                              interior_facet_domains=self.problem.fsi_boundary)

            # Assemble vector
            info("Assembling vector")
            vector = assemble(L,
                              cell_domains=self.problem.cell_domains,
                              interior_facet_domains=self.problem.fsi_boundary)

            # Apply boundary conditions
            info("Applying boundary conditions")
            for bc in bcs:
                bc.apply(matrix, vector)

            # Remove inactive dofs
            matrix.ident_zeros()

            # Solve linear system
            solve(matrix, Z0.vector(), vector)

            # Save and plot solution
            self._save_solution(Z0)
            write_dual_data(Z0, t0, self.parameters)
            self._plot_solution(Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0)

            # Copy solution to previous interval (going backwards in time)
            Z1.assign(Z0)

            end()

        # Report elapsed time
        info_blue("Dual solution computed in %g seconds." % (time() - cpu_time))

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

    def _save_solution(self, Z):
        "Save solution to VTK"

        # Check if we should save
        if not self.save_solution: return

        # Extract sub functions (shallow copy)
        (Z_F, Y_F, Z_S, Y_S, Z_M, Y_M) = Z.split()

        # Save to file
        self.Z_F_file << Z_F
        self.Y_F_file << Y_F
        self.Z_S_file << Z_S
        self.Y_S_file << Y_S
        self.Z_M_file << Z_M
        self.Y_M_file << Y_M

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
