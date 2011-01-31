"This module implements the dual FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-16

from time import time
from dolfin import *
from spaces import *
from storage import *
from dualproblem import *

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
            self.y_file = File("pvd/z.pvd")
            self.z_file = File("pvd/y.pvd")

        # Create time series for storing solution
        self.primal_series = create_primal_series()
        self.dual_series = create_dual_series()

        # Store problem and parameters
        self.problem = problem
        self.parameters = solver_parameters

    def solve(self):
        "Solve the dual fluid problem"

        # Record CPU time
        cpu_time = time()

        # Get problem parameters
        T = self.problem.end_time()
        Omega = self.problem.fluid_mesh()

        # Create function spaces
        W = create_dual_space(Omega)

        # Create test and trial functions
        (v, q) = TestFunctions(W)
        (z, y) = TrialFunctions(W)

        # Create dual functions (dual_sol contains both z and y)
        dual_sol_0, (z0, y0) = create_dual_functions(Omega)
        dual_sol_1, (z1, y1) = create_dual_functions(Omega)
        
#         Z0, (Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0) = create_dual_functions(Omega)
#         Z1, (Z_F1, Y_F1, Z_S1, Y_S1, Z_M1, Y_M1) = create_dual_functions(Omega)

        # Create primal functions used in the dual form
        uh0, ph0 = primal_sol_0 = create_primal_functions(Omega)
        uh1, ph1 = primal_sol_1 = create_primal_functions(Omega)
#         U_F0, P_F0, U_S0, P_S0, U_M0 = U0 = create_primal_functions(Omega)
#         U_F1, P_F1, U_S1, P_S1, U_M1 = U1 = create_primal_functions(Omega)

        # Create time step (value set in each time step)
        k = Constant(0.0)

        # Create variational forms for dual problem
        A, L = create_dual_forms(Omega, k, self.problem,  
                                 v, q, z, y, z0, 
                                 uh0, ph0, uh1, ph1)

        # Create dual boundary conditions
        bcs = self._create_boundary_conditions(W)

        # Time-stepping
        T  = self.problem.end_time()
        timestep_range = read_timestep_range(T, self.primal_series)
        for i in reversed(range(len(timestep_range) - 1)):

            # Get current time and time step
            t0 = timestep_range[i]
            t1 = timestep_range[i + 1]
            dt = t1 - t0
            k.assign(dt)

            # Display progress
            info("")
            info("-"*80)
            begin("* Starting new time step")
            info_blue("  * t = %g (T = %g, dt = %g)" % (t0, T, dt))

            # Read primal data
            read_primal_data(primal_sol_0, t0, Omega, self.primal_series)            
            read_primal_data(primal_sol_1, t1, Omega, self.primal_series)            

#             read_primal_data(U0, t0, Omega, Omega_F, Omega_S, self.primal_series)
#             read_primal_data(U1, t1, Omega, Omega_F, Omega_S, self.primal_series)

            # Assemble matrix
            info("Assembling matrix")
            matrix = assemble(A)

            # Assemble vector
            info("Assembling vector")
            vector = assemble(L)

            # Apply boundary conditions
            info("Applying boundary conditions")
            for bc in bcs:
                bc.apply(matrix, vector)

            # Solve linear system
            solve(matrix, dual_sol_0.vector(), vector)

            # Save and plot solution
            self._save_solution(dual_sol_0)
            write_dual_data(dual_sol_0, t0, self.dual_series)
            self._plot_solution(z0, y0)

            # Copy solution to previous interval (going backwards in time)
            dual_sol_1.assign(dual_sol_0)
#            Z1.assign(Z0)
            
            end()

        # Report elapsed time
        info_blue("Dual solution computed in %g seconds." % (time() - cpu_time))

    def _create_boundary_conditions(self, W):
        "Create boundary conditions for dual problem"

        bcs = []

        # Boundary conditions for dual velocity
        for boundary in self.problem.velocity_dirichlet_boundaries():
            bcs += [DirichletBC(W.sub(0), (0.0 , 0.0), boundary)]

        # Boundary conditions for dual pressure
        for boundary in self.problem.pressure_dirichlet_boundaries():
            bcs += [DirichletBC(W.sub(1), 0.0, boundary)]

        return bcs

    def _save_solution(self, dual_sol):
        "Save solution to VTK"

        # Check if we should save
        if not self.save_solution: return

        # Extract sub functions (shallow copy)
        (z0, y0) = dual_sol.split()

        # Save to file
        self.z_file << z0
        self.y_file << y0

    def _plot_solution(self, z, y):
        "Plot solution"

        # Check if we should plot
        if not self.plot_solution: return

        # Plot solution
        plot(z, title="Dual fluid velocity")
        plot(y, title="Dual fluid pressure")
