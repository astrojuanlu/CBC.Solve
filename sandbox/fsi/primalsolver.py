"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-16

import pylab
from time import time
from dolfin import *

from subproblems import *
from adaptivity import *

class PrimalSolver:
    "Primal FSI solver"

    def __init__(self, problem, solver_parameters):
        "Create primal FSI solver"

        # Get solver parameters
        self.plot_solution = solver_parameters["plot_solution"]
        self.save_solution = solver_parameters["save_solution"]
        self.save_series = solver_parameters["save_series"]
        self.maxiter = solver_parameters["maxiter"]
        self.itertol = solver_parameters["itertol"]
        self.tolerance = solver_parameters["tolerance"]

        # Create files for saving to VTK
        if self.save_solution:
            self.files = (File("pvd/u_F.pvd"),
                          File("pvd/p_F.pvd"),
                          File("pvd/U_S.pvd"),
                          File("pvd/P_S.pvd"),
                          File("pvd/U_M.pvd"))

        # Create time series for storing solution
        if self.save_series:
            self.time_series = (TimeSeries("bin/u_F"),
                                TimeSeries("bin/p_F"),
                                TimeSeries("bin/U_S"),
                                TimeSeries("bin/P_S"),
                                TimeSeries("bin/U_M"))

        # Store problem
        self.problem = problem

    def solve(self, ST):
        "Solve the primal FSI problem"

        # Record CPU time
        cpu_time = time()

        # Get problem parameters
        T = self.problem.end_time()
        dt = self.problem.initial_time_step()

        # Define the three subproblems
        F = FluidProblem(self.problem)
        S = StructureProblem(self.problem)
        M = MeshProblem(self.problem)

        # Get initial mesh displacement
        U_M = M.update(0)

        # Get initial structure displacement (used for plotting and checking convergence)
        V_S = VectorFunctionSpace(self.problem.structure_mesh(), "CG", 1)
        U_S0 = Function(V_S)

        # Save initial solution to file and series
        U = extract_solution(F, S, M)
        self._save_solution(U)
        self._save_series(U, 0)

        # Time-stepping
        t0 = 0.0
        t1 = dt
        at_end = False
        while True:

            # Display progress
            info("")
            info("-"*80)
            begin("* Starting new time step")
            info_blue("  * t = %g (T = %g, dt = %g)" % (t1, T, dt))

            # Fixed point iteration on FSI problem
            for iter in range(self.maxiter):

                info("")
                begin("* Starting nonlinear iteration")

                # Solve fluid subproblem
                begin("* Solving fluid subproblem (F)")
                u_F, p_F = F.step(dt)
                end()

                # Transfer fluid stresses to structure
                begin("* Transferring fluid stresses to structure (F --> S)")
                Sigma_F = F.compute_fluid_stress(u_F, p_F, U_M)
                S.update_fluid_stress(Sigma_F)
                end()

                # Solve structure subproblem
                begin("* Solving structure subproblem (S)")
                U_S, P_S = S.step(dt)
                end()

                # Transfer structure displacement to fluid mesh
                begin("* Transferring structure displacement to fluid mesh (S --> M)")
                M.update_structure_displacement(U_S)
                end()

                # Solve mesh equation
                begin("* Solving mesh subproblem (M)")
                U_M = M.step(dt)
                end()

                # Transfer mesh displacement to fluid
                begin("* Transferring mesh displacement to fluid (M --> S)")
                F.update_mesh_displacement(U_M, dt)
                end()

                # Compute increment of displacement vector
                U_S0.vector().axpy(-1, U_S.vector())
                increment = norm(U_S0.vector())
                U_S0.vector()[:] = U_S.vector()[:]

                # Plot solution
                self._plot_solution(u_F, U_S0, U_M)

                # Check convergence
                if increment < self.itertol:
                    info("")
                    info_green("    Increment = %g (tolerance = %g), converged after %d iterations" % \
                                   (increment, self.itertol, iter + 1))
                    end()
                    break
                elif iter == self.maxiter - 1:
                    raise RuntimeError, "FSI iteration failed to converge after %d iterations." % self.maxiter
                else:
                    info("")
                    info_red("    Increment = %g (tolerance = %g), iteration %d" % (increment, self.itertol, iter + 1))
                    end()

            # Evaluate user functional
            self.problem.evaluate_functional(u_F, p_F, U_S, P_S, U_M, at_end)

            # Save solution and time series to file
            U = extract_solution(F, S, M)
            self._save_solution(U)
            self._save_series(U, t1)

            # Move to next time step
            F.update(t1)
            S.update()
            M.update(t1)

            # FIXME: This should be done automatically by the solver
            F.update_extra()

            # Check if we have reached the end time
            if at_end:
                end()
                info("Finished time-stepping")
                break

            # Compute new time step
            TOL = self.tolerance
            Rk = compute_time_residual(self.time_series, t0, t1, self.problem)
            (dt, at_end) = compute_timestep(Rk, ST, TOL, dt, t1, T)
            t0 = t1
            t1 = t1 + dt

            end()

        # Report elapsed time
        info_blue("Primal solution computed in %g seconds." % (time() - cpu_time))

        # Return solution
        return u_F, p_F, U_S, P_S, U_M

    def _plot_solution(self, u_F, U_S0, U_M):
        "Plot solution"

        # Check if we should plot
        if not self.plot_solution: return

        # Plot
        plot(u_F,  title="Fluid velocity")
        plot(U_S0, title="Structure displacement", mode="displacement")
        plot(U_M,  title="Mesh displacement", mode="displacement")

    def _save_solution(self, U):
        "Save solution to VTK"

        # Check if we should save
        if not self.save_solution: return

        # Save to file
        for i in range(5):
            self.files[i] << U[i]

    def _save_series(self, U, t):
        "Save solution to time series"

        # Check if we should save
        if not self.save_series: return

        # Save to series
        for i in range(5):
            self.time_series[i].store(U[i].vector(), t)
