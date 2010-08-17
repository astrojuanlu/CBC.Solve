"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-17

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

        # Create files for saving to VTK
        if self.save_solution:
            self.u_F_file = File("pvd/u_F.pvd")
            self.p_F_file = File("pvd/p_F.pvd")
            self.U_S_file = File("pvd/U_S.pvd")
            self.P_S_file = File("pvd/P_S.pvd")
            self.U_M_file = File("pvd/U_M.pvd")

        # Create time series for storing solution
        if self.save_series:
            self.u_F_series = TimeSeries("bin/u_F")
            self.p_F_series = TimeSeries("bin/p_F")
            self.U_S_series = TimeSeries("bin/U_S")
            self.P_S_series = TimeSeries("bin/P_S")
            self.U_M_series = TimeSeries("bin/U_M")

        # Store problem
        self.problem = problem

    def solve(self):
        "Solve the primal FSI problem"

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
        self._save_solution(F, S, M)
        self._save_series(F, S, M, 0.0)

        # Time-stepping
        t = dt
        at_end = False
        while True:

            # Display progress
            info("")
            info("-"*80)
            begin("* Starting new time step")
            info_blue("  * t = %g (T = %g, dt = %g)" % (t, T, dt))

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

                # Plot solutions
                if self.plot_solution:
                    plot(u_F,  title="Fluid velocity")
                    plot(U_S0, title="Structure displacement", mode="displacement")
                    plot(U_M,  title="Mesh displacement", mode="displacement")

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
            self._save_solution(F, S, M)
            self._save_series(F, S, M, t)

            # Move to next time step
            F.update(t)
            S.update()
            M.update(t)

            # FIXME: This should be done automatically by the solver
            F.update_extra()

            # Check if we have reached the end time
            if at_end:
                end()
                info("Finished time-stepping")
                break

            # FIXME: Compute these
            Rk = 1.0
            TOL = 0.1
            ST = 1.0

            # Compute new time step
            (dt, at_end) = compute_timestep(Rk, ST, TOL, dt, t, T)
            t += dt

            end()

        # Return solution
        return u_F, p_F, U_S, P_S, U_M

    def _save_solution(self, F, S, M):
        "Save solution to VTK"

        # Check if we should save
        if not self.save_solution: return

        # Get solution
        u_F, p_F = F.solution()
        U_S, P_S = F.solution()
        U_M = M.solution()

        # Save to file
        self.u_F_file << u_F
        self.p_F_file << p_F
        self.U_S_file << U_S
        self.P_S_file << P_S
        self.U_M_file << U_M

    def _save_series(self, F, S, M, t):
        "Save solution to time series"

        # Check if we should save
        if not self.save_series: return

        # Get solution
        u_F, p_F = F.solution()
        U_S, P_S = S.solution()
        U_M = M.solution()

        # Save to series
        self.u_F_series.store(u_F.vector(), t)
        self.p_F_series.store(p_F.vector(), t)
        self.U_S_series.store(U_S.vector(), t)
        self.P_S_series.store(P_S.vector(), t)
        self.U_M_series.store(U_M.vector(), t)
