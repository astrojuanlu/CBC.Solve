"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-12-12

import pylab
from time import time
from dolfin import *

from cbc.common.utils import timestep_range 
from subproblems import *
from adaptivity import *
from storage import *
import sys

class PrimalSolver:
    "Primal FSI solver"

    def __init__(self, problem, solver_parameters):
        "Create primal FSI solver"

        # Get solver parameters
        self.plot_solution = solver_parameters["plot_solution"]
        self.save_solution = solver_parameters["save_solution"]
        self.maxiter = solver_parameters["maxiter"]
        self.tolerance = solver_parameters["tolerance"]
        self.uniform_timestep = solver_parameters["uniform_timestep"]
        self.fsi_tolerance = solver_parameters["fixed_point_tol"]

        # Create files for saving to VTK
        if self.save_solution:
            self.files = (File("pvd/u_F.pvd"),
                          File("pvd/p_F.pvd"),
                          File("pvd/U_S.pvd"),
                          File("pvd/P_S.pvd"),
                          File("pvd/U_M.pvd"))

        # Create time series for storing solution
        self.time_series = create_primal_series()

        # Store problem
        self.problem = problem

    def solve(self, ST):
        "Solve the primal FSI problem"

        # Record CPU time
        cpu_time = time()

        # Record number of time steps
        timestep_counter = 0

        # Get problem parameters
        T = self.problem.end_time()
        dt = initial_timestep(self.problem)
        TOL = self.tolerance
        w_c = self.problem.non_galerkin_error_weight()

        # Define the three subproblems
        F = FluidProblem(self.problem)
        S = StructureProblem(self.problem)
        M = MeshProblem(self.problem)

        # Extract number of dofs
        num_dofs_FSM = extract_num_dofs(F, S, M)
      
        # Get initial mesh displacement
        U_M = M.update(0)

        # Get initial structure displacement (used for plotting and checking convergence)
        V_S = VectorFunctionSpace(self.problem.structure_mesh(), "CG", 1)
        U_S0 = Function(V_S)

        # Save initial solution to file and series
        U = extract_solution(F, S, M)
        self._save_solution(U)
        write_primal_data(U, 0, self.time_series)

        # Change time step if uniform
        if self.uniform_timestep:
            dt, dt_range = timestep_range(T, dt)
            
#             # No. time steps
#             time_dofs = len(dt_range)
#             total_dofs = num_dofs_FSM * time_dofs
        
#             # Print
#             f = open("adaptivity/jada.txt", "a")
#             f.write("%g %g %g \n" %(total_dofs, num_dofs_FSM, time_dofs))
#             f.close()
#             exit(True)
        

        # Initialize time-stepping
        t0 = 0.0
        t1 = dt
        at_end = False

        # Time-stepping loop
        while True:

            # Display progress
            info("")
            info("-"*80)
            begin("* Starting new time step")
            info_blue("  * t = %g (T = %g, dt = %g)" % (t1, T, dt))

            # Compute tolerance for FSI iterations
            itertol = compute_itertol(self.problem, w_c, TOL, dt, t1)

            # Fixed point iteration on FSI problem
            for iter in range(self.maxiter):

                info("")
                begin("* Starting nonlinear iteration")
                end()
                
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
                if increment < itertol:
                    info("")
                    info_green("    Increment = %g (tolerance = %g), converged after %d iterations" % \
                                   (increment, itertol, iter + 1))
                    end()

                    # Saving number of FSI iterations 
                    save_no_FSI_iter(t1, iter + 1)
                    
                    # Evaluate user goal functional
                    goal_functional = self.problem.evaluate_functional(u_F, p_F, U_S, P_S, U_M, t1)
                    
                    # Save goal functional
                    save_goal_functional(t1, goal_functional)
                    break

                elif iter == self.maxiter - 1:
                    raise RuntimeError, "FSI iteration failed to converge after %d iterations." % self.maxiter
                else:
                    info("")
                    info_red("    Increment = %g (tolerance = %g), iteration %d" % (increment, itertol, iter + 1))
                    end()

            # Save solution and time series to file
            U = extract_solution(F, S, M)
            self._save_solution(U)
            write_primal_data(U, t1, self.time_series)

            # Move to next time step
            F.update(t1)
            S.update()
            M.update(t1)

            # Update time step counter
            timestep_counter += 1
            
            # FIXME: This should be done automatically by the solver
            F.update_extra()

            # Check if we have reached the end time 
            if at_end:
                info("")
                info_green("Finished time-stepping")
                save_dofs(num_dofs_FSM, timestep_counter)
                end()
                break

            # Use constant time step
            if self.uniform_timestep:
                t0 = t1
                t1 = t1 + dt
                at_end = t1 > T - DOLFIN_EPS
                                
            # Compute new adaptive time step 
            else:
                Rk = compute_time_residual(self.time_series, t0, t1, self.problem)
                (dt, at_end) = compute_time_step(self.problem, Rk, ST, TOL, dt, t1, T)
                t0 = t1
                t1 = t1 + dt

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
