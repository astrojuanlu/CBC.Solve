"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-04-08

import pylab
from time import time as python_time
from dolfin import *

from cbc.common.utils import timestep_range
from subproblems import *
from adaptivity import *
from storage import *
import sys

def solve_primal(problem, parameters):
    "Solve primal FSI problem"

    # Get parameters
    T = problem.end_time()
    dt = initial_timestep(problem, parameters)
    TOL = parameters["tolerance"]
    w_k = parameters["w_k"]
    w_c = parameters["w_c"]
    save_solution = parameters["save_solution"]
    plot_solution = parameters["plot_solution"]
    uniform_timestep = parameters["uniform_timestep"]
    maxiter = parameters["maximum_iterations"]
    num_smoothings = parameters["num_smoothings"]

    # Create files for saving to VTK
    level = refinement_level()
    if save_solution:
        files = (File("%s/pvd/level_%d/u_F.pvd" % (parameters["output_directory"], level)),
                 File("%s/pvd/level_%d/p_F.pvd" % (parameters["output_directory"], level)),
                 File("%s/pvd/level_%d/U_S.pvd" % (parameters["output_directory"], level)),
                 File("%s/pvd/level_%d/P_S.pvd" % (parameters["output_directory"], level)),
                 File("%s/pvd/level_%d/U_M.pvd" % (parameters["output_directory"], level)))

    # Create time series for storing solution
    primal_series = create_primal_series(parameters)

    # Create time series for dual solution
    if level > 0:
        dual_series = create_dual_series(parameters)
    else:
        dual_series = None

    # Record CPU time
    cpu_time = python_time()

    # Record number of time steps
    timestep_counter = 0

    # Define the three subproblems
    F = FluidProblem(problem)
    S = StructureProblem(problem, parameters)
    M = MeshProblem(problem, parameters)

    # Get solution values
    u_F0, u_F1, p_F0, p_F1 = F.solution_values()
    U_M0, U_M1 = M.solution_values()

    # Extract number of dofs
    num_dofs_FSM = extract_num_dofs(F, S, M)

    # Get initial structure displacement (used for plotting and checking convergence)
    structure_element_degree = parameters["structure_element_degree"]
    V_S = VectorFunctionSpace(problem.structure_mesh(), "CG", structure_element_degree)
    U_S0 = Function(V_S)

    # Save initial solution to file and series
    U = extract_solution(F, S, M)
    if save_solution: _save_solution(U, files)
    write_primal_data(U, 0, primal_series)

    # Initialize adaptive data
    init_adaptive_data(problem, parameters)

    # Initialize time-stepping
    t0 = 0.0
    t1 = dt
    at_end = False

    # Initialize integration of goal functional (assuming M(u) = 0 at t = 0)
    integrated_goal_functional = 0.0
    old_goal_functional = 0.0

    # Time-stepping loop
    while True:

        # Display progress
        info("")
        info("-"*80)
        begin("* Starting new time step")
        info_blue("  * t = %g (T = %g, dt = %g)" % (t1, T, dt))

        # Compute tolerance for FSI iterations
        itertol = compute_itertol(problem, w_c, TOL, dt, t1, parameters)

        # Fixed point iteration on FSI problem
        for iter in range(maxiter):

            info("")
            begin("* Starting nonlinear iteration")

            # Solve fluid subproblem
            begin("* Solving fluid subproblem (F)")
            F.step(dt)
            end()

            # Transfer fluid stresses to structure
            begin("* Transferring fluid stresses to structure (F --> S)")
            Sigma_F = F.compute_fluid_stress(u_F0, u_F1, p_F0, p_F1, U_M0, U_M1)
            S.update_fluid_stress(Sigma_F)
            end()

            # Solve structure subproblem
            begin("* Solving structure subproblem (S)")
            U_S1, P_S1 = S.step(dt)
            end()

            # Transfer structure displacement to fluid mesh
            begin("* Transferring structure displacement to fluid mesh (S --> M)")
            M.update_structure_displacement(U_S1)
            end()

            # Solve mesh equation
            begin("* Solving mesh subproblem (M)")
            M.step(dt)
            end()

            # Transfer mesh displacement to fluid
            begin("* Transferring mesh displacement to fluid (M --> S)")
            F.update_mesh_displacement(U_M1, dt, num_smoothings)
            end()

            # Compute increment of displacement vector
            U_S0.vector().axpy(-1, U_S1.vector())
            increment = norm(U_S0.vector())
            U_S0.vector()[:] = U_S1.vector()[:]

            # Plot solution
            if plot_solution: _plot_solution(u_F1, p_F1, U_S1, U_M1)

            # Check convergence
            if increment < itertol:
                info("")
                info_green("Increment = %g (tolerance = %g), converged after %d iterations" % (increment, itertol, iter + 1))
                info("")
                end()

                # Saving number of FSI iterations
                save_no_FSI_iter(t1, iter + 1, parameters)

                # Evaluate user goal functional
                goal_functional = assemble(problem.evaluate_functional(u_F1, p_F1, U_S1, P_S1, U_M1, dx, dx, dx))

                # Integrate goal functional
                integrated_goal_functional += 0.5 * dt * (old_goal_functional + goal_functional)
                old_goal_functional = goal_functional

                # Save goal functional
                save_goal_functional(t1, goal_functional, integrated_goal_functional, parameters)
                break

            # Check if we have reached the maximum number of iterations
            elif iter == maxiter - 1:
                raise RuntimeError, "FSI iteration failed to converge after %d iterations." % maxiter

            # Print size of increment
            info("")
            info_red("Increment = %g (tolerance = %g), iteration %d" % (increment, itertol, iter + 1))
            end()

        # Save solution and time series to file
        U = extract_solution(F, S, M)
        if save_solution: _save_solution(U, files)
        write_primal_data(U, t1, primal_series)

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
            save_dofs(num_dofs_FSM, timestep_counter, parameters)
            end()
            break

        # Use constant time step
        if uniform_timestep:
            t0 = t1
            t1 = min(t1 + dt, T)
            dt = t1 - t0
            at_end = abs(t1 - T) / T < 100.0*DOLFIN_EPS

        # Compute new adaptive time step
        else:
            Rk = compute_time_residual(primal_series, dual_series, t0, t1, problem, parameters)
            (dt, at_end) = compute_time_step(problem, Rk, TOL, dt, t1, T, w_k, parameters)
            t0 = t1
            t1 = t1 + dt

    # Save final value of goal functional
    save_goal_functional_final(goal_functional, integrated_goal_functional, parameters)

    # Report elapsed time
    info_blue("Primal solution computed in %g seconds." % (python_time() - cpu_time))
    info("")

    # Return solution
    return goal_functional

def _plot_solution(u_F, p_F, U_S, U_M):
    "Plot solution"
    plot(u_F, title="Fluid velocity")
    plot(p_F, title="Fluid pressure")
    #plot(U_S, title="Structure displacement", mode="displacement")
    #plot(U_M, title="Mesh displacement", mode="displacement")

def _save_solution(U, files):
    "Save solution to VTK"
    [files[i] << U[i] for i in range(5)]
