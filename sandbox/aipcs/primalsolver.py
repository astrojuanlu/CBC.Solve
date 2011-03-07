"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-03-07

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

    # Create files for saving to VTK
    level = refinement_level()
    if save_solution:
        files = (File("%s/pvd/level_%d/u_F.pvd" % (parameters["output_directory"], level)),
                 File("%s/pvd/level_%d/p_F.pvd" % (parameters["output_directory"], level)))

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
    F = FluidProblem(problem, parameters)

    # Get solution values
    u_F0, u_F1, p_F0, p_F1 = F.solution_values()

    # Extract number of dofs
    num_dofs = u_F0.vector().size() + p_F0.vector().size()

    # Save initial solution to file and series
    U = extract_solution(F)
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

        # Solve fluid subproblem
        begin("* Solving fluid subproblem (F)")
        F.step(dt)
        end()

        # Evaluate user goal functional
        M, cd, efd, ifd = problem.evaluate_functional(u_F1, p_F1)
        goal_functional = assemble(M, cell_domains=cd, exterior_facet_domains=efd, interior_facet_domains=ifd)

        # Integrate goal functional
        integrated_goal_functional += 0.5 * dt * (old_goal_functional + goal_functional)
        old_goal_functional = goal_functional

        # Save goal functional
        save_goal_functional(t1, goal_functional, integrated_goal_functional, parameters)

        # Save solution and time series to file
        U = extract_solution(F)
        if save_solution: _save_solution(U, files)
        write_primal_data(U, t1, primal_series)

        # Move to next time step
        F.update(t1)

        # Update time step counter
        timestep_counter += 1

        # Check if we have reached the end time
        if at_end:
            info("")
            info_green("Finished time-stepping")
            save_dofs(num_dofs, timestep_counter, parameters)
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

        end()

    # Save final value of goal functional
    save_goal_functional_final(goal_functional, integrated_goal_functional, parameters)

    # Report elapsed time
    info_blue("Primal solution computed in %g seconds." % (python_time() - cpu_time))
    info("")

    # Return solution
    return goal_functional

def _save_solution(U, files):
    "Save solution to VTK"
    [files[i] << U[i] for i in range(2)]
