"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-16

import pylab
from time import time as python_time
from dolfin import *

from cbc.common.utils import timestep_range
from subproblems import *
from adaptivity import *
from storage import *
import sys

class PrimalSolver:
    "Primal fluid solver"

    def __init__(self, problem, solver_parameters):
        "Create primal FSI solver"

        # Get solver parameters
        self.plot_solution = solver_parameters["plot_solution"]
        self.save_solution = solver_parameters["save_solution"]
        self.tolerance = solver_parameters["tolerance"]
        self.uniform_timestep = solver_parameters["uniform_timestep"]

        # Create files for saving to VTK
        if self.save_solution:
            self.files = (File("pvd/u.pvd"),
                          File("pvd/p.pvd"))

        # Create time series for storing solution
        self.time_series = create_primal_series()

        # Store problem
        self.problem = problem

    def solve(self, ST):
        "Solve the primal FSI problem"

        # Record CPU time
        cpu_time = python_time()

        # Record number of time steps
        timestep_counter = 0

        # Get problem parameters
        T = self.problem.end_time()
        dt = initial_timestep(self.problem)
        TOL = self.tolerance
        w_c = self.problem.non_galerkin_error_weight()

        # Define the fluid problem
        F = FluidProblem(self.problem)

#         # Extract number of dofs
#         num_dofs_FSM = extract_num_dofs(F)

        # Save initial solution to file and series
        U = extract_solution(F)
        self._save_solution(U)
        write_primal_data(U, 0, self.time_series)

        # Change time step if uniform
        if self.uniform_timestep:
            dt, dt_range = timestep_range(T, dt)

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

            # Solve fluid problem
            u, p = F.step(dt)

            # Save solution and time series to file
            U = extract_solution(F)
            self._save_solution(U)
            write_primal_data(U, t1, self.time_series)

            # Evaluate user goal functional
            goal_functional = self.problem.evaluate_functional(u, p, t1)

            # Integrate goal functional
            integrated_goal_functional += 0.5 * dt * (old_goal_functional + goal_functional)
            old_goal_functional = goal_functional

            # Save goal functional
            save_goal_functional(t1, goal_functional, integrated_goal_functional)#, parameters)

            # Move to next time step
            F.update(t1)

            # Update time step counter
            timestep_counter += 1
            end()

            # Check if we have reached the end time
            if at_end:
                info("")
                info_green("Finished time-stepping")
#                save_dofs(num_dofs_FSM, timestep_counter)
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
        info_blue("Primal solution computed in %g seconds." % (python_time() - cpu_time))

        # Return solution
        return u, p

    def _plot_solution(self, u, p):
        "Plot solution"

        # Check if we should plot
        if not self.plot_solution: return

        # Plot
        plot(u, title="Fluid velocity")
        plot(p, title="Fluid pressure")

    def _save_solution(self, U):
        "Save solution to VTK"

        # Check if we should save
        if not self.save_solution: return

        # Save to file
        for i in range(2):
            self.files[i] << U[i]
