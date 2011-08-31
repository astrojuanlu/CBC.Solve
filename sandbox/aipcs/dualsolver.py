"This module implements the dual FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-08-31

from time import time as python_time
from dolfin import *
from spaces import *
from storage import *
from dualproblem import *
from adaptivity import *

def solve_dual(problem, parameters):
    "Solve dual FSI problem"

    # Get parameters
    T = problem.end_time()
    Omega = problem.mesh()
    save_solution = parameters["save_solution"]
    plot_solution = parameters["plot_solution"]

    # Create files for saving to VTK
    files = None
    level = refinement_level()
    if save_solution:
        Z_F_file = File("%s/pvd/level_%d/Z_F.pvd" % (parameters["output_directory"], level))
        Y_F_file = File("%s/pvd/level_%d/Y_F.pvd" % (parameters["output_directory"], level))
        files = [Z_F_file, Y_F_file]

    # Create time series for storing solution
    primal_series = create_primal_series(parameters)
    dual_series = create_dual_series(parameters)

    # Record CPU time
    cpu_time = python_time()

    # Create mixed function space
    W = create_dual_space(Omega, parameters)

    # Create test and trial functions
    (v_F, q_F) = TestFunctions(W)
    (Z_F, Y_F) = TrialFunctions(W)

    # Create dual functions
    Z0, (Z_F0, Y_F0) = create_dual_functions(Omega, parameters)
    Z1, (Z_F1, Y_F1) = create_dual_functions(Omega, parameters)

    # Create primal functions
    U_F0, P_F0 = U0 = create_primal_functions(Omega, parameters)
    U_F1, P_F1 = U1 = create_primal_functions(Omega, parameters)

    # Create time step (value set in each time step)
    k = Constant(0.0)

    # Create variational forms for dual problem
    A, L, cd, efd, ifd = create_dual_forms(Omega, k, problem,
                                           v_F,  q_F,
                                           Z_F,  Y_F,
                                           Z_F0, Y_F0,
                                           U_F0, P_F0,
                                           U_F1, P_F1)

    # Create dual boundary conditions
    bcs = create_dual_bcs(problem, W)

    # Write initial value for dual
    write_dual_data(Z1, T, dual_series)

    # Time-stepping
    T  = problem.end_time()
    timestep_range = read_timestep_range(T, primal_series)
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
        read_primal_data(U0, t0, Omega, primal_series, parameters)
        read_primal_data(U1, t1, Omega, primal_series, parameters)

        # Assemble matrix
        info("Assembling matrix")
        matrix = assemble(A, cell_domains=cd,
                          exterior_facet_domains=efd, interior_facet_domains=ifd)

        # Assemble vector
        info("Assembling vector")
        vector = assemble(L, cell_domains=cd,
                          exterior_facet_domains=efd, interior_facet_domains=ifd)

        # Apply boundary conditions
        info("Applying boundary conditions")
        for bc in bcs:
            bc.apply(matrix, vector)

        # Remove inactive dofs
        matrix.ident_zeros()

        # Solve linear system
        solve(matrix, Z0.vector(), vector)

        # Save and plot solution
        if save_solution: _save_solution(Z0, files)
        write_dual_data(Z0, t0, dual_series)
        if plot_solution: _plot_solution(Z_F0, Y_F0)

        # Copy solution to previous interval (going backwards in time)
        Z1.assign(Z0)

        end()

    # Report elapsed time
    info_blue("Dual solution computed in %g seconds." % (python_time() - cpu_time))

def _save_solution(Z, files):
    "Save solution to VTK"

    # Extract sub functions (shallow copy)
    (Z_F, Y_F) = Z.split()

    # Save to file
    files[0] << Z_F
    files[1] << Y_F

def _plot_solution(Z_F, Y_F):
    "Plot solution"

    # Plot solution
    plot(Z_F, title="Dual fluid velocity")
    plot(Y_F, title="Dual fluid pressure")
