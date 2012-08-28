"This module implements the dual FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-05-03

from time import time as python_time
from dolfin import *
from spaces import *
from storage import *
from adaptivity import *

#G.B. In this implementation dsF is the do nothing boundary
# Original implementation
#from dualproblem import *

#G.B. In this implementation dsF is the fluid Neumann boundary
# New implementation by Gabriel
from new_dualproblem import *

def solve_dual(problem, parameters):
    "Solve dual FSI problem"

    # Get parameters
    T = problem.end_time()
    Omega = problem.mesh()
    Omega_F = problem.fluid_mesh()
    Omega_S = problem.structure_mesh()
    save_solution = parameters["save_solution"]
    plot_solution = parameters["plot_solution"]

    # Create files for saving to VTK
    files = None
    level = refinement_level()
    if save_solution:
        Z_F_file = File("%s/pvd/level_%d/Z_F.pvd" % (parameters["output_directory"], level))
        Y_F_file = File("%s/pvd/level_%d/Y_F.pvd" % (parameters["output_directory"], level))
        X_F_file = File("%s/pvd/level_%d/X_F.pvd" % (parameters["output_directory"], level))
        Z_S_file = File("%s/pvd/level_%d/Z_S.pvd" % (parameters["output_directory"], level))
        Y_S_file = File("%s/pvd/level_%d/Y_S.pvd" % (parameters["output_directory"], level))
        Z_M_file = File("%s/pvd/level_%d/Z_M.pvd" % (parameters["output_directory"], level))
        Y_M_file = File("%s/pvd/level_%d/Y_M.pvd" % (parameters["output_directory"], level))
        files = [Z_F_file, Y_F_file,
                 X_F_file, Z_S_file,
                 Y_S_file, Z_M_file,
                 Y_M_file]

    # Create time series for storing solution
    primal_series = create_primal_series(parameters)
    dual_series = create_dual_series(parameters)

    # Record CPU time
    cpu_time = python_time()

    # Create mixed function space
    W = create_dual_space(Omega, parameters)

    # Create test and trial functions
    (v_F, q_F, s_F, v_S, q_S, v_M, q_M) = TestFunctions(W)
    (Z_F, Y_F, X_F, Z_S, Y_S, Z_M, Y_M) = TrialFunctions(W)

    # Create dual functions
    #These will effect the forms
    Z0, (Z_F0, Y_F0, X_F0, Z_S0, Y_S0, Z_M0, Y_M0) = create_dual_functions(Omega, parameters)

    #These are used for data storage
    Z1, (Z_F1, Y_F1, X_F1, Z_S1, Y_S1, Z_M1, Y_M1) = create_dual_functions(Omega, parameters)

    # Create primal functions
    U_F0, P_F0, U_S0, P_S0, U_M0 = U0 = create_primal_functions(Omega, parameters)
    U_F1, P_F1, U_S1, P_S1, U_M1 = U1 = create_primal_functions(Omega, parameters)

    # Create time step (value set in each time step)
    k = Constant(0.0)

    # Create variational forms for dual problem
    A, L = create_dual_forms(Omega_F, Omega_S, k, problem,
                             v_F,  q_F,  s_F,  v_S,  q_S,  v_M,  q_M,
                             Z_F,  Y_F,  X_F,  Z_S,  Y_S,  Z_M,  Y_M,
                             Z_F0, Y_F0, X_F0, Z_S0, Y_S0, Z_M0, Y_M0,
                             U_F0, P_F0, U_S0, P_S0, U_M0,
                             U_F1, P_F1, U_S1, P_S1, U_M1,parameters)

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
        info_blue("* t = %g (T = %g, dt = %g)" % (t0, T, dt))

        # Read primal data
        read_primal_data(U0, t0, Omega, Omega_F, Omega_S, primal_series, parameters)
        read_primal_data(U1, t1, Omega, Omega_F, Omega_S, primal_series, parameters)

        # GB: In the Analytic problem there are no do nothing fluid boundaries. I am not
        # this is reflected here in the meshfunctions and their facet numberings.
        # Assemble matrix
        info("Assembling matrix")
        matrix = assemble(A,
                          cell_domains=problem.cell_domains,
                          exterior_facet_domains=problem.fsi_boundary,
                          interior_facet_domains=problem.fsi_boundary)

        # Assemble vector
        info("Assembling vector")
        vector = assemble(L,
                          cell_domains=problem.cell_domains,
                          exterior_facet_domains=problem.fsi_boundary,
                          interior_facet_domains=problem.fsi_boundary)

        # Remove inactive dofs
        info("Removing inactive dofs")
        matrix.ident_zeros()

        # Apply boundary conditions
        info("Applying boundary conditions")
        for bc in bcs:
            bc.apply(matrix, vector)

        # Solve linear system
        solve(matrix, Z0.vector(), vector)
        info("Solved linear system: ||Z|| = " + str(Z0.vector().norm("l2")))

        # Save and plot solution
        if save_solution: _save_solution(Z0, files)
        write_dual_data(Z0, t0, dual_series)
        if plot_solution: _plot_solution(Z_F0, Y_F0, X_F0, Z_S0, Y_S0, Z_M0, Y_M0)

        # Copy solution to previous interval (going backwards in time)
        Z1.assign(Z0)
        end()

    # Report elapsed time
    info_blue("Dual solution computed in %g seconds." % (python_time() - cpu_time))

def _save_solution(Z, files):
    "Save solution to VTK"

    # Extract sub functions (shallow copy)
    (Z_F, Y_F, X_F, Z_S, Y_S, Z_M, Y_M) = Z.split()

    # Save to file
    files[0] << Z_F
    files[1] << Y_F
    files[2] << X_F
    files[3] << Z_S
    files[4] << Y_S
    files[5] << Z_M
    files[6] << Y_M

def _plot_solution(Z_F, Y_F, X_F, Z_S, Y_S, Z_M, Y_M):
    "Plot solution"

    # Plot solution
    plot(Z_F, title="Dual fluid velocity")
    plot(Y_F, title="Dual fluid pressure")
    plot(X_F, title="Dual fluid Lagrange multiplier")
    plot(Z_S, title="Dual displacement", mode = "displacement")
    plot(Y_S, title="Dual displacement velocity")
    plot(Z_M, title="Dual mesh displacement")
    plot(Y_M, title="Dual mesh Lagrange multiplier")
