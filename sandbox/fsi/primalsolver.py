"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-11

from dolfin import *
from subproblems import *
from adaptivity import *

def solve_primal(problem, solver_parameters):
    "Solve the primal FSI problem"

    # Get problem parameters
    T = problem.end_time()
    dt = problem.initial_time_step()

    # Get solver parameters
    plot_solution = solver_parameters["plot_solution"]
    save_solution = solver_parameters["save_solution"]
    save_series = solver_parameters["save_series"]
    maxiter = solver_parameters["maxiter"]
    itertol = solver_parameters["itertol"]

    # Create files for saving to VTK
    if save_solution:
        file_u_F = File("pvd/u_F.pvd")
        file_p_F = File("pvd/p_F.pvd")
        file_U_S = File("pvd/U_S.pvd")
        file_P_S = File("pvd/P_S.pvd")
        file_U_M = File("pvd/U_M.pvd")

    # Create time series for storing solution
    if save_series:
        u_F_series = TimeSeries("bin/u_F")
        p_F_series = TimeSeries("bin/p_F")
        U_S_series = TimeSeries("bin/U_S")
        P_S_series = TimeSeries("bin/P_S")
        U_M_series = TimeSeries("bin/U_M")

    # Define the three subproblems
    F = FluidProblem(problem)
    S = StructureProblem(problem)
    M = MeshProblem(problem)

    # Get initial mesh displacement
    U_M = M.update(0)

    # Get initial structure displacement (used for plotting and checking convergence)
    V_S = VectorFunctionSpace(problem.structure_mesh(), "CG", 1)
    U_S0 = Function(V_S)

    # Time-stepping
    t = dt
    at_end = False
    time_data = []
    while True:

        info("")
        info("-"*80)
        begin("* Starting new time step")
        info_blue("  * t = %g (T = %g, dt = %g)" % (t, T, dt))

        # Fixed point iteration on FSI problem
        for iter in range(maxiter):

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
            U_S, P_S = S.step(dt).split(True)
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
            if plot_solution:
                plot(u_F,  title="Fluid velocity")
                plot(U_S0, title="Structure displacement", mode="displacement")
                plot(U_M,  title="Mesh displacement", mode="displacement")
                #plot(F.w,  title="Mesh velocity")

            # Check convergence
            if increment < itertol:
                info("")
                info_green("    Increment = %g (tolerance = %g), converged after %d iterations" % \
                               (increment, itertol, iter + 1))
                end()
                break
            elif iter == maxiter - 1:
                raise RuntimeError, "FSI iteration failed to converge after %d iterations." % maxiter
            else:
                info("")
                info_red("    Increment = %g (tolerance = %g), iteration %d" % (increment, itertol, iter + 1))
                end()

        # Evaluate user functional
        problem.evaluate_functional(u_F, p_F, U_S, P_S, U_M, at_end)

        # Save solution in VTK format
        if save_solution:
            file_u_F << u_F
            file_p_F << p_F
            file_U_S << U_S
            file_P_S << P_S
            file_U_M << U_M

        # Save solution in time series
        if save_series:
            u_F_series.store(u_F.vector(), t)
            p_F_series.store(p_F.vector(), t)
            U_S_series.store(U_S.vector(), t)
            P_S_series.store(P_S.vector(), t)
            U_M_series.store(U_M.vector(), t)

        # Store time and time steps
        time_data.append((t, dt))

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

        end()
