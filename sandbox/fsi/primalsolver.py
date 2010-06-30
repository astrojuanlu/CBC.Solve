"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-30

from dolfin import *
from subproblems import *
from adaptivity import *

def solve_primal(problem):
    "Solve the primal FSI problem"

    # Get problem parameters
    T = problem.end_time()
    dt = problem.initial_time_step()

    # Get solver parameters
    maxiter = problem.parameters["solver_parameters"]["maxiter"]
    itertol = problem.parameters["solver_parameters"]["itertol"]
    plot_solution = problem.parameters["solver_parameters"]["plot_solution"]
    save_solution = problem.parameters["solver_parameters"]["save_solution"]
    store_solution_data = problem.parameters["solver_parameters"]["store_solution_data"]

    # Create time series for storing solution
    u_F_series = TimeSeries("u_F")
    p_F_series = TimeSeries("p_F")
    U_S_series = TimeSeries("U_S")
    P_S_series = TimeSeries("P_S")
    U_M_series = TimeSeries("U_M")

    # Create files for saving to VTK
    if save_solution:
        file_u_F = File("u_F.pvd")
        file_p_F = File("p_F.pvd")
        file_U_S = File("U_S.pvd")
        file_P_S = File("P_S.pvd")
        file_U_M = File("U_M.pvd")

    # FIXME: Problem-specific, should not be here
    #disp_vs_t = open("displacement_nx_dt_T_smooth"+ "_" + str(nx) + "_"  +  str(dt) + "_" + str(T) + "_"+ str(mesh_smooth), "w")
    #convergence_data = open("convergence_nx_dt_T_smooth" + "_" + str(nx)  +  "_"  +  str(dt) + "_" + str(T) +  "_" + str(mesh_smooth), "w")

    # Define the three subproblems
    F = FluidProblem(problem)
    S = StructureProblem(problem)
    M = MeshProblem(problem)

    # Prepare some initial variables
    U_M = M.update(0)
    U_S_old = Vector(2*problem.structure_mesh().num_vertices())

    # Storing of adaptive data
    times = []
    timesteps = []

    # Time-stepping
    t = dt
    at_end = False
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

            # Plot solutions
            if plot_solution:
                plot(u_F, title="Fluid velocity")
                plot(U_S, title="Structure displacement", mode="displacement")
                plot(U_M, title="Mesh displacement", mode="displacement")
                plot(F.w, title="Mesh velocity")

            # Compute increment of displacement vector
            U_S_old.axpy(-1, U_S.vector())
            increment = norm(U_S_old)
            U_S_old[:] = U_S.vector()[:]

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

        # FIXME: Problem-specific, should not be here
        # Store raw data for displacement
        # Compute displacement
        #displacement = (1.0/structure_area)*assemble(U_S[0]*dx, mesh = U_S.function_space().mesh())
        #disp_vs_t.write(str(displacement) + " ,  " + str(t) + "\n")

        # Store solution in time series
        u_F_series.store(u_F.vector(), t)
        p_F_series.store(p_F.vector(), t)
        U_S_series.store(U_S.vector(), t)
        P_S_series.store(P_S.vector(), t)
        U_M_series.store(U_M.vector(), t)

        # Store time and time steps
        times.append(t)
        timesteps.append(dt)

        # Save solution in VTK format
        if save_solution:
            file_u_F << u_F
            file_p_F << p_F
            file_U_S << U_S
            file_P_S << P_S
            file_U_M << U_M

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

    # Close file
    disp_vs_t.close()

    # Compute convergence indicators
    end_displacement = (1.0/structure_area)*assemble(U_S[0]*dx, mesh = U_S.function_space().mesh())
    end_velocity = u_F((4.0, 0.5))[0]
    print "Functional 1 (displacement):", end_displacement
    print "Functional 2 (velocity):    ", end_velocity

    # Store info (some needs to be stored by hand... marked with ##)
    convergence_data.write(str("==FLUID PARAMETERS==")+ "\n")
    convergence_data.write(str("viscosity:  ") + str(F.viscosity()) + "\n")
    convergence_data.write(str("density:    ") + str(F.density()) + "\n" + "\n")
    convergence_data.write(str("==STRUCTURE PARAMETERS==")+ "\n")
    convergence_data.write(str("density: ") + str(S.reference_density()) + "\n")
    convergence_data.write(str("mu:      ") + str(75) + "\n") ##
    convergence_data.write(str("lambda:  ") + str(125) + "\n" + "\n") ##
    convergence_data.write(str("==MESH PARAMETERS==")+ "\n")
    convergence_data.write(str("no. mesh smooth:    ") + str(mesh_smooth) + "\n")
    convergence_data.write(str("alpha:              ") + str(M.alpha) + "\n")
    convergence_data.write(str("mu:                 ") + str(M.mu) + "\n")
    convergence_data.write(str("lambda:             ") + str(M.lmbda) + "\n" + "\n" + "\n" + "\n")
    convergence_data.write(str("****MESH/TIME*****") +  "\n")
    convergence_data.write(str("Mesh size (nx*ny):     ") + str(mesh.num_cells()) + "\n")
    convergence_data.write(str("Min(hK):               ") + str(mesh.hmin()) + "\n")
    convergence_data.write(str("End time T:            ") + str(T) + "\n")
    convergence_data.write(str("Time step kn:          ") + str(dt) + "\n")
    convergence_data.write(str("Tolerance (FSI f.p.):  ") + str(tol) + "\n" + "\n")
    convergence_data.write(str("****INDICATORS****")+ "\n")
    convergence_data.write(str("Functional 1 (displacement): ") + str(end_displacement) + "\n")
    convergence_data.write(str("Functional 2 (velocity):     ") + str(end_velocity) + "\n")
    convergence_data.close()
