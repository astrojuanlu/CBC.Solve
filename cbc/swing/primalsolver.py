"This module implements the primal FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-08-01
#Gabriel Balaban - Added Newton primal solver option.

import math
import pylab
from time import time as python_time
from dolfin import *

from cbc.common.utils import timestep_range
from subproblems import *
from adaptivity import *
from storage import *
import sys
from cbc.swing.fsinewton.solver.solver_fsinewton import FSINewtonSolver
import fsinewton.utils.misc_func as mf
import copy

#GB time
from cbc.swing.fsinewton.utils.timings import timings


class PrimalSolver(object):
    def __init__(self):
        self.g_numiter = 0
        
    def solve_primal(self,problem, parameters):
        "Solve primal FSI problem"
        #def __init__()
        # Get parameters
        T = problem.end_time()
        dt = initial_timestep(problem, parameters)
        TOL = parameters["tolerance"]
        w_k = parameters["w_k"]
        w_c = parameters["w_c"]
        save_solution = parameters["save_solution"] 
        uniform_timestep = parameters["uniform_timestep"]
        plot_solution = parameters["plot_solution"]

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

        # Update of user problem at t = 0 (important for initial conditions)
        problem.update(0.0, 0.0, dt)

        # Define the three subproblems
        F = FluidProblem(problem, solver_type=parameters["fluid_solver"])
        S = StructureProblem(problem, parameters)
        M = MeshProblem(problem, parameters)

        # Get solution values
        u_F0, u_F1, p_F0, p_F1 = F.solution_values()
        U_M0, U_M1 = M.solution_values()

        # Extract number of dofs
        num_dofs_FSM = extract_num_dofs(F, S, M)
        info("FSI problem has %d dofs (%d + %d + %d)" % \
            (num_dofs_FSM, F.num_dofs, S.num_dofs, M.num_dofs))

        # Get initial structure displacement (used for plotting and checking convergence)
        structure_element_degree = parameters["structure_element_degree"]
        V_S = VectorFunctionSpace(problem.structure_mesh(), "CG", structure_element_degree)
        U_S0 = Function(V_S)

        # Save initial solution to file and series
        U = extract_solution(F, S, M)
        if save_solution:
            _save_solution(U, files)
            write_primal_data(U, 0, primal_series)

        # Initialize adaptive data
        init_adaptive_data(problem, parameters)

        # Initialize time-stepping
        t0 = 0.0
        t1 = dt
        at_end = abs(t1 - T) / T < 100.0*DOLFIN_EPS

        # Initialize integration of goal functional (assuming M(u) = 0 at t = 0)
        integrated_goal_functional = 0.0
        old_goal_functional = 0.0

        # Get reference value of goal functional
        if hasattr(problem, "reference_value"):
            reference_value = problem.reference_value()
        else:
            reference_value = None

        if parameters["primal_solver"] == "Newton":
            #If no initial_step function try to generate one
            if not hasattr(problem,"initial_step"):
                problem.initial_step = lambda :parameters["initial_timestep"]
            
            # Initialize an FSINewtonSolver Object
            fsinewtonsolver = FSINewtonSolver(problem,\
                                params = parameters["FSINewtonSolver"].to_dict())

            #initialize the solve settings
            fsinewtonsolver.prepare_solve()
            
        #def solve_primal()
        while True:

            # Display progress
            info("")
            info("-"*80)
            begin("* Starting new time step")
            info_blue("  * t = %g (T = %g, dt = %g)" % (t1, T, dt))

            # Update of user problem
            problem.update(t0, t1, dt)

            # Compute tolerance for FSI iterations
            itertol = compute_itertol(problem, w_c, TOL, dt, t1, parameters)
            if parameters["primal_solver"] == "Newton":
                #Newtonsolver has it's own timings
                assert save_solution,"Parameter save_solution must be true to use the Newton Solver"
                U_S1,U_S0,P_S1,increment,numiter = newton_solve(F,S,M,U_S0,dt,parameters,itertol,problem,fsinewtonsolver)
            elif parameters["primal_solver"] == "fixpoint":
                timings.startnext("FixpointSolve")
                U_S0,U_S1,P_S1,increment,numiter = fixpoint_solve(F,S,U_S0,M,dt,t1,parameters,itertol,problem)
                timings.stop("FixpointSolve")
            else:
                raise Exception("Only 'fixpoint' and 'Newton' are possible values \
                                for the parameter 'primal_solver'")
            self.g_numiter = numiter
        #####################################################
        #The primal solve worked so now go to post processing
        #####################################################
        #def postprocessing():
            # Plot solution
            if plot_solution:
                _plot_solution(u_F1, p_F1, U_S0, U_M1)
            if problem.exact_solution() is not None:
                update_exactsol(u_F1,p_F1,U_S1,U_M1,F,problem,t1)      

            info("")
            info_green("Increment = %g (tolerance = %g), converged after %d iterations" % (increment, itertol, numiter + 1))
            info("")
            end()

            # Saving number of FSI iterations
            save_no_FSI_iter(t1, numiter + 1, parameters)

            # Evaluate user goal functional
            goal_functional = assemble(problem.evaluate_functional(u_F1, p_F1, U_S1, P_S1, U_M1, dx, dx, dx))

            # Integrate goal functional
            integrated_goal_functional += 0.5 * dt * (old_goal_functional + goal_functional)
            old_goal_functional = goal_functional

            # Save goal functional
            save_goal_functional(t1, goal_functional, integrated_goal_functional, parameters)


            # Save solution and time series to file
            U = extract_solution(F, S, M)
            if save_solution:
                _save_solution(U, files)
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
        #End of Time loop
        #Call post processing for the Newton Solver if necessary.
        if parameters["primal_solver"] == "Newton":
            fsinewtonsolver.post_processing()
        # Save final value of goal functional
        save_goal_functional_final(goal_functional, integrated_goal_functional, reference_value, parameters)

        # Report elapsed time
        info_blue("Primal solution computed in %g seconds." % (python_time() - cpu_time))
        info("")

        # Return solution
        info(timings.report_str())
        return (goal_functional, integrated_goal_functional)

def _plot_solution(u_F, p_F, U_S, U_M):
    "Plot solution"
    plot(u_F, title="Fluid velocity")
    plot(p_F, title="Fluid pressure")
    plot(U_S, title="Structure displacement")
    plot(U_M, title="Approx U_M")
    #interactive()


def _save_solution(U, files):
    "Save solution to VTK"
    [files[i] << U[i] for i in range(5)]

def fixpoint_solve(F,S,U_S0,M,dt,t1,parameters,itertol,problem):
    """Return the value at the next time step using fixpoint iteration"""
    # Get Parameters
    maxiter = parameters["maximum_iterations"]
    num_smoothings = parameters["num_smoothings"]
    
    # Get solution values
    u_F0, u_F1, p_F0, p_F1 = F.solution_values()
    U_M0, U_M1 = M.solution_values()
    
    # Fixed point iteration on FSI problem
    for numiter in range(maxiter):
        
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
        begin("* Transferring structure displacement to mesh (S --> M)")
        M.update_structure_displacement(U_S1)
        end()

        # Solve mesh equation
        begin("* Solving mesh subproblem (M)")
        M.step(dt)
        end()

        # Transfer mesh displacement to fluid
        begin("* Transferring mesh displacement to fluid (M --> F)")
        F.update_mesh_displacement(U_M1, dt, num_smoothings)
        end()

        # Compute increment of displacement vector
        U_S0.vector().axpy(-1, U_S1.vector())
        increment = norm(U_S0.vector())
        U_S0.vector()[:] = U_S1.vector()[:]

        # Check convergence
        if increment < itertol and numiter > 2:            
            info_green("Increment is %g. Maybe plotting" % increment)
            print "numiter = ",numiter
            return (U_S0, U_S1, P_S1, increment,numiter)

        # Check if we have reached the maximum number of iterations
        elif numiter == maxiter - 1:
            raise RuntimeError, "FSI fixed point iteration failed to converge after %d iterations." % maxiter

        # Print size of increment
        info("")
        info_red("Increment = %g (tolerance = %g), iteration %d" % (increment, itertol, numiter + 1))
        end()
    
def newton_solve(F,S,M,U0_S,dt,parameters,itertol,problem,fsinewtonsolver):
    """Solve for the time step using Newton's method"""
    # Get mappings from local to global mesh
    dofmaps =  get_globaldof_mappings(problem.Omega,problem.Omega_F,problem.Omega_S, parameters)
    dofmaps = {"U_F":dofmaps[0],"P_F":dofmaps[1],"U_S":dofmaps[2],"P_S":dofmaps[3],"U_M":dofmaps[4]}

    #Input data
    fsinewtonsolver.dt = dt
    fsinewtonsolver.newtonsolver.tol = itertol
    
    #Save U0_S
    U0_S.vector()[:] = mf.extract_subfunction(fsinewtonsolver.U0_S).vector()[dofmaps["U_S"]]

    #solve the time step
    fsinewtonsolver.time_step()

    #map the data back into the local functions
    Uloc = extract_solution(F, S, M)
    Uloc = {"U_F":Uloc[0],"P_F":Uloc[1],"U_S":Uloc[2],"P_S":Uloc[3],"U_M":Uloc[4]}
    #Warning! changed notation
    U1glob = {"U_F":fsinewtonsolver.U1_F,"P_F":fsinewtonsolver.P1_F, \
              "U_S":fsinewtonsolver.D1_S,"P_S":fsinewtonsolver.U1_S, \
              "U_M":fsinewtonsolver.D1_F}
    
    subloc = fsinewtonsolver.spaces.subloc
    for funcname in Uloc:
        #Copy the dofs from global to local
        Uloc[funcname].vector()[:] = mf.extract_subfunction(U1glob[funcname]).vector()[dofmaps[funcname]]

    #Gather together the information to be returned
    U1_S = Uloc["U_S"]
    P1_S = Uloc["P_S"]
    U1_M = Uloc["U_M"]
    numiter = len(fsinewtonsolver.last_itr)
    
    # Compute increment of displacement vector
    U0_S.vector().axpy(-1, U1_S.vector())
    increment = norm(U0_S.vector())
    U0_S.vector()[:] = U1_S.vector()[:]
    
    #Transfer mesh displacement to fluid
    begin("* Transferring mesh displacement to fluid (M --> F)")
    F.update_mesh_displacement(U1_M, dt, 2)
    end()
    
    #Update Structure
    velocityoffset = Uloc["P_S"].function_space().dim()
    S.solver.U.vector()[:velocityoffset] = Uloc["U_S"].vector()
    S.solver.U.vector()[velocityoffset:] = Uloc["P_S"].vector()
    return (U1_S,U0_S,P1_S,increment,numiter)

def update_exactsol(u_F1,p_F1,U_S1,U_M1,F,problem,t1):
    """Print errors and update exact solutions"""
    #Update the exact solutions
    (u_F_ex, p_F_ex, U_S_ex, P_S_ex, U_M_ex) \
        = problem.exact_solution()
    u_F_ex.t = t1
    p_F_ex.t = t1
    U_S_ex.t = t1
    U_M_ex.t = t1

    print "||u_F_ex - u_F || = %.15g" % errornorm(u_F_ex, u_F1),
    print "||u_F_ex|| = %.15g"        % norm(u_F_ex, mesh=F.mesh()),
    print "||u_F|| = %.15g"           % norm(u_F1, mesh=F.mesh())

    print "||p_F_ex - p_F || = %.15g" % errornorm(p_F_ex, p_F1),
    print "||p_F_ex|| = %.15g"        % norm(p_F_ex, mesh=F.mesh()),
    print "||p_F|| = %.15g"           % norm(p_F1, mesh=F.mesh())

    print "||U_S_ex - U_S || = %.15g" % errornorm(U_S_ex, U_S1),
    print "||U_S_ex|| = %.15g"        % norm(U_S_ex, mesh=problem.structure_mesh()),
    print "||U_S|| = %.15g"           % norm(U_S1, mesh=problem.structure_mesh())

    print "||U_M_ex - U_M || = %.15g" % errornorm(U_M_ex, U_M1),
    print "||U_M_ex|| = %.15g"        % norm(U_M_ex, mesh=problem.fluid_mesh()),
    print "||U_M|| = %.15g"           % norm(U_M1, mesh=problem.fluid_mesh())
