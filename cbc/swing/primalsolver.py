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

#class primalsolver
def solve_primal(problem, parameters):
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
    at_end = False

    # Initialize integration of goal functional (assuming M(u) = 0 at t = 0)
    integrated_goal_functional = 0.0
    old_goal_functional = 0.0

    # Get reference value of goal functional
    if hasattr(problem, "reference_value"):
        reference_value = problem.reference_value()
    else:
        reference_value = None


    if parameters["primal_solver"] == "Newton":
        # Initialize an FSINewtonsolver Object

        #In order to change the Newton Solver parameters edit the variable
        #solver_params in module fsinewton.solver.default_params
        fsinewtonsolver = FSINewtonSolver(problem)

        #initialize the solve settings
        fsinewtonsolver.params["solve"] = False
        fsinewtonsolver.solve()
        
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
            assert save_solution,"Parameter save_solution must be true to use the Newton Solver"
          ##  U_S0,U_S1,P_S1,increment,numiter = newton_solve(F,S,U_S0,M,dt,t1,parameters,itertol,problem,fsinewtonsolver)
            U_S1,P_S1 = newton_solve(F,S,M,primal_series,dt,t1,parameters,problem,fsinewtonsolver)
        elif parameters["primal_solver"] == "fixpoint":
            U_S0,U_S1,P_S1,increment,numiter = fixpoint_solve(F,S,U_S0,M,dt,t1,parameters,itertol,problem)
        else:
            raise Exception("Only 'fixpoint' and 'Newton' are possible values \
                            for the parameter 'primal_solver'")
        
    #####################################################
    #The primal solve worked so now go to post processing
    #####################################################
    #def postprocessing():
        # Plot solution
        if plot_solution:
            _plot_solution(u_F1, p_F1, U_S0, U_M1)
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

    # Save final value of goal functional
    save_goal_functional_final(goal_functional, integrated_goal_functional, reference_value, parameters)

    # Report elapsed time
    info_blue("Primal solution computed in %g seconds." % (python_time() - cpu_time))
    info("")

    # Return solution
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
        if increment < itertol and iter > 2:            
            info_green("Increment is %g. Maybe plotting" % increment)
            return (U_S0, U_S1, P_S1, increment,numiter)

        # Check if we have reached the maximum number of iterations
        elif numiter == maxiter - 1:
            raise RuntimeError, "FSI iteration failed to converge after %d iterations." % maxiter

        # Print size of increment
        info("")
        info_red("Increment = %g (tolerance = %g), iteration %d" % (increment, itertol, numiter + 1))
        end()
        
##def newton_solve(F,S,U_S0,M,dt,t1,parameters,itertol,problem,fsinewtonsolver):
def newton_solve(F,S,M,primal_series,dt,t1,parameters,problem,fsinewtonsolver):
    """Solve for the time step using Newton's method"""
    # Get solution values

    #Input time data
    fsinewtonsolver.t = t1
    fsinewtonsolver.dt = dt

    #Read data from previous time step
    Uglob = create_primal_functions(problem.Omega, parameters)
    read_primal_data(Uglob, t1, problem.Omega, problem.Omega_F, problem.Omega_S, primal_series, parameters)

    newtonfunc_to_primefunc = {"U_F":Uglob[0],"P_F":Uglob[1],"U_S":Uglob[2],"P_S":Uglob[3],"U_M":Uglob[4]}
    subloc = fsinewtonsolver.spaces.subloc
    for funcname in newtonfunc_to_primefunc:
        #Make sure the function spaces match
        newtondim = subloc.spaceends[funcname] - subloc.spacebegins[funcname]
        funcdim = newtonfunc_to_primefunc[funcname].function_space().dim() 
        assert newtondim == funcdim,"Error in inserting data into FSINewtonSolver, \
                                      Mismatch in dimension of function %s"%function
                
        #input the previous solution values into the solver
        fsinewtonsolver.U0.vector()[subloc.spacebegins[funcname]:subloc.spaceends[funcname]] = \
        newtonfunc_to_primefunc[funcname].vector()[:]
    
    #solve the time step
    fsinewtonsolver.time_step()

    #map the data back into the local functions
##    (global_dofs_U_F, global_dofs_P_F,global_dofs_U_S,global_dofs_P_S,global_dofs_U_M)
    dofmaps =  get_globaldof_mappings(problem.Omega,problem.Omega_F,problem.Omega_S, parameters)
    globalfuncs = [fsinewtonsolver.U1_F,fsinewtonsolver.P1_F,fsinewtonsolver.U1_S,fsinewtonsolver.U1_M]
    Uloc = extract_solution(F, S, M)
    
    for globfunc,locfunc,dofmap in zip(globalfuncs,Uloc,dofmaps):
        locfunc = mf.extract_subfunction(globfunc).vector()[dofmap]
           #U1_S,P1_S   
    return Uloc[2],Uloc[3]
