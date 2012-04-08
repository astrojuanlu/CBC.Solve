__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-04-09

__all__ = ["FSISolver"]

from time import time as python_time
from dolfin import parameters as dolfin_parameters
from dolfin import *
from cbc.common import CBCSolver

from primalsolver import solve_primal
from dualsolver import solve_dual
from adaptivity import estimate_error, refine_mesh, refine_timestep, save_mesh

class FSISolver(CBCSolver):

    def __init__(self, problem):
        "Initialize FSI solver"

        # Initialize base class
        CBCSolver.__init__(self)

        # Store problem
        self.problem = problem

    def solve(self, parameters):
        "Solve the FSI problem (main adaptive loop)"

        # Get parameters
        tolerance = parameters["tolerance"]
        w_h = parameters["w_h"]
        max_num_refinements = parameters["max_num_refinements"]

        # Set DOLFIN parameters
        dolfin_parameters["form_compiler"]["cpp_optimize"] = True
        dolfin_parameters["refinement_algorithm"] = parameters["refinement_algorithm"]

        # Create empty solution (return value when primal is not solved)
        U = 5*(None,)

        # Save initial mesh
        save_mesh(self.problem.mesh(), parameters)

        # Adaptive loop
        cpu_time = python_time()
        goal_functional = None
        final = False
        for level in range(max_num_refinements + 1):

            # Solve primal problem
            if parameters["solve_primal"]:
                begin("Solving primal problem")
                goal_functional = solve_primal(self.problem, parameters)
                end()
            else:
                info("Not solving primal problem")

            # Solve dual problem
            if parameters["solve_dual"]:
                begin("Solving dual problem")
                solve_dual(self.problem, parameters)
                end()
            else:
                info("Not solving dual problem")

            # Estimate error and compute error indicators
            if parameters["estimate_error"]:
                begin("Estimating error and computing error indicators")
                error, indicators, E_h, E_k, E_c = estimate_error(self.problem, parameters)
                end()
            else:
                error = max(1.0, 2*tolerance)
                info("Not estimating error, setting error to max(1, 2*tolerance) = %g" % error)

            # Check if error is small enough
            begin("Checking error estimate")
            if error <= tolerance:
                info_green("Adaptive solver converged on level %d: error = %g <= TOL = %g" % (level, error, tolerance))
                break
            elif final:
                info_green("Adaptive solver converged on level %d: error = %g (TOL = %g)" % (level, error, tolerance))
                info("Error too large but it doesn't get any better than this. ;-)")
                break
            else:
                info_red("Error too large, need to refine: error = %g > TOL = %g" % (error, tolerance))
            end()

            # Check if we reached the maximum number of refinements
            if level == max_num_refinements:
                info_blue("Reached maximum number of refinement levels (%d)", max_num_refinements)
                return goal_functional

            # Mesh adaptivity
            begin("Checking space error estimate")
            mesh_tolerance = w_h * tolerance
            if parameters["uniform_mesh"]:
                info_red("Refining mesh uniformly")
                refined_mesh = refine(self.problem.mesh())
                self.problem.init_meshes(refined_mesh, parameters)
            elif E_h <= mesh_tolerance:
                info_blue("Freezing current mesh: E_h = %g <= TOL_h = %g" % (E_h, mesh_tolerance))
                info_blue("Starting final round!")
                final = True
                refined_mesh = self.problem.mesh()
            else:
                info_red("Refining mesh adaptively")
                refined_mesh = refine_mesh(self.problem, self.problem.mesh(), indicators, parameters)
                self.problem.init_meshes(refined_mesh, parameters)
            end()

            # Time step adaptivity
            if parameters["uniform_timestep"]:
                info_red("Refining time step uniformly")
                parameters["initial_timestep"] = 0.5 * parameters["initial_timestep"]
            elif E_h < E_k or E_h < E_c:
                info_red("Refining time step adaptively")
                refine_timestep(E_k, parameters)
            else:
                info_blue("Keeping time step tolerance fixed")

            # Update and save mesh
            mesh = refined_mesh
            save_mesh(mesh, parameters)

        # Report elapsed time
        info_blue("Solution computed in %g seconds." % (python_time() - cpu_time))

        # Return solution
        return goal_functional
