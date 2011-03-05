__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-03-06

__all__ = ["FSISolver"]

from time import time as python_time
from dolfin import parameters as dolfin_parameters
from dolfin import *
from cbc.common import CBCSolver

from primalsolver import solve_primal
from dualsolver import solve_dual
from adaptivity import estimate_error, refine_mesh, save_mesh

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
                info("Not estimating error")
                error = 0.0

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

            # Check if mesh error is small enough
            begin("Checking space error estimate")
            mesh_tolerance = w_h * tolerance
            if E_h <= mesh_tolerance:

                # Freeze mesh
                info_blue("Freezing current mesh: E_h = %g <= TOL_h = %g" % (E_h, mesh_tolerance))
                info_blue("Starting final round!")
                final = True
                refined_mesh = self.problem.mesh()

                # Refine timestep
                refine_timestep(E_k, parameters)

            elif parameters["uniform_mesh"]:
                info_red("Refining mesh uniformly")
                refined_mesh = refine(self.problem.mesh())
                self.problem.init_meshes(refined_mesh, parameters)
            else:
                info_red("Refining mesh adaptively")
                refined_mesh = refine_mesh(self.problem, self.problem.mesh(), indicators, parameters)
                self.problem.init_meshes(refined_mesh, parameters)
            end()

            # Update and save mesh
            mesh = refined_mesh
            save_mesh(mesh, parameters)

            # Check if we reached the maximum number of refinements
            if level == max_num_refinements:
                info_blue("Reached maximum number of refinement levels (%d)", max_num_refinements)

        # Report elapsed time
        info_blue("Solution computed in %g seconds." % (python_time() - cpu_time))

        # Return solution
        return goal_functional
