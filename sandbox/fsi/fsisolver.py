__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-11-09

__all__ = ["FSISolver"]

from time import time

from dolfin import *
from cbc.common import CBCSolver

from primalsolver import PrimalSolver
from dualsolver import DualSolver
from adaptivity import estimate_error, refine_mesh, save_mesh

class FSISolver(CBCSolver):

    def __init__(self, problem):
        "Initialize FSI solver"

        # Initialize base class
        CBCSolver.__init__(self)

        # Set solver parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("solve_primal", True)
        self.parameters.add("solve_dual", True)
        self.parameters.add("estimate_error", True)
        self.parameters.add("plot_solution", False)
        self.parameters.add("save_solution", True)
        self.parameters.add("save_series", True)
        self.parameters.add("tolerance", 0.1)
        self.parameters.add("maxiter", 100)
        self.parameters.add("num_smoothings", 50)
        self.parameters.add("uniform_timestep", False)

        # Set DOLFIN parameters
        parameters["form_compiler"]["cpp_optimize"] = True

        # Store problem
        self.problem = problem

    def solve(self):
        "Solve the FSI problem (main adaptive loop)"

        # Create empty solution (return value when primal is not solved)
        U = 5*(None,)

        # Initial guess for stability factor
        ST = 1.0

        # Adaptive loop
        cpu_time = time()
        while True:

            # Solve primal problem
            if self.parameters["solve_primal"]:
                begin("Solving primal problem")
                primal_solver = PrimalSolver(self.problem, self.parameters)
                U = primal_solver.solve(ST)
                end()
            else:
                info("Not solving primal problem")

            # Solve dual problem
            if self.parameters["solve_dual"]:
                begin("Solving dual problem")
                dual_solver = DualSolver(self.problem, self.parameters)
                dual_solver.solve()
                end()
            else:
                info("Not solving dual problem")

            # Estimate error and compute error indicators
            if self.parameters["estimate_error"]:
                begin("Estimating error and computing error indicators")
                error, indicators, ST, E_h, E_k = estimate_error(self.problem)
                end()
            else:
                info("Not estimating error")
                error = 0

            # Check if error is small enough
            begin("Checking error estimate")
            tolerance = self.parameters["tolerance"]
            if error <= tolerance:
                info_green("Adaptive solver converged: error = %g <= TOL = %g" % (error, tolerance))
                break
            else:
                info_red("Error too large, need to refine: error = %g > TOL = %g" % (error, tolerance))
            end()

            # Check if mesh error is small enough
            begin("Checking space error estimate")
            mesh_tolerance = tolerance * self.problem.space_error_weight()
            if E_h <= mesh_tolerance:
                info_blue("Freezing current mesh: E_h = %g <= TOL_h = %g" % (E_h, mesh_tolerance))
                refined_mesh = self.problem.mesh()
            else:
                begin("Refining mesh")
                refined_mesh = refine_mesh(self.problem, self.problem.mesh(), indicators)
                self.problem.init_meshes(refined_mesh)
            end()

            # Save mesh to file and update
            save_mesh(self.problem.mesh(), refined_mesh)
            mesh = refined_mesh

        # Report elapsed time
        info_blue("Solution computed in %g seconds." % (time() - cpu_time))

        # Return solution
        return U
