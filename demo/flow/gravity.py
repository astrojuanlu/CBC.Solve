__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-16

from cbc.flow import *

class Gravity(NavierStokes):

    def mesh(self):
        return UnitSquare(16, 16)

    def viscosity(self):
        return 1.0 / 8.0

    def body_force(self, V):
        gravity = Expression(("0.0", "-9.81*t"), t=0.0)
        return gravity

    def velocity_dirichlet_values(self):
        return [(0, 0)]

    def velocity_dirichlet_boundaries(self):
        return ["x[1] < DOLFIN_EPS || x[1] > 1.0 - DOLFIN_EPS"]

    def pressure_dirichlet_values(self):
        return [1, 0]

    def pressure_dirichlet_boundaries(self):
        return ["x[0] < DOLFIN_EPS", "x[0] > 1 - DOLFIN_EPS"]

    def velocity_initial_condition(self):
        return (0, 0)

    def pressure_initial_condition(self):
        return "1 - x[0]"

    def end_time(self):
        return 0.5

    def __str__(self):
        return "Pressure-driven channel with time-dependent gravity"

# Solve problem
problem = Gravity()
problem.parameters["solver_parameters"]["plot_solution"] = True
problem.parameters["solver_parameters"]["save_solution"] = True
u, p = problem.solve()
