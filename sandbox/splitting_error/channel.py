__author__ = "Kristian Valen-Sendstad and Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-16

from cbc.flow import *

class Channel(NavierStokes):

    def mesh(self):
        return UnitSquare(16, 16)

    def viscosity(self):
        return 1.0 / 8.0

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

    def functional(self, u, p):
        return u((1.0, 0.5))[0]

    def reference(self, t):
        num_terms = 30
        u = 1.0
        c = 1.0
        for n in range(1, 2*num_terms, 2):
            a = 32.0 / (DOLFIN_PI**3*n**3)
            b = (1.0/8.0)*DOLFIN_PI**2*n**2
            c = -c
            u += a*exp(-b*t)*c
        return u

    def __str__(self):
        return "Pressure-driven channel (2D)"

# Solve problem
problem = Channel()
problem.parameters["solver_parameters"]["plot_solution"] = True
problem.parameters["solver_parameters"]["save_solution"] = True
u, p = problem.solve()

# Check error
e = problem.functional(u, p) - problem.reference(0.5)
print "Error is", e
