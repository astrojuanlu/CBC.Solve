__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-08-31

from fsiproblem import *

# Read parameters
application_parameters = read_parameters()

# Define inflow at the top
top = "near(x[1], 1.0)"

# Define bottom boundary
bottom = "near(x[1], 0.0)"

# Define noslip boundary
noslip = "on_boundary && !(%s)" % top

class LidDrivenCavity(FSI):

    def __init__(self):

        # Create inital mesh
        n = 2
        mesh = UnitSquare(n, n)

        # Create Riesz representer for goal functional
        self.psi = Expression("c*exp(-((x[0] - x0)*(x[0] - x0) + \
                                       (x[1] - x1)*(x[1] - x1)) / (2.0*r*r))",
                              c = 1.0, r = 0.15, x0 = 0.75, x1 = 0.75)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 1.0

    def evaluate_functional(self, u, p):
        self.psi.c /= assemble(self.psi*dx, mesh=self.Omega)
        return u[1]*self.psi*dx, None, None, None

    def __str__(self):
        return "Lid-driven cavity problem"

    #--- Fluid specific parameters ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip, top]

    def fluid_velocity_dirichlet_values(self):
        return [(0.0, 0.0), Expression(("x[0]*(1.0 - x[0])", "0.0"))]

        # Mats and Fredik's profile
        # return [(0.0, 0.0), Expression(("1.0 - pow(1.0 - x[0], p)", "0.0"), p=18)]

    def fluid_pressure_dirichlet_boundaries(self):
        return [bottom]

    def fluid_pressure_dirichlet_values(self):
        return [0.0]

    def fluid_velocity_initial_condition(self):
        return [0.0, 0.0]

    def fluid_pressure_initial_condition(self):
        return 0.0

# Define and solve problem
problem = LidDrivenCavity()
problem.solve(application_parameters)
