__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-08-11

from fsiproblem import *

# Read parameters
application_parameters = read_parameters()

# Define the inflow boundary for regulized profile
top_left   = "x[0] == 0.0 && x[0] < 0.25 - DOLFIN_EPS &&\
              x[1] > 2.0 - DOLFIN_EPS"
top_middle = "x[0] > 0.25 + DOLFIN_EPS  && x[0] < 1.75 - DOLFIN_EPS &&\
              x[1] > 2 - DOLFIN_EPS "
top_right  = "x[0] > 1.75 && x[0] <= 2 &&\
              x[1] > 2 - DOLFIN_EPS "

# Define the entire top boundary
top = "near(x[1], 1.0)"

# Define noslip boundary
noslip = "on_boundary && !(%s)" % top

class LidDrivenCavity(FSI):

    def __init__(self):

        # Number of inital elements
        n = 10
        nx = n
        ny = n

        # Create mesh
#        mesh = UnitSquare(nx, ny)
        mesh = Rectangle(0.0, 0.0, 2.0, 2.0, nx, ny)

        # Create Riesz representer for goal functional
        self.psi = Expression("c*exp(-((x[0] - x0)*(x[0] - x0) + (x[1] - x1)*(x[1] - x1)) / (2.0*r*r))", c = 1.0, r = 0.10, x0 = 0.5, x1 = 0.75)

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
        return [noslip, top_left, top_middle, top_right]

    def fluid_velocity_dirichlet_values(self):
        # return [(0.0, 0.0), Expression(("4.0*x[0]*(1.0 - x[0])", "0.0"))]
        return [(0.0, 0.0), Expression(("2*x[1]", "0.0")), Expression(("0.5", "0.0")), Expression(("2*(2-x[1])", "0.0"))]

    def fluid_pressure_dirichlet_boundaries(self):
        return [top]

    def fluid_pressure_dirichlet_values(self):
        return [0.0]

    def fluid_velocity_initial_condition(self):
        return [0.0, 0.0]

    def fluid_pressure_initial_condition(self):
        return 0.0

# Define and solve problem
problem = LidDrivenCavity()
problem.solve(application_parameters)
