__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-08-28


from fsiproblem import *

# Read parameters
application_parameters = read_parameters()

# Define the top boundary
top = "near(x[1], 2.0)"

# Define noslip boundary
noslip = "on_boundary && !(%s)" % top

# Define the inflow boundary for regulized profile
top_left   = "x[0] == 0.0 && x[0] < 0.25 - DOLFIN_EPS &&\
              near(x[1], 2.0)"
top_middle = "x[0] > 0.25 + DOLFIN_EPS  && x[0] < 1.75 - DOLFIN_EPS &&\
              near(x[1], 2.0)"
top_right  = "x[0] > 1.75 && x[0] <= 2 &&\
              near(x[1], 2.0)"

class LidDrivenCavity(FSI):

    def __init__(self):

        # Number of inital elements
        n = 40
        nx, ny = n, n

        # Create mesh
        mesh = Rectangle(0.0, 0.0, 2.0, 2.0, nx, ny)

        # Create Riesz representer for goal functional
        self.psi = Expression("c*exp(-((x[0] - x0)*(x[0] - x0) + (x[1] - x1)*(x[1] - x1)) / (2.0*r*r))", c = 1.0, r = 0.15, x0 = 1.5, x1 = 1.25)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 2.5

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
        return [(0.0, 0.0), Expression(("4*pow(x[1], p)", "0.0"), p=1), Expression(("1.0", "0.0")), Expression(("2*pow(2-x[1], p)", "0.0"), p=1)]

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
