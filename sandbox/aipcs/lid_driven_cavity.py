__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-08-11

from fsiproblem import *

# Read parameters
application_parameters = read_parameters()

# Define inflow boundary
inflow = "x[0] == 0.0 && x[1] < 1.0 - DOLFIN_EPS"

# Define noslip boundary
noslip = "on_boundary && !(%s)" % inflow

class LidDrivenCavity(FSI):

    def __init__(self):

        # Number of inital elements
        n = 2
        nx = n
        ny = n

        # Create mesh
        mesh = Rectangle(0.0, 0.0, 1.0, 1.0, nx, ny)
        # cell_domains = CellFunction("uint", mesh)
        # cell_domains.set_all(0)
        # structure = Structure()
        # structure.mark(cell_domains, 1)
        # mesh = SubMesh(mesh, cell_domains, 0)

        # Create subdomains for goal functionals
        # self.outflow_domain = compile_subdomains(outflow)
        # self.top_domain = compile_subdomains(top)

        # Create Riesz representer for goal functional
        self.psi = Expression("c*exp(-((x[0] - x0)*(x[0] - x0) + (x[1] - x1)*(x[1] - x1)) / (2.0*r*r))", c = 1.0, r = 0.15, x0 = 0.5, x1 = 0.3)

        # Uncomment for testing
        #mesh = refine(mesh)
        #mesh = refine(mesh)
        #self.psi.c /= assemble(self.psi*dx, mesh=mesh)
        #print "Normalization:", self.psi.c
        #plot(self.psi, interactive=True, mesh=mesh)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 2.5

    def evaluate_functional(self, u, p):

        # Goal functional 1: integration against Gaussian
        if application_parameters["goal_functional"] == 1:
            info("Goal functional is integration against Gaussian")

            c /= assemble(self.psi*dx, mesh=self.Omega)
            return u[0]*self.psi*dx, None, None, None

    def __str__(self):
        return "Lid driven cavity"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        return [(0.0, 0.0)]

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip]

    def fluid_pressure_dirichlet_values(self):
        return [0.0]

    def fluid_pressure_dirichlet_boundaries(self):
        return [inflow]

    def fluid_velocity_initial_condition(self):
        return "4.0*x[1]*(1-x[1])"

    def fluid_pressure_initial_condition(self):
        return 0.0

# Define and solve problem
problem = LidDrivenCavity()
problem.solve(application_parameters)
