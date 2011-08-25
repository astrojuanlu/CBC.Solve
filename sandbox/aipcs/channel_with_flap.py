__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-03-17

from fsiproblem import *

# Read parameters
application_parameters = read_parameters()

# Constants related to the geometry of the problem
channel_length  = 4.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.8
structure_top   = 0.6

# Define inflow/outflow boundaries
inflow  = "x[0] < DOLFIN_EPS && \
           x[1] > -DOLFIN_EPS && \
           x[1] < %g + DOLFIN_EPS" % channel_height
outflow = "x[0] > %g - DOLFIN_EPS && \
           x[1] > -DOLFIN_EPS && \
           x[1] < %g + DOLFIN_EPS" % (channel_length, channel_height)

# Define now-slip boundary
inflow_inner  = "x[0] < DOLFIN_EPS && \
                 x[1] > DOLFIN_EPS && \
                 x[1] < %g - DOLFIN_EPS" % channel_height
outflow_inner = "x[0] > %g - DOLFIN_EPS && \
                 x[1] > DOLFIN_EPS && \
                 x[1] < %g - DOLFIN_EPS" % (channel_length, channel_height)
noslip  = "on_boundary && !(%s) && !(%s)" % (inflow_inner, outflow_inner)

# Define top of flap boundary
top = "x[0] > %g - DOLFIN_EPS && x[0] < %g + DOLFIN_EPS && std::abs(x[1] - %g) < DOLFIN_EPS" % \
      (structure_left, structure_right, structure_top)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

class ChannelWithFlap(FSI):

    def __init__(self):

        n = 2

        nx = n*20
        ny = n*5

        # Create mesh
        mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)
        cell_domains = CellFunction("uint", mesh)
        cell_domains.set_all(0)
        structure = Structure()
        structure.mark(cell_domains, 1)
        mesh = SubMesh(mesh, cell_domains, 0)

        # Create subdomains for goal functionals
        self.outflow_domain = compile_subdomains(outflow)
        self.top_domain = compile_subdomains(top)

        # Create Riesz representer for goal functional
        self.psi = Expression("c*exp(-((x[0] - x0)*(x[0] - x0) + (x[1] - x1)*(x[1] - x1)) / (2.0*r*r))", c = 1.0, r = 0.15, x0 = 2.2, x1 = 0.3)

        # Old version
        # self.psi.c = 1.0
        # self.psi.r = 0.15
        # self.psi.x0 = 2.2
        # self.psi.x1 = 0.3

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

        # Goal functional 0: shear stress on top of flap
        if application_parameters["goal_functional"] == 0:
            info("Goal functional is shear stress on top of flap")

            mu = self.fluid_viscosity()
            sigma = mu*(grad(u) + grad(u).T)
            return sigma[0, 1]*ds, None, self.top_domain, None

        # Goal functional 1: integration against Gaussian
        if application_parameters["goal_functional"] == 1:
            info("Goal functional is integration against Gaussian")

            self.psi.c /= assemble(self.psi*dx, mesh=self.Omega)
            return u[0]*self.psi*dx, None, None, None

        # Goal functional 2: outflow
        if application_parameters["goal_functional"] == 2:
            info("Goal functional is total outflow")

            return u[0]*ds, None, self.outflow_domain, None

    def __str__(self):
        return "Channel flow with an immersed elastic flap"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.002

    def fluid_velocity_dirichlet_values(self):
        return [(0.0, 0.0)]

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip]

    def fluid_pressure_dirichlet_values(self):
        return 1.0, 0.0

    def fluid_pressure_dirichlet_boundaries(self):
        return inflow, outflow

    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
        return "1.0 - x[0] / %g" % channel_length

# Define and solve problem
problem = ChannelWithFlap()
problem.solve(application_parameters)
