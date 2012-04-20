__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-04-10

from fsiproblem import *

# Read parameters
application_parameters = read_parameters()
application_parameters["tolerance"] = 1e-16

application_parameters["initial_timestep"] = 0.01
application_parameters["uniform_timestep"] = True
#application_parameters["output_directory"] = "results-k=0.0025"
application_parameters["output_directory"] = "test_M"

application_parameters["solve_primal"] = True
application_parameters["solve_dual"] = True
application_parameters["estimate_error"] = True
#application_parameters["max_num_refinements"] = 0

# Constants related to the geometry of the problem
channel_length  = 4.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.8
structure_top   = 0.6

# Define boundaries
inflow  = "x[0] < DOLFIN_EPS && \
           x[1] > DOLFIN_EPS && \
           x[1] < %g - DOLFIN_EPS" % channel_height
outflow = "x[0] > %g - DOLFIN_EPS && \
           x[1] > DOLFIN_EPS && \
           x[1] < %g - DOLFIN_EPS" % (channel_length, channel_height)
noslip  = "on_boundary && !(%s) && !(%s)" % (inflow, outflow)
fixed   = "x[1] < DOLFIN_EPS && x[0] > %g - DOLFIN_EPS && x[0] < %g + DOLFIN_EPS" % (structure_left, structure_right)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

class ChannelWithFlap(FSI):

    def __init__(self):

        ny = 5
        nx = 20
        if application_parameters["crossed_mesh"]:
            mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny, "crossed")
        else:
            mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 0.5

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        A = (structure_right - structure_left) * structure_top
        return (1.0/A) * U_S[0] * dx_S

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
        return "1.0 - 0.25*x[0]"

    #--- Structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 0.25*15.0

    def structure_mu(self):
        return 0.25*75.0

    def structure_lmbda(self):
        return 0.25*125.0

    def structure_dirichlet_values(self):
        return [(0.0, 0.0)]

    def structure_dirichlet_boundaries(self):
        return [fixed]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    #--- Parameters for mesh problem ---

    def mesh_mu(self):
        return 3.8461

    def mesh_lmbda(self):
        return 5.76

    def mesh_alpha(self):
        return 1.0

# Define and solve problem
problem = ChannelWithFlap()
problem.solve(application_parameters)
