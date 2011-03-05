__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-03-06

from fsiproblem import *

# Read parameters
application_parameters = read_parameters()

# Constants related to the geometry of the problem
channel_length  = 2.4
channel_height  = 0.4

# Define boundaries
inflow  = "x[0] < DOLFIN_EPS && \
           x[1] > DOLFIN_EPS && \
           x[1] < %g - DOLFIN_EPS" % channel_height
outflow = "x[0] > %g - DOLFIN_EPS && \
           x[1] > DOLFIN_EPS && \
           x[1] < %g - DOLFIN_EPS" % (channel_length, channel_height)
noslip  = "on_boundary && !(%s) && !(%s)" % (inflow, outflow)

class FlowPastCylinder(FSI):

    def __init__(self):

        # Read mesh
        mesh = Mesh("cylinder.xml.gz")

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 8.0

    def evaluate_functional(self, u_F, p_F, dx):
        return u_F[0] * dx

    def __str__(self):
        return "Channel flow with an immersed elastic flap"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.001

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
        return "1.0 - x[0]/2.4"

# Define and solve problem
problem = FlowPastCylinder()
problem.solve(application_parameters)
