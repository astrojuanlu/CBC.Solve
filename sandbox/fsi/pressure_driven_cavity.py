__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-24

from fsiproblem import *

# Read parameters
application_parameters = read_parameters()

# ------------------------------------------
#                  NOSLIP
#  p = 1                             p = 0
#
#  NOSLIP                            NOSLIP
# -------                          ---------
#        |                        |
#        |                        |
#        |                        |
#        |         FLUID          |
#        |                        |
#        |                        |
#        |                        |
#        |                        |
#        |________________________| (1.0, 0.25)
#        |                        |
# FIXED  |        STRUCTURE       |  FIXED
#        |                        |
#         ------------------------ (1.0, 0.0)
#                   FREE

# Constants related to the geometry of the problem
cavity_length = 1.0
cavity_height = 1.5
structure_left = 0.0
structure_right = 1.0
structure_top = 0.25
inflow_top = 1.5
inflow_bottom = 1.25

# Define boundaries
inflow      = "x[0] < DOLFIN_EPS && \
               x[1] > %g - DOLFIN_EPS && \
               x[1] < %g + DOLFIN_EPS" % (inflow_bottom, inflow_top)
outflow     = "x[0] > %g - DOLFIN_EPS && \
               x[1] > %g - DOLFIN_EPS && \
               x[1] < %g + DOLFIN_EPS" % (cavity_length, inflow_bottom, inflow_top)
fixed_left  = "x[0] < DOLFIN_EPS && \
               x[1] > -DOLFIN_EPS && \
               x[1] < %g + DOLFIN_EPS" % structure_top
fixed_right = "x[0] > %g - DOLFIN_EPS && \
               x[1] > -DOLFIN_EPS && \
               x[1] < %g + DOLFIN_EPS" % (structure_right, structure_top)
noslip      = "on_boundary && !(%s) && !(%s)" % (inflow, outflow)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

# Define problem class
class PressureDrivenCavity(FSI):

    def __init__(self):

        # Define mesh
        nx = 4
        ny = 6
        mesh = Rectangle(0.0, 0.0, cavity_length, cavity_height, nx, ny)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 1.0

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        A = (structure_right - structure_left) * structure_top
        return (1.0/A) * U_S[1] * dx_S

    def __str__(self):
        return "Pressure driven cavity flow with an elastic bottom"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.01

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
        return "1.0 - x[0]"

    #--- Structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 1.0

    def structure_mu(self):
        return 10.0

    def structure_lmbda(self):
        return 10.0

    def structure_dirichlet_values(self):
        return [(0.0, 0.0), (0.0, 0.0)]

    def structure_dirichlet_boundaries(self):
        return [fixed_left, fixed_right]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    #--- Mesh problem ---

    def mesh_mu(self):
        return 3.8461

    def mesh_lmbda(self):
        return 5.76

    def mesh_alpha(self):
        return 1.0

# Define and solve problem
problem = PressureDrivenCavity()
problem.solve(application_parameters)
