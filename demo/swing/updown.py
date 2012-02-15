__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.swing import *

# Read parameters
application_parameters = read_parameters()

# Define boundaries
fluid_left = "x[0] < DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS"
fluid_top  = "x[1] > 1.0 - DOLFIN_EPS"
fluid_right = "x[0] > 1.0 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS"
fluid_bottom = "fabs(x[1] - 0.5) < DOLFIN_EPS"
solid_bottom = "x[1] < DOLFIN_EPS"

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 0.5 + DOLFIN_EPS

class UpDown(FSI):

    def __init__(self):

        N = 10
        if application_parameters["crossed_mesh"]:
            mesh = UnitSquare(N, N, "crossed")
        else:
            mesh UnitSquare(N, N)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 0.10

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        A = (structure_right - structure_left) * structure_top
        return (1.0/A) * U_S[0] * dx_S

    def __str__(self):
        return "Simple FSI problem with a reference solution"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.002

    def fluid_velocity_dirichlet_values(self, w):
        return [w]

    def fluid_velocity_dirichlet_boundaries(self):
        return [fluid_bottom]

    def fluid_pressure_dirichlet_values(self):
        return 1.0, 0.0

    def fluid_pressure_dirichlet_boundaries(self):
        return fluid_left, fluid_top, fluid_right

    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
        return "1.0 - 0.25*x[0]"

    #--- Structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 10.0

    def structure_mu(self):
        return 5.0

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
problem = UpDown()
problem.solve(application_parameters)
