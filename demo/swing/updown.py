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

solid_left = "x[0] < DOLFIN_EPS && x[1] < 0.5 + DOLFIN_EPS"
solid_right = "x[0] > 1.0 - DOLFIN_EPS && x[1] < 0.5 + DOLFIN_EPS"
solid_bottom = "x[1] < DOLFIN_EPS"

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.5 + DOLFIN_EPS

class UpDown(FSI):

    def __init__(self):

        N = 10
        if application_parameters["crossed_mesh"]:
            mesh = UnitSquare(N, N, "crossed")
        else:
            mesh = UnitSquare(N, N)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 10.0

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        # FIXME: Add correct form
        return U_S[0] * dx_S

    def __str__(self):
        return "Simple FSI problem with a reference solution"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.002

    def fluid_velocity_dirichlet_values(self):
         return [(0.0, 0.0)]

    def fluid_velocity_dirichlet_boundaries(self):        return [fluid_bottom]

    def fluid_pressure_dirichlet_values(self):
        return 0.0, 0.0

    def fluid_pressure_dirichlet_boundaries(self):
        return fluid_top, fluid_right

    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
        return "0.0"

    #--- Structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 0.1

    def structure_mu(self):
        return 5.0

    def structure_lmbda(self):
        return 0.25*125.0

    def structure_dirichlet_values(self):
        return [Expression(("0.0","-2*A*x[1]*sin(pi*t)"),
                           A=0.01, pi=DOLFIN_PI, t=0.0),
                (0.0, 0.0),
                Expression(("0.0","-2*A*x[1]*sin(pi*t)"),
                           A=0.01, pi=DOLFIN_PI, t=0.0),
                Expression(("0.0","-A*sin(pi*t)"),
                           A=0.01, pi=DOLFIN_PI, t=0.0)]

    def structure_dirichlet_boundaries(self):
        return [solid_left, solid_bottom, solid_right, fluid_bottom]

    def structure_neumann_boundaries(self):
        return [fluid_bottom]

    def structure_body_force(self):
        return Expression(("0.0", "2*pow(pi, 2.0)*A*rho_S*x[1]*sin(pi*t)"),
                           pi=DOLFIN_PI, A=0.01, rho_S=self.structure_density(), t=0.0)


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
