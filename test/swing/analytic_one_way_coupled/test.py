__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-04-29
# Last changed: 2012-04-29

# Modified by Marie E. Rognes


from cbc.swing import *
from right_hand_sides_revised import *

# Read parameters
application_parameters = read_parameters()

application_parameters["solve_primal"] = True
application_parameters["save_solution"] = False
application_parameters["solve_dual"] = False
application_parameters["estimate_error"] = False
application_parameters["plot_solution"] = False
application_parameters["uniform_timestep"] = True
application_parameters["uniform_mesh"] = True
application_parameters["tolerance"] = 1e-16
application_parameters["initial_timestep"] = 0.01
application_parameters["output_directory"] = "results"
application_parameters["max_num_refinements"] = 0

C = 1.0

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.5 + DOLFIN_EPS

class AnalyticOneWayCoupled(FSI):

    def __init__(self):

        # Create mesh
        n = 16
        mesh = UnitSquare(n, n)

        # Create analytic expressions for various forces
        self.f_F = Expression(cpp_f_F, degree=2)
        self.f_F.C = C
        self.F_S = Expression(cpp_F_S, degree=6)
        self.F_S.C = C
        self.F_M = Expression(cpp_F_M, degree=2)
        self.F_M.C = C
        self.G_0 = Expression(cpp_G_S0, degree=2)
        self.G_0.C = C

        # Exact solutions
        self.u_F = Expression(cpp_u_F, degree=2)  # Exact fluid velocity
        self.u_F.C = C
        self.p_F = Expression(cpp_p_F, degree=1)  # Exact pressure
        self.p_F.C = C
        self.U_S = Expression(cpp_U_S, degree=2)  # Exact structure disp
        self.U_S.C = C
        self.U_M = Expression(cpp_U_M, degree=2)  # Exact mesh disp
        self.U_M.C = C

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 0.1

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        return U_S[1] * dx_S

    def update(self, t0, t1, dt):
        t = 0.5*(t0 + t1)

        self.f_F.t = t
        self.F_S.t = t
        self.F_M.t = t
        self.G_0.t = t

        self.u_F.t = t1
        self.p_F.t = t1
        self.U_M.t = t1
        self.U_S.t = t1

    def exact_solution(self):
        return self.u_F, self.p_F, self.U_S, None, self.U_M

    def __str__(self):
        return "One way coupled channel flow with an immersed elastic flap"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        return [self.u_F]

    def fluid_velocity_dirichlet_boundaries(self):
        return ["on_boundary"]

    def fluid_pressure_dirichlet_values(self):
        return []

    def fluid_pressure_dirichlet_boundaries(self):
        return []

    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
        return 0.0

    def fluid_body_force(self):
        return self.f_F

    def fluid_traction_values(self):
        return self.g_F

    def mesh_velocity(self, V):
        w = Function(V)
        return w

    #--- Structure problem ---
    # Use known solution on entire mesh
    def structure(self):
        return Structure()

    def structure_density(self):
        return 100.0

    def structure_mu(self):
        return 1.0

    def structure_lmbda(self):
        return 2.0

    def structure_dirichlet_values(self):
        return [self.U_S]

    def structure_dirichlet_boundaries(self):
        return ["near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0)"]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    def structure_body_force(self):
        return self.F_S

    def structure_boundary_traction_extra(self):
        return self.G_0

    #--- Parameters for mesh problem ---

    def mesh_mu(self):
        return 1.0

    def mesh_lmbda(self):
        return 2.0

    def mesh_alpha(self):
        return 1.0

    def mesh_right_hand_side(self):
        return self.F_M

# Define and solve problem
problem = AnalyticOneWayCoupled()
value = problem.solve(application_parameters)

# Check that value matches
regression_value = 0.00506396922092
diff = abs(value - regression_value)
assert diff < 1.e-8, "Test failed: difference =  %g" % diff
info_green("Test passed with diff = %g (tol = 1.e-8)" % diff )


