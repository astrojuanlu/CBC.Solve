__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-04
# Last changed: 2012-05-02

# Modified by Marie E. Rognes

from cbc.swing import *
from right_hand_sides import *

ref = 0
N = 5
dt = 0.1/N/2**(ref)

application_parameters = read_parameters()
application_parameters["mesh_element_degree"] = 3
application_parameters["structure_element_degree"] = 2
application_parameters["save_solution"] = True
application_parameters["solve_primal"] = True
application_parameters["solve_dual"] = False
application_parameters["estimate_error"] = False
application_parameters["plot_solution"] = False
application_parameters["uniform_timestep"] = True
application_parameters["uniform_mesh"] = True
application_parameters["tolerance"] = 1e-16
application_parameters["fixedpoint_tolerance"] = 1.e-14
application_parameters["initial_timestep"] = dt
application_parameters["output_directory"] = "results_analytic_fluid_test"
application_parameters["max_num_refinements"] = 0
application_parameters["use_exact_solution"] = False

application_parameters["fluid_solver"] = "taylor-hood"
#application_parameters["fluid_solver"] = "ipcs"

# Define relevant boundaries
right = "near(x[0], 2.0)"
left = "near(x[0], 0.0)"
noslip = "near(x[1], 0.0) || near(x[1], 1.0)"

# Constant used in definition of analytic solutions
C = 1.0

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 1.0 - DOLFIN_EPS

class SimpleAnalytic(FSI):

    def __init__(self):

        # Create mesh
        n = N*(2**ref)
        mesh = Rectangle(0.0, 0.0, 2.0, 1.0, 2*n, n)

        # Create analytic expressions for various forces
        self.f_F = Expression(cpp_f_F, degree=1)
        self.g_F = Expression(cpp_g_F, degree=1)
        self.F_S = Expression(cpp_F_S, degree=2)
        self.F_M = Expression(cpp_F_M, degree=3)
        self.G_0 = Expression(cpp_G_S0, degree=3)

        # Initialize
        forces = [self.f_F, self.g_F, self.F_S, self.F_M, self.G_0]
        for f in forces:
            f.C = C
            f.t = 0.0

        # Exact solutions
        self.u_F = Expression(cpp_u_F, degree=2)
        self.p_F = Expression(cpp_p_F, degree=1)
        self.U_S = Expression(cpp_U_S, degree=2)
        self.P_S = Expression(cpp_P_S, degree=2)
        self.U_M = Expression(cpp_U_M, degree=3)
        self.P_M = Expression(cpp_P_M, degree=2)
        solutions = [self.u_F, self.p_F, self.U_S, self.P_S, self.U_M]

        # Initialize these too
        for s in solutions:
            s.C = C
            s.t = 0.0

        # Testing
        self.exact_F = False
        self.exact_S = False

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 0.1

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        return U_S[0] * dx_S

    def update(self, t0, t1, dt):
        t = 0.5*(t0 + t1)

        self.f_F.t = t
        self.g_F.t = t
        self.F_S.t = t
        self.G_0.t = t
        self.F_M.t = t

        self.u_F.t = t1
        self.p_F.t = t1
        self.U_S.t = t1
        self.P_S.t = t1
        self.U_M.t = t1

    def exact_solution(self):
        return self.u_F, self.p_F, self.U_S, self.P_S, self.U_M

    def __str__(self):
        return "Channel flow pushing an elastic box"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        return [self.u_F]

    def fluid_velocity_dirichlet_boundaries(self):
        if self.exact_F:
            return ["0.0 < 1.0"]
        else:
            return [noslip]

    def fluid_pressure_dirichlet_values(self):
        if self.exact_F:
            return [self.p_F]
        else:
            return []

    def fluid_pressure_dirichlet_boundaries(self):
        if self.exact_F:
            return ["0.0 < 1.0"]
        else:
            return []
        #return ["near(x[0], 0.0)"]

    def fluid_velocity_initial_condition(self):
        return self.u_F

    def fluid_pressure_initial_condition(self):
        return self.p_F

    def fluid_body_force(self):
        return self.f_F

    def fluid_boundary_traction(self, V):
        return self.g_F

    def mesh_velocity(self, V):
        w = Function(V)
        return w

    #--- Structure problem ---
    # Use known solution on entire mesh
    def structure(self):
        return Structure()

    def structure_density(self):
        #return 1000000.0
        return 100.0

    def structure_mu(self):
        return 1.0

    def structure_lmbda(self):
        return 2.0

    def structure_dirichlet_values(self):
        return [self.U_S, self.U_S]
        #return [self.U_S]

    def structure_dirichlet_boundaries(self):
        if self.exact_S:
            return ["0.0 < 1.0", "0.0 < 1.0"]
        else:
            return [noslip, right]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    def structure_body_force(self):
        return self.F_S

    def structure_boundary_traction_extra(self):
        return self.G_0

    def structure_initial_conditions(self):
        return (self.U_S, self.P_S)

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
problem = SimpleAnalytic()
goal = problem.solve(application_parameters)

interactive()
