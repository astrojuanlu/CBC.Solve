__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-04
# Last changed: 2012-04-30

from cbc.swing import *
from right_hand_sides_revised import *

# Read parameters
application_parameters = read_parameters()

# Used for testing
test = True
ref = 2
if test:
    application_parameters["save_solution"] = True
    application_parameters["solve_primal"] = True
    application_parameters["solve_dual"] = False
    application_parameters["estimate_error"] = False
    application_parameters["plot_solution"] = True
    application_parameters["uniform_timestep"] = True
    application_parameters["uniform_mesh"] = True
    application_parameters["tolerance"] = 1e-16
    #application_parameters["fixedpoint_tolerance"] = 1e-10
    application_parameters["initial_timestep"] = 0.01/2**(ref)
    application_parameters["output_directory"] = "results_analytic_fluid_test"
    application_parameters["max_num_refinements"] = 0
    application_parameters["use_exact_solution"] = False

    application_parameters["fluid_solver"] = "taylor-hood"
    #application_parameters["fluid_solver"] = "ipcs"

# Define boundaries
top = "near(x[1], 1.0)"
noslip = "near(x[0], 0.0) || near(x[0], 1.0)"
fixed  = "near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0)"

# Constant used in definition of analytic solutions
C = 1.0

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.5 + DOLFIN_EPS

class Analytic(FSI):

    def __init__(self):

        # Create mesh
        n = 8*2**ref
        mesh = UnitSquare(n, n)
        print mesh.hmin()

        # Create analytic expressions for various forces
        self.f_F = Expression(cpp_f_F, degree=2)
        self.f_F.C = C
        self.g_F = Expression(cpp_g_F, degree=1)
        self.g_F.C = C
        self.F_S = Expression(cpp_F_S, degree=6)
        self.F_S.C = C
        self.F_M = Expression(cpp_F_M, degree=2)
        self.F_M.C = C
        self.G_0 = Expression(cpp_G_S0, degree=2)
        self.G_0.C = C

        # Helpers
        # Exact mesh velocity
        self.P_M = Expression(cpp_P_M, degree=2)
        self.P_M.C = C
        self.p_F = Expression(cpp_p_F, degree=1)
        self.p_F.C = C
        self.U_S = Expression(cpp_U_S, degree=2)
        self.U_S.C = C
        self.u_F = Expression(cpp_u_F, degree=2)
        self.u_F.C = C
        self.U_M = Expression(cpp_U_M, degree=2)
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
        self.g_F.t = t   # To be used as top boundary force for IPCS if desired
        self.f_F.t = t   # Used as body force for IPCS; checked.
        self.F_S.t = t   # Body force for the structure; looks ok from paper.
        self.F_M.t = t   # Body force for the mesh; looks ok from paper.
        self.G_0.t = t   # Used for extra stress exerted by fluid; ok.

        self.p_F.t = t1 # Used as bc for pressure if given, checked.
        self.U_M.t = t1 # Used as bc for mesh if given.
        self.P_M.t = t1 # Used as mesh velocity if specified
        self.U_S.t = t1 # Used as bc for structure if given.
        self.u_F.t = t1 # Used as bc for fluid velocity, checked.

    def exact_solution(self):
        return self.u_F, self.p_F, self.U_S, self.P_M, self.U_M

    def __str__(self):
        return "Channel flow with an immersed elastic flap"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        #return [(0.0, 0.0), self.u_F]
        #return [(0.0, 0.0)]
        return [self.u_F]#, self.u_F]

    def fluid_velocity_dirichlet_boundaries(self):
        #return [noslip, top]
        #return [noslip]
        return [noslip]#, "on_boundary && !near(x[1], 1.0)"]

    def fluid_pressure_dirichlet_values(self):
        #return [self.p_F]
        return []

    def fluid_pressure_dirichlet_boundaries(self):
        #return ["near(x[0], 1.0)"]
        return []

    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
        return 0.0

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
        return 100.0

    def structure_mu(self):
        return 1.0

    def structure_lmbda(self):
        return 2.0

    def structure_dirichlet_values(self):
        return [self.U_S]

    def structure_dirichlet_boundaries(self):
        return [fixed]
        #return ["x[0] < 2.0"]

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
problem = Analytic()
goal = problem.solve(application_parameters)
if goal is None:
    print "Hm. Goal is None -- why?"
    exit()

(M_h_T, M_h) = goal

#M = C / (24.0*pi)
# Print reference value of functional at t = 0.1:
M =  (12*pi*C - 120*C*cos(pi/10)*sin(pi/10))/(1440*pi**2)

print "Exact value of goal functional:", M
print "Approximate value of goal functional:", M_h
print "Exact error:", (M - M_h)

interactive()
