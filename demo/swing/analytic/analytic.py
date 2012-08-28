__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-04
# Last changed: 2012-05-03

# Modified by Marie E. Rognes

from math import sin
from cbc.swing import *
from right_hand_sides import *
from cbc.swing.parameters import read_parameters
from cbc.swing.fsiproblem import FSI

# Read parameters
application_parameters = read_parameters()

# Used for testing
test = True
if test:
    ref = 0
    application_parameters["mesh_element_degree"] = 1
    application_parameters["structure_element_degree"] = 1
    application_parameters["save_solution"] = False
    application_parameters["solve_primal"] = True
    application_parameters["solve_dual"] = True
    application_parameters["estimate_error"] = True
    application_parameters["plot_solution"] = True
    application_parameters["uniform_timestep"] = True
    application_parameters["uniform_mesh"] = True
    application_parameters["tolerance"] = 1e-16
    application_parameters["iteration_tolerance"] = 1.e-14
    application_parameters["initial_timestep"] = 0.02/ (2**ref)
    application_parameters["output_directory"] = "results-analytic-test"
    application_parameters["global_storage"] = False
    application_parameters["max_num_refinements"] = 0
    application_parameters["fluid_solver"] = "taylor-hood"
    application_parameters["primal_solver"] ="fixpoint" #for newtonsolve use Newtonanalytic.py
##    application_parameters["fluid_solver"] = "ipcs"

    #Dual Solver settings
    application_parameters["dualsolver"]["timestepping"] = "FE"
    application_parameters["dualsolver"]["fluid_domain_time_discretization"] = "mid-point"
else:
    ref = 0

# Define relevant boundaries
right = "near(x[0], 2.0)"
interface = "near(x[0], 1.0)"
left = "near(x[0], 0.0)"
noslip = "near(x[1], 0.0) || near(x[1], 1.0)"

#Exclude FSI Nodes
influid = "x[0] < 1.0 - DOLFIN_EPS"
meshbc = "on_boundary &&" + influid

# Constant used in definition of analytic solutions
C = 1.0

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 1.0 - DOLFIN_EPS

# Define FSI problem
class Analytic(FSI):
    def __init__(self):

        # Create mesh
        mesh = Rectangle(0.0, 0.0, 2.0, 1.0, 10*2**ref, 5*2**ref)

        # Create analytic expressions for various forces
        self.f_F = Expression(cpp_f_F, degree=2)
        self.g_F = Expression(cpp_g_F, degree=1)
        self.F_S = Expression(cpp_F_S, degree=2)
        self.F_M = Expression(cpp_F_M, degree=3)
        self.G_0 = Expression(cpp_G_S0, degree=5)

        # Exact solutions
        self.u_F = Expression(cpp_u_F, degree=2)
        self.p_F = Expression(cpp_p_F, degree=1)
        self.U_S = Expression(cpp_U_S, degree=2)
        self.P_S = Expression(cpp_P_S, degree=2)
        self.U_M = Expression(cpp_U_M, degree=3)

        # Initialize expressions
        forces = [self.f_F, self.g_F, self.F_S, self.F_M, self.G_0]
        solutions = [self.u_F, self.p_F, self.U_S, self.P_S, self.U_M]
        for f in forces + solutions:
            f.C = C
            f.t = 0.0

        # Initialize base class
        FSI.__init__(self, mesh, application_parameters)

    #--- Common ---

    def end_time(self):
        return 0.1

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        return U_S[0] * dx_S

    def reference_value(self):
        T = self.end_time()
        return C*(T - sin(T)) / 6.0

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
        return [noslip]

    def fluid_pressure_dirichlet_values(self):
        if application_parameters["fluid_solver"] == "ipcs":
            return [self.p_F, self.p_F]
        else:
            return []

    def fluid_pressure_dirichlet_boundaries(self):
        if application_parameters["fluid_solver"] == "ipcs":
            return [left, interface]
        else:
            return []

    def fluid_velocity_initial_condition(self):
        return self.u_F

    def fluid_pressure_initial_condition(self):
        return self.p_F

    def fluid_body_force(self):
        return self.f_F

    def fluid_boundary_traction(self):
        return self.g_F

    #--- Structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 100.0

    def structure_mu(self):
        return 1.0

    def structure_lmbda(self):
        return 2.0

    def structure_dirichlet_values(self):
        return [self.U_S, self.U_S]

    def structure_dirichlet_boundaries(self):
        return [noslip, right]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    def structure_body_force(self):
        return self.F_S

    def structure_boundary_traction_extra(self):
        return self.G_0

    def struc_displacement_initial_condition(self):
        return (self.U_S)

    def struc_velocity_initial_condition(self):
        return (self.P_S)    

    #--- Parameters for mesh problem ---

    def mesh_mu(self):
        return 1.0

    def mesh_lmbda(self):
        return 2.0

    def mesh_alpha(self):
        return 1.0

    def mesh_right_hand_side(self):
        return self.F_M

    #GB this helps the Newton solver and could help with the dual
    def mesh_dirichlet_boundaries(self):
        return [meshbc]

if __name__ == "__main__":
    # Define and solve problem
    problem = Analytic()
    goal = problem.solve(application_parameters)
    interactive()
