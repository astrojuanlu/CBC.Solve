__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-12-08

from fsiproblem import *

# ------------------------------------------ 
#                 NOSLIP
#  p = 1                            p = 0
#                         
#  NOSLIP                           NOSLIP
# -------                          ---------
#        |                        |
#        |                        |
#        |                        |
#        |         FLUID          |
#        |                        |
#        |                        |
#        |                        |  
#        |                        |
#        __________________________ (1.0, 0.25)
#        |                        |
# FIXED  |       STRUCTURE        |  FIXED 
#        |                        |
#        -------------------------- (1.0, 0.0)
#                  FREE

# Create application parameters set
application_parameters = Parameters("application_parameters")
application_parameters.add("end_time", 0.25)
application_parameters.add("dt", 0.02)
application_parameters.add("ny", 1)
application_parameters.add("TOL", 0.1)
application_parameters.add("w_h", 0.1) 
application_parameters.add("w_k", 0.85)
application_parameters.add("w_c", 0.05)
application_parameters.add("fraction", 0.5)
application_parameters.add("mesh_alpha", 1.0)
application_parameters.add("solve_primal", True)
application_parameters.add("solve_dual", True)
application_parameters.add("estimate_error", True)
application_parameters.add("dorfler_marking", False)
application_parameters.add("uniform_timestep", False)
application_parameters.add("fixed_point_tol", 1e-12)
application_parameters.parse()

# Collect parameters
parameter_info = application_parameters.option_string()

# Constants related to the geometry of the problem
cavity_length  = 1.0
cavity_height  = 1.5
structure_left  = 0.0
structure_right = 1.0
structure_top   = 0.25
inflow_top = 1.5
inflow_bottom = 1.25

# Define boundaries 
inflow        = "x[0] < DOLFIN_EPS && x[1] > %g + DOLFIN_EPS &&  x[1] < %g - DOLFIN_EPS" %(inflow_bottom, inflow_top)
outflow       = "x[0] > %g - DOLFIN_EPS && \
                 x[1] > %g - DOLFIN_EPS && x[1] < %g - DOLFIN_EPS"  %(cavity_length, inflow_bottom, inflow_top)
fixed_left    = "x[0] == 0.0  && x[1] >= DOFLIN_EPS" 
fixed_right   = "x[0] > %g - DOLFIN_EPS  && x[1] >= 0.0" %structure_right
noslip        = "on_boundary && !(%s) && !(%s) " %(inflow, outflow)

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

        # Define mesh based on a scale factor 
        scale_factor = application_parameters["ny"]
        ny = 30 * scale_factor
        nx = 20 * scale_factor
        mesh = Rectangle(0.0, 0.0, cavity_length, cavity_height, nx, ny)

        # Save original mesh
        file = File("adaptivity/mesh_0.xml")
        file << mesh

        # Report problem parameters
        mesh_size = mesh.hmin()
        f = open("adaptivity/pressure_driven_cavity.txt", "w")
        f.write(parameter_info)
        f.write(str("Mesh size:  ") + (str(mesh_size)) + "\n \n")
        f.close()

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Solver options ---

    def solve_primal(self):
        return application_parameters["solve_primal"]

    def solve_dual(self):
        return application_parameters["solve_dual"]

    def estimate_error(self):
        return application_parameters["estimate_error"]
    
    def dorfler_marking(self):
        return application_parameters["dorfler_marking"]

    def uniform_timestep(self):
        return application_parameters["uniform_timestep"]

    #--- Common parameters ---

    def end_time(self):
        return application_parameters["end_time"]

    def TOL(self):
        return application_parameters["TOL"]

    def initial_timestep(self):
        return application_parameters["dt"]

    def space_error_weight(self):
        return application_parameters["w_h"]

    def time_error_weight(self):
        return application_parameters["w_k"]

    def non_galerkin_error_weight(self):
        return application_parameters["w_c"]

    def fraction(self):
        return application_parameters["fraction"]

    def fixed_point_tol(self):
        return application_parameters["fixed_point_tol"]

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dt):

        # Compute displacement in y-direction
        displacement = assemble(U_S[1]*dx, mesh=U_S.function_space().mesh())
        return displacement

    def __str__(self):
        return "Pressure driven cavity with an elastic bottom" 

    #--- Parameters for fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.01

    def fluid_velocity_dirichlet_values(self):
        return [(0.0, 0.0)]

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip]

    def fluid_pressure_dirichlet_values(self):
        return [1.0, 0.0]
#        return [Expression("1.0 - x[0]"), 0.0]

    def fluid_pressure_dirichlet_boundaries(self):
        return [inflow, outflow]

    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
        return 0.0

    #--- Parameters for structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 3.0

    def structure_mu(self):
        return 3.0

    def structure_lmbda(self):
        return 3.0

    def structure_dirichlet_values(self):
        return [(0.0, 0.0), (0.0, 0.0)]

    def structure_dirichlet_boundaries(self):
        return [fixed_left, fixed_right]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    #--- Parameters for mesh problem ---

    def mesh_mu(self):
        return 3.8461

    def mesh_lmbda(self):
        return 5.76

    def mesh_alpha(self):
        return application_parameters["mesh_alpha"]

# Define problem
problem = PressureDrivenCavity()
problem.parameters["solver_parameters"]["solve_primal"] = problem.solve_primal()
problem.parameters["solver_parameters"]["solve_dual"] = problem.solve_dual() 
problem.parameters["solver_parameters"]["estimate_error"] = problem.estimate_error()
problem.parameters["solver_parameters"]["uniform_timestep"]  = problem.uniform_timestep()
problem.parameters["solver_parameters"]["tolerance"] = problem.TOL()
problem.parameters["solver_parameters"]["fixed_point_tol"] = problem.fixed_point_tol()

# Solve problem
u_F, p_F, U_S, P_S, U_M = problem.solve()

