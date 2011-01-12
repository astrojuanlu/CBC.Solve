__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-11-01

from fsiproblem import *


#   REGULIZED VELOCITY PROFILE     
#  
#   ------------------------->
#   
#   |                        |
#   |                        |
#   |                        |
#   |        FLUID           |
#   |                        |
#   |                        |
#   |                        |  
#   |                        |
#   __________________________ (2.0, 0.5)
#   |                        |
#   |      STRUCTURE         |
#   |                        |
#   -------------------------- (2.0, 2.0)
#        FIXED BOTTOM

# Create application parameters set
application_parameters = Parameters("application_parameters")
application_parameters.add("end_time", 0.25)
application_parameters.add("dt", 0.02)
application_parameters.add("ny", 30)
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

# Save parameters to file
parameter_info = application_parameters.option_string()

# Constants related to the geometry of the problem
cavity_length  = 2.0
cavity_height  = 2.0
structure_left  = 0.0
structure_right = 2.0
structure_top   = 0.5

# Define boundaries
inflow_left =   "x[0] == 0.0 && x[0] < 0.25 - DOLFIN_EPS &&\
                 x[1] > %g - DOLFIN_EPS  " % cavity_height
inflow_middle = "x[0] > 0.25 + DOLFIN_EPS  && x[0] < 1.75 - DOLFIN_EPS &&\
                 x[1] > %g - DOLFIN_EPS " % cavity_height
inflow_right = " x[0] > 1.75 && x[0] <= %g &&\
                 x[1] > %g - DOLFIN_EPS " % (cavity_length, cavity_height)
noslip  =        "on_boundary && !(%s) && !(%s) && !(%s)" % (inflow_left, inflow_middle, inflow_right)
fixed_left   =   "x[0] == 0.0  && x[1] >= DOFLIN_EPS" 
fixed_right  =   "x[0] > %g - DOLFIN_EPS  && x[1] >= 0.0" % structure_right
fixed_bottom =   "x[1] == 0.0"

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

class DrivenCavityFixedBottom(FSI):

    def __init__(self):

        ny = application_parameters["ny"]
        nx = ny
        mesh = Rectangle(0.0, 0.0, cavity_length, cavity_height, nx, ny)

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
        return "Lid Driven Cavity with an Elastic Fixed Bottom" 

    #--- Parameters for fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        return [(0,0), Expression(("0.5*4*x[1]", "0.0")), Expression(("0.5*1", "0.0")), Expression(("0.5*4*(2 - x[1]) ", "0.0"))]

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip, inflow_left, inflow_middle, inflow_right]

    def fluid_pressure_dirichlet_values(self):
        return [0, 0, 0]

    def fluid_pressure_dirichlet_boundaries(self):
        return [inflow_left, inflow_middle, inflow_right]

    def fluid_velocity_initial_condition(self):
        return (0, 0)

    def fluid_pressure_initial_condition(self):
        return 0.0

    #--- Parameters for structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 3.0 

    def structure_mu(self):
        return 1.0

    def structure_lmbda(self):
        return 1.0

    def structure_dirichlet_values(self):
        return [(0,0), (0,0), (0,0)]

    def structure_dirichlet_boundaries(self):
        return [fixed_left, fixed_right, fixed_bottom]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    #--- Parameters for mesh problem ---

    def mesh_mu(self):
        return 3.8461

    def mesh_lmbda(self):
        return 5.76

    def mesh_alpha(self):
        return application_parameters["mesh_alpha"]

# Define and solve
problem = DrivenCavityFixedBottom()

# Plot solution
problem.parameters["solver_parameters"]["plot_solution"] = False

# Solve problem
u_F, p_F, U_S, P_S, U_M = problem.solve()

