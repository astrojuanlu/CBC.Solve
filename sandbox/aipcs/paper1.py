__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-10-01

from fsiproblem import *

#     -------------------------------> u_max = (1,0)
#     ---> INFLOW                --->
#     -->                        -->   u_min = (0.0)
#        |                        |
#        |                        |
#        |                        |
#        |         FLUID          |
#        |                        |
#        |                        |
#        |                        |  
#        |                        |
#        __________________________ (3.0, 0.5)
#        |                        |
# FIXED  |      STRUCTURE         | FIXED 
#        |                        |
#        -------------------------- (3.0, 3.0)
#                 FREE

# Create application parameters set
application_parameters = Parameters("application_parameters")
application_parameters.add("end_time", 2.5)
application_parameters.add("dt", 0.02)
application_parameters.add("ny", 30)
application_parameters.add("TOL", 0.5)
application_parameters.add("w_h", 0.1) 
application_parameters.add("w_k", 0.85)
application_parameters.add("w_c", 0.05)
application_parameters.add("dorfler_fraction", 0.2)
application_parameters.add("mesh_alpha", 1.0)
application_parameters.add("solve_primal", True)
application_parameters.add("solve_dual", False)
application_parameters.add("estimate_error", False)
application_parameters.add("uniform_timestep",True)
application_parameters.add("fixed_point_tol", 1e-7)
application_parameters.parse()

# Collect parameters
parameter_info = application_parameters.option_string()

# Constants related to the geometry of the problem
cavity_length  = 3.0
cavity_height  = 3.0
structure_left  = 0.0
structure_right = 3.0
structure_top   = 0.5
inflow_top = 3.0
inflow_bottom = 2.5

# Define boundaries 
inflow        = "x[1] > %g + DOLFIN_EPS &&  x[1] < %g - DOLFIN_EPS" %(inflow_bottom, inflow_top)
outflow       = "x[0] > %g - DOLFIN_EPS && \
                 x[1] > %g - DOLFIN_EPS && x[1] < %g - DOLFIN_EPS"  % (cavity_length, inflow_bottom, inflow_top)
fixed_left    = "x[0] == 0.0  && x[1] >= DOFLIN_EPS" 
fixed_right   = "x[0] > %g - DOLFIN_EPS  && x[1] >= 0.0" % structure_right
fixed_bottom  = "x[0] == 0.0" 
flow_top      = "x[1] == %g" %cavity_height
noslip        = "on_boundary && !(%s) && !(%s) && !(%s) " % (inflow, outflow, flow_top)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

# Define problem class
class Paper1(FSI):
    def __init__(self):

        ny = application_parameters["ny"]
        nx = ny
        mesh = Rectangle(0.0, 0.0, cavity_length, cavity_height, nx, ny)

        # Report problem parameters
        mesh_size = mesh.hmin()
        f = open("adaptivity/paper1.txt", "w")
        f.write("=========PAPER I ========= \n \n \n")
        f.write(str("Mesh size:  ") + (str(mesh_size)) + "\n \n")
        f.write(parameter_info)
        f.close()
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

    def dorfler_fraction(self):
        return application_parameters["dorfler_fraction"]

    def fixed_point_tol(self):
        return application_parameters["fixed_point_tol"]

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dt):

        # Compute average displacement
        structure_area = (structure_right - structure_left) * structure_top
#        structure_area = 1.0
        displacement = (1.0/structure_area)*assemble(U_S[0]*dx + U_S[1]*dx, mesh=U_S.function_space().mesh())

        return displacement

    def __str__(self):
        return "Paper I" 

    #--- Parameters for fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        return [(0.0, 0.0), Expression(("x[1]*(x[1] - 2.5)*(2.0/3.0)", "0.0")), (1.0, 0.0)]

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip, inflow, flow_top]

    def fluid_pressure_dirichlet_values(self):
        return [0.0, 0.0]

    def fluid_pressure_dirichlet_boundaries(self):
        return inflow, outflow

    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
        return 0.0

    #--- Parameters for structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 0.25*15.0

    def structure_mu(self):
        return 0.25*75.0

    def structure_lmbda(self):
        return 0.25*125.0

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

# Solve problem
problem = Paper1()
problem.parameters["solver_parameters"]["solve_primal"] = problem.solve_primal()
problem.parameters["solver_parameters"]["solve_dual"] = problem.solve_dual() 
problem.parameters["solver_parameters"]["estimate_error"] = problem.estimate_error()
#problem.parameters["solver_parameters"]["plot_solution"] =  False
problem.parameters["solver_parameters"]["uniform_timestep"]  = problem.uniform_timestep()
problem.parameters["solver_parameters"]["tolerance"] = problem.TOL()
problem.parameters["solver_parameters"]["fixed_point_tol"] = problem.fixed_point_tol()
u_F, p_F, U_S, P_S, U_M = problem.solve()

