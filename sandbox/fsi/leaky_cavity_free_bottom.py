__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-10-01

from fsiproblem import *

#            WALL (NOSLIP)
#        __________________________
#     ----->                     -----> u_max = (1,0)
#     ---> INFLOW                --->
#     -->                        -->
#        |                        |
#        |                        |
#        |                        |
#        |       FLUID            |
#        |                        |
#        |                        |
#        |                        |  
#        |                        |
#        __________________________ (3.0, 0.5)
#        |                        |
# FIXED  |      STRUCTURE         | FIXED 
#        |                        |
#        -------------------------- (3.0, 3.0)
#                  FREE

# Create application parameters set
application_parameters = Parameters("application_parameters")
application_parameters.add("ny", 30)
application_parameters.add("dt", 0.02)
application_parameters.add("T", 0.5)
application_parameters.add("mesh_alpha", 1.0)
application_parameters.add("space_error_weight", 0.85) 
application_parameters.add("time_error_weight", 0.10)
application_parameters.add("non_galerkin_error_weight", 0.15)
application_parameters.add("dorfler_fraction", 0.6)
application_parameters.parse()

# Constants related to the geometry of the problem
cavity_length  = 3.0
cavity_height  = 3.0
structure_left  = 0.0
structure_right = 3.0
structure_top   = 0.5
inflow_top = 3.0
inflow_bottom = 2.5

# Define boundaries
inflow  = "x[0] == 0.0  && \
           x[1] > %g + DOLFIN_EPS && x[1] < %g - DOLFIN_EPS " %(inflow_bottom, inflow_top)
outflow = "x[0] > %g - DOLFIN_EPS && \
           x[1] > %g - DOLFIN_EPS && x[1] < %g - DOLFIN_EPS"  % (cavity_length, inflow_bottom, inflow_top)
noslip  = "on_boundary && !(%s) && !(%s)" % (inflow, outflow)
fixed_left   = "x[0] == 0.0  && x[1] >= DOFLIN_EPS" 
fixed_right  = "x[0] > %g - DOLFIN_EPS  && x[1] >= 0.0" % structure_right

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

class LeakyDrivenCavityFreeBottom(FSI):

    def __init__(self):

        ny = application_parameters["ny"]
        nx = ny
        mesh = Rectangle(0.0, 0.0, cavity_length, cavity_height, nx, ny)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common parameters ---

    def end_time(self):
        return application_parameters["T"]

    def initial_timestep(self):
        return application_parameters["dt"]

    def dorfler_fraction(self):
        return application_parameters["dorfler_fraction"]

    def space_error_weight(self):
        return application_parameters["space_error_weight"]

    def time_error_weight(self):
        return application_parameters["time_error_weight"]

    def non_galerkin_error_weight(self):
        return application_parameters["non_galerkin_error_weight"]

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, at_end):

        # Only evaluate functional at the end time
        if not at_end: return
        
        # Compute average displacement
        structure_area = (structure_right - structure_left) * structure_top
        displacement = (1.0/structure_area)*assemble(U_S[0]*dx, mesh=U_S.function_space().mesh())

        # Print values of functionals
        info("")
        info_blue("Functional (displacement): %g", displacement)
        info("")
        
    def __str__(self):
        return "Lid Driven Cavity with an Elastic Bottom" 

    #--- Parameters for fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        return [(0,0), Expression(("x[1]*(x[1] - 2.5) - 0.16", "0.0"))] 

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip, inflow]

    def fluid_pressure_dirichlet_values(self):
        return [0]

    def fluid_pressure_dirichlet_boundaries(self):
        return [inflow]

    def fluid_velocity_initial_condition(self):
        return (0, 0)

    def fluid_pressure_initial_condition(self):
        return 0

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
        return [(0,0), (0,0)]

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
problem = LeakyDrivenCavityFreeBottom()
problem.parameters["solver_parameters"]["solve_primal"] = True
problem.parameters["solver_parameters"]["solve_dual"]  =  False
problem.parameters["solver_parameters"]["estimate_error"] = False
problem.parameters["solver_parameters"]["plot_solution"] = False
problem.parameters["solver_parameters"]["tolerance"] = 0.5
u_F, p_F, U_S, P_S, U_M = problem.solve()

