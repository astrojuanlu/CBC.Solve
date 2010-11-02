__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-11-01

from fsiproblem import *

# Create application parameters set
application_parameters = Parameters("application_parameters")
application_parameters.add("ny", 20)
application_parameters.add("T", 0.5)
application_parameters.add("dt", 0.02)
application_parameters.add("w_h", 0.45) 
application_parameters.add("w_k", 0.45)
application_parameters.add("w_c", 0.1)
application_parameters.add("mesh_alpha", 1.0)
application_parameters.add("dorfler_fraction", 0.5)
application_parameters.add("adaptive_tolerance", 0.5)
application_parameters.parse()

# Save parameters to file
parameter_info = application_parameters.option_string()
f = open("adaptivity/channel_flap_parameters.txt", "w")
f.write("Channel with Flap \n \n ")
f.write(parameter_info)
f.close()

# Constants related to the geometry of the channel and the obstruction
channel_length  = 4.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.6
structure_top   = 0.5

# Define boundaries
inflow  = "x[0] < DOLFIN_EPS && \
           x[1] > DOLFIN_EPS && \
           x[1] < %g - DOLFIN_EPS" % channel_height
outflow = "x[0] > %g - DOLFIN_EPS && \
           x[1] > DOLFIN_EPS && \
           x[1] < %g - DOLFIN_EPS" % (channel_length, channel_height)
noslip  = "on_boundary && !(%s) && !(%s)" % (inflow, outflow)
fixed   = "x[1] < DOLFIN_EPS && x[0] > %g - DOLFIN_EPS && x[0] < %g + DOLFIN_EPS" % (structure_left, structure_right)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

class ChannelWithFlap(FSI):

    def __init__(self):

        ny = application_parameters["ny"]
        nx = 4*ny
        mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common parameters ---

    def end_time(self):
        return application_parameters["T"]

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

    def adaptive_tolerance(self):
        return application_parameters["adaptive_tolerance"]

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dt):
        
        # Compute average displacement
        structure_area = (structure_right - structure_left) * structure_top
        displacement = (1.0/structure_area)*assemble(U_S[0]*dx, mesh=U_S.function_space().mesh())

        # Write to file
        f = open("adaptivity/goal_functional.txt", "a")
        f.write("%g %g \n" % (dt, displacement))
        f.close()
        
        # Compute velocity at outflow
        velocity = u_F((4.0, 0.5))[0]
        
        # Print values of functionals
        info("")
        info_blue("Functional 1 (displacement): %g", displacement)
        info_blue("Functional 2 (velocity):     %g", velocity)
        info("")
        
    def __str__(self):
        return "Channel with flap FSI problem"

    #--- Parameters for fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.002

    def fluid_velocity_dirichlet_values(self):
        return [(0, 0)]

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip]

    def fluid_pressure_dirichlet_values(self):
        return 1, 0

    def fluid_pressure_dirichlet_boundaries(self):
        return inflow, outflow

    def fluid_velocity_initial_condition(self):
        return (0, 0)

    def fluid_pressure_initial_condition(self):
        return "1 - x[0]"

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
        return [(0, 0)]

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
        return application_parameters["mesh_alpha"]

# Solve problem
problem = ChannelWithFlap()
problem.parameters["solver_parameters"]["solve_primal"] = True
problem.parameters["solver_parameters"]["solve_dual"] = True
problem.parameters["solver_parameters"]["estimate_error"] = True
problem.parameters["solver_parameters"]["plot_solution"] = False
problem.parameters["solver_parameters"]["tolerance"] = problem.adaptive_tolerance()
#problem.parameters["solver_parameters"]["itertol"] = problem.adaptive_tolerance() * problem.non_galerkin_error_weight()
u_F, p_F, U_S, P_S, U_M = problem.solve()
