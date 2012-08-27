__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-04-08

from dolfin import *
from cbc.swing import *
from cbc.swing.parameters import read_parameters
from cbc.swing.fsiproblem import FSI

# Read parameters
application_parameters = read_parameters()

# Used for testing
test = True
if test:
    application_parameters["primal_solver"] = "fixpoint"
    application_parameters["output_directory"] = "results_channel_with_flap_test"
    application_parameters["global_storage"] = True
    application_parameters["solve_primal"] = True
    application_parameters["solve_dual"] = False
    application_parameters["estimate_error"] = False
    application_parameters["uniform_timestep"] = True
    application_parameters["plot_solution"] = True
    application_parameters["max_num_refinements"] = 0
    application_parameters["initial_timestep"] = 0.02 / 8.0
    application_parameters["iteration_tolerance"] = 1.0e-6
    application_parameters["fluid_solver"] = "ipcs" #"taylor-hood"
     
application_parameters["FSINewtonSolver"]["optimization"]["reuse_jacobian"] = False
application_parameters["FSINewtonSolver"]["optimization"]["simplify_jacobian"] = False
application_parameters["FSINewtonSolver"]["optimization"]["reduce_quadrature"] = 0
application_parameters["FSINewtonSolver"]["jacobian"]= "buff"

# Constants related to the geometry of the problem
channel_length  = 4.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.8
structure_top   = 0.6

# Define boundaries
inflow  = "x[0] < DOLFIN_EPS && \
           x[1] > DOLFIN_EPS && \
           x[1] < %g - DOLFIN_EPS" % channel_height
outflow = "x[0] > %g - DOLFIN_EPS && \
           x[1] > DOLFIN_EPS && \
           x[1] < %g - DOLFIN_EPS" % (channel_length, channel_height)
fixed = "x[1] < DOLFIN_EPS && x[0] > %g - DOLFIN_EPS && x[0] < %g + DOLFIN_EPS" % (structure_left, structure_right)

if application_parameters["primal_solver"] == "fixpoint":
    noslip = "on_boundary && !(%s) && !(%s) && !(%s)" % (inflow, outflow,fixed)
else:
    noslip = "on_boundary && !(%s) && !(%s)" % (inflow, outflow)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

class DoNothing(SubDomain):
    def inside(self, x ,on_boundary):
        return \
           x[0] < DOLFIN_EPS or \
           x[0] > channel_length - DOLFIN_EPS and \
           x[1] > DOLFIN_EPS and \
           x[1] <  channel_height - DOLFIN_EPS           

class ChannelWithFlap(FSI):
    def __init__(self):

        # Create mesh
        ny = 5
        nx = 20
        if application_parameters["crossed_mesh"]:
            mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny, "crossed")
        else:
            mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)

        for i in range(3):
            mesh = refine(mesh)

        # Set material parameters
        self.E = 100.0
        self.nu = 0.3

        # Initialize base class
        FSI.__init__(self, mesh, application_parameters)

    #--- Common ---

    def end_time(self):
        return 0.1

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        A = (structure_right - structure_left) * structure_top
        return (1.0/A) * U_S[0] * dx_S

    def __str__(self):
        return "Channel flow with an immersed elastic flap"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.001
    
    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)
    
    def fluid_velocity_dirichlet_values(self):
        return [(0.0, 0.0)]
    
    def fluid_velocity_dirichlet_boundaries(self):
       return [noslip]

##    def fluid_pressure_dirichlet_values(self):
##        return [10.0, 0.0]

##    def fluid_pressure_dirichlet_boundaries(self):
##        return [inflow, outflow]
##
    def fluid_pressure_dirichlet_values(self):
        return [10.0,0.0]
    
    def fluid_pressure_dirichlet_boundaries(self):
        return [inflow,outflow]

    def fluid_pressure_initial_condition(self):
        return "10.0*(1.0 - x[0]/%f)"%channel_length
    
    def fluid_donothing_boundaries(self):
        return [DoNothing()]

    #--- Structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 100.0

    def structure_mu(self):
        return self.E / (2.0*(1.0 + self.nu))

    def structure_lmbda(self):
        return self.E * self.nu / ((1.0 + self.nu)*(1 - 2*self.nu))

    def structure_dirichlet_values(self):
        return [(0.0, 0.0)]

    def structure_dirichlet_boundaries(self):
        return [fixed]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    #--- Parameters for mesh problem ---

    def mesh_mu(self):
        return 1.0

    def mesh_lmbda(self):
        return 1.0

    def mesh_alpha(self):
        return 1.0

# Define and solve problem
if __name__ == "__main__":
    problem = ChannelWithFlap()
    problem.solve(application_parameters)
    interactive()
