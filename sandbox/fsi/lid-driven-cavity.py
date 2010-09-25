__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-23

from fsiproblem import *

# Create application parameters set
application_parameters = Parameters("application_parameters")
application_parameters.add("ny", 20)
application_parameters.add("dt", 0.02)
application_parameters.add("T", 0.06)
application_parameters.add("mesh_alpha", 1.0)
#application_parameters.add("smooth", 50)
application_parameters.add("dorfler_fraction", 0.5)
application_parameters.parse()

# Print command-line option string
print "\nCommand-line option string"
print application_parameters.option_string()

# Constants related to the geometry of the channel and the obstruction
cavity_length  = 2.0
cavity_height  = 2.0
structure_left  = 0.0
structure_right = 2.0
structure_top   = 0.5

#   __________________________
#   |                        |
#   |                        |
#   |                        |
#   |       FLUID            |
#   |                        |
#   |                        |
#   |                        |  
#   |                        |
#   __________________________ (2.0, 0.5)
#   |                        |
#   |      STRUCTURE         |
#   |                        |
#   -------------------------- (2.0, 2.0)

# Define boundaries
inflow  = "x[0] > DOLFIN_EPS && \
           x[1] > %g - DOLFIN_EPS"  % cavity_height
outflow = "x[0] > %g - DOLFIN_EPS && \
           x[1] > %g - DOLFIN_EPS"  % (cavity_length, cavity_height)
noslip  = "on_boundary && !(%s) && !(%s)" % (inflow, outflow)
fixed_left   = "x[0] > DOLFIN_EPS && x[1] < %g - DOLFIN_EPS" % structure_top
fixed_right  = "x[0] > %g -DOLFIN_EPS  && x[1] < %g - DOLFIN_EPS" % (structure_top, structure_top) 


# "x[1] < DOLFIN_EPS && x[0] > %g - DOLFIN_EPS && x[0] < %g + DOLFIN_EPS" % (structure_left, structure_right)


# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] > structure_left  - DOLFIN_EPS and \
            x[0] < structure_right + DOLFIN_EPS and \
            x[1] < structure_top   + DOLFIN_EPS

class LidDrivenCavity(FSI):

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

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, at_end):

        # Only evaluate functional at the end time
        if not at_end: return
        
        # Compute average displacement
        structure_area = (structure_right - structure_left) * structure_top
        displacement = (1.0/structure_area)*assemble(U_S[0]*dx, mesh=U_S.function_space().mesh())

        # Compute velocity at outflow
        velocity = u_F((1.0, 1.0))[0]
        
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
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        return [(0,0), (1,0), (1,0)]

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip, inflow, outflow]

    def fluid_pressure_dirichlet_values(self):
        return [0, 0]

    def fluid_pressure_dirichlet_boundaries(self):
        return inflow, outflow

    def fluid_velocity_initial_condition(self):
        return (0, 0)

    def fluid_pressure_initial_condition(self):
        return 0

    #--- Parameters for structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 15.0

    def structure_mu(self):
        return 75.0

    def structure_lmbda(self):
        return 125.0

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
problem = LidDrivenCavity()
problem.parameters["solver_parameters"]["solve_primal"] = True
problem.parameters["solver_parameters"]["solve_dual"]  = True
problem.parameters["solver_parameters"]["estimate_error"] = True
problem.parameters["solver_parameters"]["plot_solution"] = True
problem.parameters["solver_parameters"]["tolerance"] = 0.01
u_F, p_F, U_S, P_S, U_M = problem.solve()

