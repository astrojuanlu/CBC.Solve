__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-30

from fsiproblem import *

# Command-line parameters
command_line_parameters = Parameters("command_line_parameters")
command_line_parameters.parse()
command_line_parameters.add("ny", 20)
command_line_parameters.parse()

# Constants related to the geometry of the channel and the obstruction
channel_length  = 4.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.6
structure_top   = 0.5

# Define inflow boundary
def inflow(x):
    return \
        x[0] < DOLFIN_EPS and \
        x[1] > DOLFIN_EPS and \
        x[1] < channel_height - DOLFIN_EPS

# Define outflow boundary
def outflow(x):
    return \
        x[0] > channel_length - DOLFIN_EPS and \
        x[1] > DOLFIN_EPS and \
        x[1] < channel_height - DOLFIN_EPS

# Define noslip boundary
def noslip(x, on_boundary):
    return on_boundary and not inflow(x) and not outflow(x)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] >= structure_left  - DOLFIN_EPS and \
            x[0] <= structure_right + DOLFIN_EPS and \
            x[1] <= structure_top   + DOLFIN_EPS

class ChannelWithFlap(FSI):

    def __init__(self):

        # Initialize base class
        FSI.__init__(self)

        # Create the complete mesh
        ny = command_line_parameters["ny"]
        self.Omega = Rectangle(0.0, 0.0, channel_length, channel_height, 4*ny, ny)

        # Create submeshes for fluid and structure
        self.update()

    def update(self):

        # Create cell markers (0 = fluid, 1 = structure)
        D = self.Omega.topology().dim()
        cell_domains = MeshFunction("uint", self.Omega, D)
        cell_domains.set_all(0)
        structure = Structure()
        structure.mark(cell_domains, 1)

        # Extract submeshes for fluid and structure
        self.Omega_F = SubMesh(self.Omega, cell_domains, 0)
        self.Omega_S = SubMesh(self.Omega, cell_domains, 1)

#    def mesh(self):
#        return self.Omega

    #--- Parameters for fluid problem ---

    def fluid_mesh(self):
        return self.Omega_F

    def fluid_viscosity(self):
        return 0.002

    def fluid_density(self):
        return 1.0

    def end_time(self):
        return 1.0

    def fluid_boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0, 0)), noslip)

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(1), inflow)
        bcp1 = DirichletBC(Q, Constant(0), outflow)

        return [bcu], [bcp0, bcp1]

    #--- Parameters for structure problem ---

    def structure_mesh(self):
        return self.Omega_S

    #--- Parameters for mesh problem ---

    def mesh_parameters(self):
        return (3.8461, 5.76, 1.0)

    def __str__(self):
        return "Channel with flap FSI problem"

# Solve problem
problem = ChannelWithFlap()
problem.parameters["solver_parameters"]["plot_solution"] = True
u_F, p_F, U_S, P_S, U_M, P_M = problem.solve(1e-3)
