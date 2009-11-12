# A simple FSI problem involving a hyperelastic obstruction in a
# Navier-Stokes flow field. Lessons learnt from this exercise will be
# used to construct an FSI class in the future.

from cbc.flow import *
from cbc.twist import *

# Constants related to the geometry of the channel and the obstruction
channel_length  = 3.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.6
structure_top   = 0.5
nx = 60
ny = 20
    
# Create the complete mesh
mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= structure_left) and (x[0] <= structure_right) \
            and (x[1] <= structure_top)
    
# Create subdomain markers (0: fluid, 1: structure)
sub_domains = MeshFunction("uint", mesh, mesh.topology().dim())
sub_domains.set_all(0)
structure = Structure()
structure.mark(sub_domains, 1)

# Extract submeshes for fluid and structure
fluid_mesh = SubMesh(mesh, sub_domains, 0)  
structure_mesh = SubMesh(mesh, sub_domains, 1)

# Extract matching indices for fluid and structure
structure_to_fluid = compute_vertex_map(structure_mesh, fluid_mesh)

# Define inflow boundary
def inflow(x):
    return x[0] < DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < channel_height - DOLFIN_EPS

# Define outflow boundary
def outflow(x):
    return x[0] > channel_length - DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < channel_height - DOLFIN_EPS

# Define noslip boundary
def noslip(x, on_boundary):
    return on_boundary and not inflow(x) and not outflow(x)

# Define fluid problem
class FluidProblem(NavierStokesProblem):
    
    def mesh(self):
        return fluid_mesh

    def viscosity(self):
        return 1.0 / 8.0
    
    def end_time(self):
        return 0.5
    
    def boundary_conditions(self, V, Q):
        
        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant(V.mesh(), (0, 0)), noslip)
        
        # FIXME: Anders fix DirichletBC to take int or float instead of Constant
        
        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(Q.mesh(), 1), inflow)
        bcp1 = DirichletBC(Q, Constant(Q.mesh(), 0), outflow)

        return [bcu], [bcp0, bcp1]
    
    def info(self):
        return "Pressure driven channel (2D) with an obstructure"

    def time_step(self):
        return 0.05

# Define struture problem
class StructureProblem(HyperelasticityProblem):
        
    def mesh(self):
        return structure_mesh

    def dirichlet_conditions(self, vector):
        fix = Expression(("0.0", "0.0"), V = vector)
        return [fix]

    def dirichlet_boundaries(self):
        #FIXME: Figure out how to use the constants above in the
        #following boundary definitions
        bottom = "x[1] == 0.0 && x[0] >= 1.4 && x[0] <= 1.6"
        return [bottom]

    def neumann_conditions(self, vector):
        pull = Expression(("force*t", "0.0"), V = vector)
        pull.force = 0.02
        return [pull]

    def neumann_boundaries(self):
        # Return the entire structure boundary as the Neumann
        # boundary, knowing that the Dirichlet boundary will overwrite
        # it at the bottom
        return["on_boundary"]

    def material_model(self):
        mu       = 3.8461
        lmbda    = 5.76
        material = StVenantKirchhoff([mu, lmbda])
        return material

    def time_step(self):
        return 0.05

    def __str__(self):
        return "The structure problem"

fluid = FluidProblem()
structure = StructureProblem()

t = 0
T = 1
dt = 0.05

while t < T:

    print "Solving the problem at t = ", str(t)
    
    u, p = fluid.step(dt)
    plot(u)
    plot(p)
    fluid.update()

    w = structure.step(dt)
    structure.update()

    t += dt
interactive()
