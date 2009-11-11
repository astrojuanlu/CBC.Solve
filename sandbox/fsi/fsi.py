# A problem script for a simple fsi-problem

from cbc.flow import *
from cbc.twist import *

# FIXME: Use variables for width and height etc
channel_length = 3.0
channel_height = 1.0

# Define structure sub domain
class Structure(SubDomain):
     def inside(self, x, on_boundary):
          return x[0] >= 1.4 and x[0] <= 1.6 and x[1] <= 0.5

# Create mesh 
mesh = Rectangle(0.0, 0.0, channel_length, channel_height, 60, 20)
    
# Create sub domain markers, 0 for fluid, 1 for structure
sub_domains = MeshFunction("uint", mesh, mesh.topology().dim())
sub_domains.set_all(0)
structure = Structure()
structure.mark(sub_domains, 1)

# Extract sub meshes for fluid and structure
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


# Define fluid solver
class FluidProblem(NavierStokesProblem):
                 
     def mesh(self):
          return fluid_mesh

     def viscosity(self):
          return 1.0 / 8.0

     def end_time(self):
          return 0.5

     #def initial_conditions(self, V, Q):
     #     u0 = Constant(V.mesh(), (0, 0))
     #     p0 = Expression("1 - x[0]", V=Q)
     #     return u0, p0

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

# Define struture solver
class StructureProblem(StaticHyperelasticityProblem):
     
     def __init__(self):
          print "not implemented yet..."
          pass


fluid = FluidProblem()

t = 0
T = 1
dt = 0.05

while t < T:

     u1, p1 = fluid.step(dt)
     plot(u1)
     plot(p1)

     fluid.update()
     t += dt

interactive()
