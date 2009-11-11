# A problem script for a simple fsi-problem

from cbc.flow import *
from cbc.twist import *

# FIXME: Use variables for width and height etc
channel_length  = 3.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.6
structure_top   = 0.5

# Define structure sub domain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
         return x[0] >= structure_left and x[0] <= structure_right \
             and x[1] <= structure_top
    
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

def bottom(x, on_boundary):
    return on_boundary and x[1] < DOLFIN_EPS \
        and x[0] > structure_left - DOLFIN_EPS \
        and x[0] < structure_right + DOLFIN_EPS

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
class StructureProblem(StaticHyperelasticityProblem):
        
    def mesh(self):
        return structure_mesh
     
    def boundary_conditions(self, vector):
        bcu = DirichletBC(vector, Constant(vector.mesh(), (0, 0, 0)), bottom)
        return [bcu]
     
#     def surface_force(self, t, vector):
#         # Need to specify Neumann boundary somewhere
#         T = Expression(("0.0", "0.0", "0.0"), V = vector)
#         T.t = t
#         return T
    
    def material_model(self):
        mu       = 3.8461
        lmbda    = 5.76
        material = StVenantKirchhoff([mu, lmbda])
        return material

    def info(self):
        return "The structure problem"

fluid = FluidProblem()
structure = StructureProblem()

t = 0
T = 1
dt = 0.05

while t < T:

 #     u1, p1 = fluid.step(dt)
#      plot(u1)
#      plot(p1)

#      fluid.update()
#      t += dt
    us = structure.solve()
    plot(us)
    interactive()
