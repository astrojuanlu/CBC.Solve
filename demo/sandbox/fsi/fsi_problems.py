# A problem script for a simple fsi-problem

from cbc.flow import *
from cbc.twist import *

#######################################################################################
# MESH AND BOUNDARY DEFINITION
#######################################################################################

# Create structure sub domain
# FIX: Use variables!
class Structure(SubDomain):
     def inside(self, x, on_boundary):
          return x[0] >= 1.4 and x[0] <= 1.6 and x[1] <= 0.5

# Create mesh 
# FIX: Use variables!
mesh = Rectangle(0.0, 0.0, 3.0, 1.0, 60, 20)
    
# Create sub domain markers, 0 for fluid, 1 for structure
sub_domains = MeshFunction("uint", mesh, mesh.topology().dim())
sub_domains.set_all(0)
structure = Structure()
structure.mark(sub_domains, 1)

# Extract sub meshes for fluid and structure
structure_mesh = SubMesh(mesh, sub_domains, 1)
fluid_mesh = SubMesh(mesh, sub_domains, 0)  

# Extract matching indices for fluid and structure
structure_to_fluid = compute_vertex_map(structure_mesh, fluid_mesh)

# Define inflow boundary for the fluid
class InflowBoundary(SubDomain):
     def inside(self, x, on_boundary):
          return on_boundary and x[0] < DOLFIN_EPS

# Define outflow boundary for the fluid
# FIX: Use variables!
class OutflowBoundary(SubDomain):
     def inside(self, x, on_boundary):
          return on_boundary and x[0] > 3.0 - DOLFIN_EPS

# Define noslip boundary for the fluid
# FIX: Use variables!
class NoslipBoundary(SubDomain):
     def inside(self, x, on_boundary):
          on_inflow  = x[0] == 0.0 and x[1] > 0.0 and x[1] < 1.0
          on_outflow = x[0] == 3.0 and x[1] > 0.0 and x[1] < 1.0
          return on_boundary and not on_inflow and not on_outflow


# Create no-slip boundary for the fluid
noslip_boundary = NoslipBoundary()

# Create inflow-boundary for the fluid
inflow_boundary = InflowBoundary()

# Create inflow-boundary for the fluid
outflow_boundary = OutflowBoundary()



#######################################################################################
# SOLVERS
#######################################################################################


# Define fluid solver
class FluidProblem(NavierStokesProblemStep):

     def intial_conditions(self, V, Q):
         "Return initial conditions for velocity and pressure"
         self.u0 = Constant(fluid_mesh, (0, 0))
         self.p0 = Expression("1 - x[0]", V = Q)

     def mesh(self):
          return fluid_mesh

     def viscosity(self):
          return 1.0 / 8.0

     def boundary_conditions(self, V, Q):
          
          # Create no-slip boundary condition for velocity
          bcu = DirichletBC(V, Constant(V.mesh(), (0, 0)), noslip_boundary)

          # Create inflow and outflow boundary conditions for pressure
          bcp0 = DirichletBC(Q, Constant(Q.mesh(), 1), inflow_boundary)
          bcp1 = DirichletBC(Q, Constant(Q.mesh(), 0), outflow_boundary)

          return [bcu], [bcp0, bcp1]

     def time_step(self):
          return 0.05

     def info(self):
          return "Pressure driven channel (2D) with an obstructure"


# Define struture solver
class StructureProblem(StaticHyperelasticityProblem):
     
     def __init__(self):
          print "Structure solver not implemented yet..."
          pass



# # Solve fluid problem
# fluid = FSI_FLUID()
# fluid.solve()


if __name__ == "__main__":

    problem = fsi_problems()
    fsi.solve(problem)
