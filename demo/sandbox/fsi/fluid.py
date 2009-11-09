from cbc.flow import *


def noslip_boundary(x):
     return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

def inflow_boundary(x):
    return x[0] < DOLFIN_EPS

def outflow_boundary(x):
    return x[0] > 3 - DOLFIN_EPS


# Create structure sub domain
class Structure(SubDomain):
     def inside(self, x, on_boundary):
          return x[0] >= 1.4 and x[0] <= 1.6 and x[1] <= 0.5

# Create mesh CHANGE TO VARIABLES!!!
mesh = Rectangle(0.0, 0.0, 3.0, 1.0, 60, 20)
    
# Create sub domain markers, 0 for fluid, 1 for structure
sub_domains = MeshFunction("uint", mesh, mesh.topology().dim())
sub_domains.set_all(0)
structure = Structure()
structure.mark(sub_domains, 1)

# Extract sub meshes for fluid and structure
structure_mesh = SubMesh(mesh, sub_domains, 1)
fluid_mesh = SubMesh(mesh, sub_domains, 0)  


class FSIChannel(NavierStokesProblem):
    
     

#      # Extract matching indices for fluid and structure
#      self.structure_to_fluid = compute_vertex_map(self.structure_mesh, self.fluid_mesh)
        
     def mesh(self):
          return fluid_mesh

     def structure_mesh(self):
          return self.structure_mesh()

     def viscosity(self):
          return 1.0 / 8.0

     def end_time(self):
          return 0.5

     def initial_conditions(self, V, Q):
          u0 = Constant(V.mesh(), (0, 0))
          p0 = Expression("1 - x[0]", V=Q)

          return u0, p0


     def boundary_conditions(self, V, Q):
          
          # Create no-slip boundary condition for velocity
          bcu = DirichletBC(V, Constant(V.mesh(), (0, 0)), noslip_boundary)

          # Create inflow and outflow boundary conditions for pressure
          bcp0 = DirichletBC(Q, Constant(Q.mesh(), 1), inflow_boundary)
          bcp1 = DirichletBC(Q, Constant(Q.mesh(), 0), outflow_boundary)

          return [bcu], [bcp0, bcp1]

    
     def info(self):
          return "Pressure driven channel (2D) with an obstructure"

     def time_step(self):
          return 0.001


# Solve problem
fsichannel = FSIChannel()
fsichannel.solve()
