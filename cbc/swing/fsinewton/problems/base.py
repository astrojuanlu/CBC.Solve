"""Base classes for all FSI problems to be solved with Newtons Method"""
__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import fsinewton.utils.interiorboundary as intb

#Interior Boundary
FSI_BOUND = 2
#Exterior Boundaries
STRUCBOUND = 1
DONOTHINGBOUND = 2
FLUIDNEUMANNBOUND = 3
  
class NewtonFSI():
    """Basic problem class for Newton's method FSI"""
    def __init__(self,mesh):
        self.mesh = mesh
        self.strucdomain = self.structure()

        #Mesh Function
        self.cellfunc = MeshFunction("uint", mesh, mesh.topology().dim())
        self.cellfunc.set_all(0)
        strucdomain.mark(self.cellfunc,1)

        #Generate exterior boundaries 
        class StructureBound(SubDomain):
            def inside(self,x, on_boundary):
                return on_boundary and strucdomain.inside(x,on_boundary)
        
        #Generate fluid Domain
        class FluidDomain(SubDomain):
            def inside(self,x,on_boundary):
                return not strucdomain.inside(x,on_boundary)
        self.fluiddomain = FluidDomain()

        #Generate External boundaries
        self.extboundfunc = FacetFunction("uint",mesh)
        self.extboundfunc.set_all(0)
        StructureBound().mark(self.extboundfunc,STRUCBOUND)
        
        if self.fluid_donothing_boundaries() is not None:
            self.fluid_donothing_boundaries().mark(self.extboundfunc,DONOTHINGBOUND)
            self.dsDN = ds(DONOTHINGBOUND)
        else:
            self.dsDN = None
            
        if self.fluid_velocity_neumann_boundaries() is not None:
            self.fluid_velocity_neumann_boundaries().mark(self.extboundfunc,FLUIDNEUMANNBOUND)
            self.dsF = ds(FLUIDNEUMANNBOUND)
        else:
            self.dsF = None
            
        #Generate submeshs for the structure and fluid
        self.strucmesh = SubMesh(mesh,self.cellfunc,1)
        self.fluidmesh = SubMesh(mesh,self.cellfunc,0)

##        plot(mesh,title = "Whole mesh")
##        plot(self.strucmesh,title = "Structure mesh")
##        plot(self.fluidmesh,title = "Fluid mesh")
##        interactive()        
        
        #Generate interior boundary
        self.fsibound = intb.InteriorBoundary(mesh)
        self.fsibound.create_boundary(self.strucmesh)
        self.fsiboundfunc = self.fsibound.boundaries[0]

        #Generate Nonfluidboundary measures
        self.dxS = dx(1)          #Structure
        self.dxF = dx(0)          #Fluid
        self.dFSI = dS(FSI_BOUND) #FSI Boundary
        self.dsS = ds(0)          #Structure outer boundary
        self.dxlist = [self.dxF,self.dxF,self.dFSI,self.dxS,self.dxS,self.dxF,self.dFSI]

    #Defualt parameters
    def fluid_density(self):
        return 1.0
    def fluid_viscosity(self):
        return 1.0
    def structure_density(self):
        return 1.0
    def structure_mu(self):
        return 1.0
    def structure_lmbda(self):
        return 1.0
    def mesh_mu(self):
        return 1.0
    def mesh_lmbda(self):
        return 1.0

    #Initial Conditions
    def fluid_velocity_initial_condition(self):
        pass
    def fluid_pressure_initial_condition(self):
        pass
    def struc_displacement_initial_condition(self):
        pass
    def struc_velocity_initial_condition(self):
        pass
    def mesh_displacement_initial_condition(self):
        pass

    #Boundary conditions
    def fluid_velocity_dirichlet_values(self):
        pass
    def fluid_velocity_dirichlet_boundaries(self):
        pass
    def fluid_velocity_neumann_boundaries(self):
        pass
    def fluid_velocity_neumann_values(self):
        pass
    def fluid_donothing_boundaries(self):
        pass
    def fluid_pressure_dirichlet_values(self):
        pass
    def fluid_pressure_dirichlet_boundaries(self):
        pass
    def structure_dirichlet_values(self):
        pass
    def structure_dirichlet_boundaries(self):
        pass
    def structure_velocity_dirichlet_boundaries(self):
        pass
    def structure_velocity_dirichlet_values(self):
        pass
    def mesh_dirichlet_boundaries(self):
        pass
    def mesh_dirichlet_values(self):
        pass
    def fluid_lm_dirichlet_boundaries(self):
        pass
    def mesh_lm_dirichlet_boundaries(self):
        pass
    def fluid_lm_dirichlet_values(self):
        pass
    def mesh_lm_dirichlet_values(self):
        pass
    
    #this can be used to prescribe a fluid stress on the structure
    def fluid_fsi_stress(self):
        pass

    #Body Forces
    def fluid_body_force(self):
        pass
    def structure_body_force(self):
        pass
    def mesh_right_hand_side(self):
        pass

    def structure_boundary_traction_extra(self):
        pass

    def fluid_boundary_traction(self):
        pass
