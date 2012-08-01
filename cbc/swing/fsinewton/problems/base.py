"""Base classes for all FSI problems to be solved with Newtons Method"""
__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import fsinewton.utils.interiorboundary as intb
from cbc.swing.fsiproblem import FSI
from cbc.swing.parameters import default_parameters

#Interior Boundary
FSI_BOUND = 2
#Exterior Boundaries
STRUCBOUND = 1
DONOTHINGBOUND = 2
FLUIDNEUMANNBOUND = 3

class FsiNewton(FSI):
    """The FSI_Newton problem used when integrated with CBC"""
    def __init__(self,mesh, parameters=default_parameters()):
        self.parameters = parameters
        super(FsiNewton,self).__init__(mesh)
        
    def solve(self):
        # Solve and return computed solution (U_F, P_F, U_S, P_S, U_M, P_M)
        # Create submeshes and mappings (only first time)
        if self.Omega is None:
            # Refine original mesh
            mesh = self._original_mesh
            for i in range(self.parameters["num_initial_refinements"]):
                mesh = refine(mesh)
            # Initialize meshes
            self.init_meshes(mesh, self.parameters)
        # Create solver
        fsisolver = FSINewtonSolver(self)
        return fsisolver.solve()
    
class FsiNewtonTest(FSI):
    """This base class is a test class used to get the FSI newtons method
       To work before integration with CBC solve"""
    def __init__(self,mesh,strucdomain):
        FSI.__init__(self,mesh)
        self.mesh = mesh
        self.strucdomain = strucdomain

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
        #List attaching the measures to a space#
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
