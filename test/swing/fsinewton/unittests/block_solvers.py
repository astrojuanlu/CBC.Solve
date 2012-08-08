"""
A set of block solvers of the full FSI Solver
"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import numpy as np
import fsinewton.problems.minimal_problem as pm
import fsinewton.solver.solver_fsinewton as sfsi
import fsinewton.solver.spaces as space
import fsinewton.utils.misc_func as mf
import fsinewton.solver.mynewtonsolver as mns
from fsinewton.solver.default_params import solver_params
np.set_printoptions(precision = 2, edgeitems = np.inf)

class BlockTester(object):
    """Quick and dirty way to add information to a solver for a single time step"""
    def __init__(self):
        self.sublocator = space.FSISubSpaceLocator(self.spaces.fsispace)
    
    def input_info(self,U,start,end):
        """Input information"""
        try:
            V = U.function_space()
        except:
            V = U.function_space().collapse()
        zero = Function(V)
        #Assume U is a function over FSI space
        try:
            #Delete the old dof's
            self.U0.vector()[start:end] = zero.vector()[start:end]
            #Add the new value
            self.U0.vector()[start:end] += U.vector()[start:end]
        #If not it must be in the space we are trying to input to
        except:
            #Delete the old dof's
            self.U0.vector()[start:end] = zero.vector()
            #Add the new value
            self.U0.vector()[start:end] += U.vector()

############################################################
#Single Block Solvers
############################################################
class FluidBlockSolver(sfsi.FSINewtonSolver,BlockTester):
    """A Fluid Solver, all fsi spaces included"""
    def __init__(self,problem,solver_params = solver_params):
        super(FluidBlockSolver,self).__init__(problem,params = solver_params)
        BlockTester.__init__(self)

        #Use just the fluid forms
        self.r = self.blockresiduals["r_F"]
##        self.j = self.rowjacobians["j_F"]
        

class AutoDerivativeFluidBlockSolver(FluidBlockSolver):
    """Fluid Block Solver with automated derivative"""
    def __init__(self,problem,solver_params = solver_params):
        FluidBlockSolver.__init__(self,problem,solver_params)
        print "Using automated derivative"
        self.j = derivative(self.r,self.U1)

class StrucBlockSolver(sfsi.FSINewtonSolver,BlockTester):
    """A structure solver, all fsi spaces included"""
    def __init__(self,problem,params = solver_params):
        super(StrucBlockSolver,self).__init__(problem,params = solver_params)
        BlockTester.__init__(self)

        #Use just the structure forms
        self.r = self.blockresiduals["r_S"]
        self.j = self.rowjacobians["j_S"]

class AutoDerivativeStrucBlockSolver(StrucBlockSolver):
    """Structure block solver with automated derivative"""
    def __init__(self,problem,solver_params = None):
        StrucBlockSolver.__init__(self,problem,solver_params)
        print "Using automated derivative"
        self.j = derivative(self.r,self.U1)

class MeshBlockSolver(sfsi.FSINewtonSolver,BlockTester):
    """A Mesh solver, all fsi spaces included"""
    def __init__(self,problem):
        super(MeshBlockSolver,self).__init__(problem)
        BlockTester.__init__(self)
        
        #Use just the Mesh forms
        self.r = self.blockresiduals["r_M"]
        self.j = self.rowjacobians["j_M"]

class AutoDerivativeMeshBlockSolver(MeshBlockSolver):
    """Mesh block solver with automated derivative"""
    def __init__(self,problem):
        MeshBlockSolver.__init__(self,problem)
        print "Using automated derivative"
        self.j = derivative(self.r,self.U1)

############################################################
#One way Double Block Solvers
############################################################
class FluidStrucBlockSolver(sfsi.FSINewtonSolver,BlockTester):
    """Fluid Structure solver. The coupling should be two-way
        as the velocity and traction should be equal on the
        FSI boundary. The change in geometry due to the structure
        moveing is ignored."""
    
    def __init__(self,problem):
        super(FluidStrucBlockSolver,self).__init__(problem)
        BlockTester.__init__(self)
                
        #Use just Fluid and the structure forms
        self.r = self.blockresiduals["r_F"] + self.blockresiduals["r_S"]
        self.j = self.rowjacobians["j_F"] + self.rowjacobians["j_S"] 
        
class AutoDerivativeFluidStrucBlockSolver(FluidStrucBlockSolver):
    """Fluid Structure block solver with automated derivative"""
    def __init__(self,problem):
        FluidStrucBlockSolver.__init__(self,problem)
        print "Using automated derivative"
        self.j = derivative(self.r,self.U1)

class StrucMeshBlockSolver(sfsi.FSINewtonSolver,BlockTester):
    """Structure Mesh solver. The coupling should be struc->mesh
        as the struc displacement on the FSI boundary should
        be matched in the mesh"""
    
    def __init__(self,problem,BlockTester):
        super(StrucMeshBlockSolver,self).__init__(problem)
        BlockTester.__init__(self)
        #Restrict BC to the structure and mesh domain
        self.bcall = self.create_structure_bc()
        self.bcall += self.create_mesh_bc()
                
        #Use just Mesh and the structure forms
        self.r = self.blockresiduals["r_S"] + self.blockresiduals["r_M"]
        self.j = self.rowjacobians["j_S"] + self.rowjacobians["j_M"] 

class AutoDerivativeStrucMeshBlockSolver(StrucMeshBlockSolver):
    """Structure Mesh block solver with automated derivative"""
    def __init__(self,problem):
        StrucMeshBlockSolver.__init__(self,problem)
        print "Using automated derivative"
        self.j = derivative(self.r,self.U1)

class MeshFluidBlockSolver(sfsi.FSINewtonSolver,BlockTester):
    """Mesh Fluid solver. The coupling should be mesh->fluid. Unless some
        structure information is given I assume the result should
        be the same as using a pure fluid solver"""
    def __init__(self,problem,BlockTester):
        super(MeshFluidBlockSolver,self).__init__(problem)
        BlockTester.__init__(self)
        #Restrict BC to the mesh and fluid domain
        self.bcall = self.create_mesh_bc()
        self.bcall += self.create_fluid_bc()
                
        #Use just Mesh and the fluid forms
        self.r = self.blockresiduals["r_F"] + self.blockresiduals["r_M"]
        self.j = self.rowjacobians["j_F"] + self.rowjacobians["j_M"] 
        
class AutoDerivativeMeshFluidBlockSolver(MeshFluidBlockSolver):
    """Structure Mesh block solver with automated derivative"""
    def __init__(self,problem):
        MeshFluidBlockSolver.__init__(self,problem)
        print "Using automated derivative"
        self.j = derivative(self.r,self.U1)

if __name__ == "__main__":
    problem = pm.FSIMini()
    solver = FluidStrucBlockSolver(problem)
    solver.solve(single_step = True)
    mf.plot_single(solver.U0,3,"Struc displacement", mode = "displacement")
    interactive()
    
