"""
A set of tests to insure that the submesh and boundary generation of
Interiorboundary and  FSIMini are done properly
"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import numpy as np
import demo.swing.minimal.minimalproblem as pm
import cbc.swing.fsinewton.solver.solver_fsinewton as sfn
import cbc.swing.fsinewton.utils.misc_func as mf

class TestMeshSetup(object):
    """A test class to see if the Mesh for FSI problem is set up properly"""
    def setup_class(self):
        """New problems to be quality controlled can be added here"""
        self.problems = [pm.FSIMini()]
        #The actual lengths (in edges) are read from the problem modules
        self.fsiboundfacets = [pm.nx]
        self.fsiboundlengths = [pm.meshlength]
        self.fluidspaces = [FunctionSpace(problem.fluidmesh,"CG",1) for problem in self.problems]
        self.strucspaces = [FunctionSpace(problem.strucmesh,"CG",1) for problem in self.problems]
        
    def test_FSIbounds(self):
        for i,problem in enumerate(self.problems):
            """Test to see if the length of the FSI boundary is correct"""
            vol = self.fsibound_volume(problem)
            voltrue = self.fsiboundlengths[i]
            assert near(vol,voltrue), "Error in FSI boundary creation, approximate volume %g does not equal 2"%(vol)
            #Test to see if the number of facets is correct
            cfound = problem.fsibound.countfacets
            cactual = self.fsiboundfacets[i]
            assert cfound == cactual, "Error in FSI boundary creation of "+problem.__str__()+ " the number of facets in the generated boundary \
                                      %d does not equal that of the strucmesh boundary %d"%(cfound,cactual)
            print " ".join(["Problem",problem.__str__(),"passed the length of FSI boundary test"])
        print
            
    def fsibound_volume(self,problem):
        """Gives the volume of the FSI boundary"""
        V = FunctionSpace(problem.singlemesh,"CG",1)
        one = interpolate(Constant(1),V)
        volform = one('-')*problem.dFSI
        return assemble(volform,interior_facet_domains = problem.fsiboundfunc)

    def test_static_boundaries(self):
        """ Test to see if the non FSI interface boundaries are set up correctly"""
        for i,problem in enumerate(self.problems):
            self.bchittest(problem.fluid_velocity_dirichlet_boundaries(),\
                                  "Fluid Mesh", self.fluidspaces[i],problem)
            
            self.bchittest(problem.fluid_pressure_dirichlet_boundaries(),\
                                  "Fluid Mesh", self.fluidspaces[i],problem)
            
            self.bchittest(problem.structure_dirichlet_boundaries(), \
                                  "Structure Mesh", self.strucspaces[i],problem)
            
            self.bchittest(problem.mesh_dirichlet_boundaries(), \
                           "Fluid Mesh", self.fluidspaces[i],problem)
            print " ".join(["Problem",problem.__str__(),"passed the static boundary tests"])
        print
                    
    def bchittest(self,boundaries,meshname,funcspace,problem):
        """Test to see if a BC hits a given mesh by examining its values and dofs"""
        if boundaries != None:
            for boundary in boundaries:
                bc = DirichletBC(funcspace,0.0,boundary)
                assert not bc.get_boundary_values() == {},"Error in " + problem.__str__() + " BC " + boundary + " BC does not effect " + meshname
                ##self.bchittest(bound,space,meshname, problem.__str__())
                
    def test_initial_conditions(self):
        """create the initial conditions and see if applying BC changes any values"""
        for problem in self.problems:
            solver = sfn.FSINewtonSolver(problem)
            #get the initial condition
            initcond = solver.U0
            initcopy = Function(solver.spaces.fsispace)
            initcopy.assign(initcond)
            for index,bc in enumerate(solver.fsibc.bcallU1_ini):
                mf.apply_to(bc,initcopy)
                #Test to see if the bc have changed anything
                for i in range(len(initcond.vector())):
                    if not initcond.vector()[i] == initcopy.vector()[i]:
                        plot(initcond[2], title = "Initial Condition")
                        mf.plot_bc(solver,bc,i)
                        assert 1== 0,"Error in BC " + str(index) + " disagreement with initial conditions on DOF " +str(i)
                        
            print " ".join(["Problem",problem.__str__(),"passed the initial conditions test"])
        print
                
if __name__ == "__main__":
    t = TestMeshSetup()
    t.setup_class()
    t.test_FSIbounds()
    t.test_static_boundaries()
    t.test_initial_conditions()

