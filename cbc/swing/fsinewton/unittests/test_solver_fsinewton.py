""" Test individual methods of the fsi newton solver"""
__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import fsinewton.solver.solver_fsinewton as sfsi
import fsinewton.problems.minimal_problem as pm
import test_block_solvers as tbs
import fsinewton.utils.misc_func as mf
import fsinewton.problems.mesh_problem as mp

class TestFSINewtonSolver(mf.NotZeroTester):
    """Py test class for the class FSINewtonSolver"""

    def setup_class(self):
        self.meshproblem = mp.MeshProblem()
        self.fsimeshproblem = mp.FSIMeshProblem()
        self.miniproblem = pm.FSIMini()

    def test_initial_structure_displacement(self):
        """See if an initial struc displacement is generated correctly"""
        solver = sfsi.FSINewtonSolver(self.meshproblem)

        #There should be an initial struc displacement on the
        #FSI boundary
        self.checknotzero(mf.extract_subfunction(solver.U0_S),\
                          dx = self.meshproblem.dFSI,\
                          desc = "FSI Structure displacement", \
                          interior_facet_domains = self.meshproblem.fsiboundfunc,\
                          restrict = True)
        print "Initial structure displacement on FSI boundary test passed"
        
    def test_fsimeshproblem(self):
        """In the FSImeshproblem an initial structure displacement
        is perscribed which causes changes in the other variables"""
        problem = self.miniproblem
        
        solver = sfsi.AutoDerivativeFSINewtonSolver(problem)
        solver.solve(single_step = True)

        variables = [solver.U1_F,solver.U1_S,solver.U1_M]
        names = ["FSI Fluid displacement","FSI Structure displacement",\
                 "FSI Mesh displacement"]

        for var,name in zip(variables,names):
            #All variables should be nonzero        
            self.checknotzero(mf.extract_subfunction(var),desc = name)   
        print "FSIMeshProblem test passed"

if __name__ == "__main__":
    t = TestFSINewtonSolver()
    t.setup_class()
    t.test_initial_structure_displacement()
    t.test_fsimeshproblem()
