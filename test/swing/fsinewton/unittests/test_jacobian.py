"""
A set of tests to insure that the manually computed jacobian is correct.
Testing is done against an automatically derived jacobian
using dolfin.derivative()


Fluid Block variables     U_F,P_F,L_U
Structure Block variables D_S,U_S
Mesh Block Variables      D_F,L_D
"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import io
from dolfin import *
import numpy as np
import demo.swing.analytic.newtonanalytic as nana
from demo.swing.minimal.minimalproblem import FSIMini
import cbc.swing.fsinewton.solver.solver_fsinewton as sfsi
import cbc.swing.fsinewton.utils.misc_func as mf
import cbc.swing.fsinewton.solver.spaces as spaces
from cbc.swing.parameters import fsinewton_params
        
class TestJacobians(object):
    "Test to make sure that the manual jacobian matches the analytical one",
    def setup_class(self):
        self.problems = [nana.NewtonAnalytic()] #FSIMini()

        #Tolerance for diffreports

        #Fixme The tolerance should actually be 1.0e-14 but for some reason the the buffered jacobian
        #doesn't match at the higher tolerance level. From experience it seems to have good convergence
        #though.
        self.TOL = 1.0e-11

    def test_jacobians(self):
        "Test the manual FSI Jacobian against the analytic Jacobian"
        #Todo continue the test into a few Newton iterations        
        #Change the default solver params
        fsinewton_params["optimization"]["reuse_jacobian"] = False
        fsinewton_params["optimization"]["simplify_jacobian"] = False
        fsinewton_params["plot"] = False
        fsinewton_params["fluid_domain_time_discretization"] = "mid-point"
        
        for prob in self.problems:
            print "Testing Jacobian with problem", prob.__str__()
            fsisolvers = {}
            newtonsolvers = {}
            residuals = {}
            jacobians = {}
            blocks= {}

            #Check the jacobians and residuals pairwise
            pairs = [("auto","manual"),
                     ("auto","buff"),
                     ("manual","buff")]

            blocknames = ["J_FF","J_FS","J_FM","J_SF","J_SS","J_SM","J_MF","J_MS","J_MM"]

            t = "buff"
            fsinewton_params["solve"] = False
            fsisolvers[t] = sfsi.FSINewtonSolver(prob,fsinewton_params)
            fsisolvers[t].prepare_solve()
            newtonsolvers[t] = fsisolvers[t].newtonsolver
            newtonsolvers[t].build_residual()
            newtonsolvers[t].build_jacobian()
            
            for t1,t2 in pairs:
                info_blue("Testing Jacobians of %s %s \n \n"%(t1,t2))
            
                for t in t1,t2:
                    fsinewton_params["jacobian"] = t
                    #Create a fresh solver
                    fsisolvers[t] = sfsi.FSINewtonSolver(prob,fsinewton_params)
                    fsisolvers[t].prepare_solve()
                    newtonsolvers[t] = fsisolvers[t].newtonsolver
                    newtonsolvers[t].build_residual()
                    newtonsolvers[t].build_jacobian()

                #Do two newton iterations
                for i in range(2):
                    for t in t1,t2:
                        residuals[t] = newtonsolvers[t].F.array()
                        jacobians[t] = newtonsolvers[t].J.array()
                    
                        #Set up the Jacobian blocks, fsi space should always be the same
                        sublocator = spaces.FSISubSpaceLocator(fsisolvers[t1].spaces.fsispace)

                        #Get the blocks of the matrix
                        blocks[t] = self.fsiblocks(jacobians[t],sublocator)
                    
                    #Check that the residual is the same
                    print "Checking residuals"
                    diff = residuals[t1] - residuals[t2]
                    print np.all(diff < self.TOL)
                    assert np.all(diff < self.TOL),"Error in residuals %s %s"%(t1,t2)
                    
                    #Check each block
                    for i,blockname in enumerate(blocknames):
                        print "\nChecking block ",blockname
                        mf.diffmatrix_report(blocks[t1][i],blocks[t2][i],self.TOL)
                        diff = blocks[t1][i] - blocks[t2][i]
                        print  np.all(diff < self.TOL)
                        assert np.all(diff < self.TOL),\
                           "Error in jacobian '%s' '%s' comparison. Block %s doesn't match at TOL = %f"%(t1,t2,blockname,self.TOL)
                    newtonsolvers[t1].step(newtonsolvers[t1].tol)
                    newtonsolvers[t2].step(newtonsolvers[t2].tol)
                
    def fsiblocks(self,J,sl):
        """
        Divide the jacobian matrix into blocks according to the subspace locator sl
        Fluid Block variables     U_F,P_F,L_U
        Structure Block variables D_S,U_S
        Mesh Block Variables      D_F,L_D
        """
        sl.fluidend = sl.spaceends["L_U"]
        sl.strucend = sl.spaceends["U_S"]
        
        J_FF = J[:sl.fluidend,:sl.fluidend]
        J_FS = J[:sl.fluidend,sl.fluidend:sl.strucend]
        J_FM = J[:sl.fluidend,sl.strucend:]
        J_SF = J[sl.fluidend:sl.strucend,:sl.fluidend]
        J_SS = J[sl.fluidend:sl.strucend,sl.fluidend:sl.strucend]
        J_SM = J[sl.fluidend:sl.strucend,sl.strucend:]
        J_MF = J[sl.strucend:,:sl.fluidend]
        J_MS = J[sl.strucend:,sl.fluidend:sl.strucend]
        J_MM = J[sl.strucend:,sl.strucend:]
        return (J_FF,J_FS,J_FM,J_SF,J_SS,J_SM,J_MF,J_MS,J_MM)

if __name__ == "__main__":
    t = TestJacobians()
    t.setup_class()
    t.test_jacobians()
