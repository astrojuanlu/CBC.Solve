"""
A set of tests to insure that the fluid block of the FSINewtonSolver is working properly
"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import io
from dolfin import *
import numpy as np
import demo.swing.minimal_problem as pm
import cbc.swing.fsinewton.solver.solver_fsinewton as sfsi
import cbc.swing.fsinewton.utils.misc_func as mf
import cbc.swing.fsinewton.solver.mynewtonsolver as mns
import block_solvers as bs
import fsinewton.solver.spaces as space
np.set_printoptions(precision = 2, edgeitems = np.inf)

class FluidAnalytical(object):   
    """Test the x value of the fluid solution at the midpoint of the right edge """
    def diff(self,u,t):
        return self.functional(u) - self.reference(t) 
      
    def functional(self,u):
        return u((pm.meshlength, pm.fluidheight*0.5))[0]

    def reference(self,t):
        num_terms = 30
        u = 1.0
        c = 1.0
        for n in range(1, 2*num_terms, 2):
            a = 32.0 / (DOLFIN_PI**3*n**3)
            b = (1.0/8.0)*DOLFIN_PI**2*n**2
            c = -c
            u += a*exp(-b*t)*c
        return u

class FluidSolverSingle(sfsi.FSINewtonSolver):
    """Fluid Solver restricted to Fluid Spaces"""
    def __init__(self,problem,solver_params):
        
        sfsi.FSINewtonSolver.__init__(self,problem,params = solver_params)
        #Make the full solution space just the fluid space
        self.fsispace = MixedFunctionSpace((self.spaces.V_F,self.spaces.Q_F,self.spaces.L_F))
        
        #Redefine subspaces
        [self.V_F,self.Q_F,self.L_F] = [self.spaces.fsispace.sub(i) for i in range(3)]

        #Extract the pressure from self.U0 (belongs to the FSINewtonSolver)
        subspacenew = space.SubSpaceLocator(self.fsispace)
        subspaceold = space.SubSpaceLocator(self.U0.function_space())
        U0new = Function(self.fsispace)
        U0new.vector()[:] = self.U0.vector()[0:subspaceold.final_dofs[2]]
        self.U0 = U0new

        #Functions
        (v,q,m) = TestFunctions(self.fsispace)
        Iu,Ip,Il = TrialFunctions(self.fsispace)
        self.U1 = Function(self.fsispace)
        (u1,p1,l1) = (as_vector((self.U1[0], self.U1[1])), \
                      self.U1[2],as_vector((self.U1[3], self.U1[4])))
        
        (u0,p0,l0) = (as_vector((self.U0[0], self.U0[1])), self.U0[2], \
                      as_vector((self.U0[3], self.U0[4])))
        self.u1 = u1
        self.p1 = p1

        #Dummy function
        dum = Function(self.V_F.collapse())
        
        #New function lists with dum in non fluid space
        self.U0list = [u0,p0,l0,dum,dum,dum,dum]
        self.U1list = [u1,p1,l1,dum,dum,dum,dum]
        self.V = [v,q,m,dum,dum,dum,dum]
        self.IU = [Iu,Ip,Il,dum,dum,dum,dum]
        self.Umid,self.Udot = self.time_discreteU(self.U1list,self.U0list,self.kn)
        self.IUmid,self.IUdot = self.time_discreteI(self.IU,self.kn)
        #Create all forms
        self.create_forms()
        
        #Use just the fluid forms 
        self.r = self.blockresiduals["r_F"]
        #Note that these two derivatives should agree!
        #self.j = derivative(self.r_F,self.U1)
        self.j = self.blockjacobians["j_FF"]

        #Restrict BC to the fluid domain
        self.bcall = self.fsibc.create_fluid_bc()
        self.plotu = Function(self.spaces.V_FC)

#Testing class for pytest    
class TestFluid():
    """Class to test the fluid block of the monolithic fsi problem"""
    def setup_class(self):
        self.TOL = 1.0e-3
        self.problem = pm.FSIMini()
        #Check that the viscocity hasn't been tampered with.
        assert self.problem.fluid_viscosity() == 1.0/8.0,"Someone has tampered with the fluid viscocity, \
                                                        for the minimal problem, it should be = "+ str(1.0/8.0)
        #Set a new end time
        self.problem.end_time = lambda :0.5

        #Initialize solvers

        from fsinewton.solver.default_params import solver_params
        solver_params["jacobian"] = "manual"
        solver_params["solve"] = False
        self.solvers =[bs.FluidBlockSolver(self.problem,solver_params = solver_params)]
##        self.solvers = [bs.FluidBlockSolver(self.problem,solver_params = solver_params),
##                            ,FluidSolverSingle(self.problem,solver_params = solver_params)\
##                            ,bs.AutoDerivativeFluidBlockSolver(self.problem,solver_params = solver_params)] 
        #Initialize an analytical solution
        self.anasol = FluidAnalytical()
        
    def test_analytic_fluid(self):
        """Compare the fluid solution to a known analytical solution for various solvers"""
        for solver in self.solvers:
            #Solve 2 time steps and see if error stays below TOL
            print "Testing Solver",solver.__doc__

            #Initializise the solve objects
            solver.solve()
            solver.prebuild_jacobians()
            for i in range(2):
                solver.__time_step__()
                print "Error equals"
                E = self.anasol.diff(solver.U1.split()[0], solver.t)
                print E
                print
                assert E < self.TOL,"Error in fluid test solver " +solver.__doc__+ \
                       " computed solution - analytic greater than TOL," + str(self.TOL)

if __name__ == "__main__":
    t = TestFluid()
    t.setup_class()
    t.test_analytic_fluid()
