"""
A step by step Newton Solver used for testing purposes
"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import numpy as np
import cbc.swing.fsinewton.solver.mynewtonsolver as mns

class MyNewtonSolverSBS(mns.MyNewtonSolver):
    """Step by step Newton solver used for testing"""
    def __init__(self,problem):
        #super(MyNewtonSolverSBS,self)
        mns.MyNewtonSolver.__init__(self,problem,itrmax = 6)
    
    def step1_assemble(self):
        self.build_residual()
        self.build_jacobian()
        return self.F,self.J
    
    def step2_ident(self):
        self.J.ident_zeros()
        return self.J

    def step3_applybc(self):
        if self.problem.bc != None:
            [bc.apply(self.J) for bc in self.problem.bc]
            [bc.apply(self.F) for bc in self.problem.bc]
        return self.F,self.J

    def step4_solve_system(self):
        #Solve
        solve(self.J,self.inc.vector(),-self.F,"lu")
        return self.inc

    def step4a_numpy_solve(self):
        self.A = self.J.array()
        self.b = self.F.array()
        numpyinc = np.linalg.solve(self.A,self.b)
        #Important! Clear the inc vector
        self.inc.vector().zero()
        self.inc.vector()[:] += numpyinc[:]
        return numpyinc
    
    def step5_add_increment(self):
        self.problem.w.vector()[:] += self.inc.vector()
        return self.problem.w
        
    def step6_check_convergence(self):
        return np.linalg.norm(self.inc.vector().array(),ord = 2)

