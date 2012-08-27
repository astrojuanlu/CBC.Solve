"""My own Newton Solver, used in solving Nonlinear PDE problems"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import numpy as np
import cbc.swing.fsinewton.utils.misc_func as mf
import cbc.swing.fsinewton.solver.spaces as sp
from cbc.swing.fsinewton.utils.matrixdoctor import MatrixDoctor
from cbc.swing.fsinewton.solver.spaces import FSISubSpaceLocator
from cbc.swing.fsinewton.utils.timings import timings
from cbc.swing.fsinewton.utils.newtonsolveruntimedata import MyNewtonSolverRunTimeData
import copy

class MyNonlinearProblem:
    def __init__(self,f,w,bc,j, J_buff = None, cell_domains = None,
                 interior_facet_domains = None, exterior_facet_domains = None,
                 spaces = None):
        """ F = Nonlinear form 
            w = Initial guess function THIS SHOULD MATCH DIRICHLET BC exactly
            bc = list of boundary conditions
            j = Jacobian forms of F
            J_buff = A buffered part of the jacobian matrix. 
        """
        info("Creating Nonlinear Problem")
        self.f = f
        self.w = w
        self.bc = bc
        self.j = j
        self.J_buff = J_buff
        self.cell_domains = cell_domains
        self.interior_facet_domains = interior_facet_domains
        self.exterior_facet_domains = exterior_facet_domains
        self.spaces = spaces

class MyNewtonSolver:
    """General purpose Python Newton Solver"""
    def __init__(self,problem, tol = 1.0e-13, itrmax = 30,reuse_jacobian = False,
                 max_reuse_jacobian = 5, runtimedata = "False",reduce_quadrature = 0):
        self.tol = tol
        self.itrmax = itrmax
        self.itr = 0
        self.E = self.tol + 1
        self.jacobian_itr = 0
        self.res = [] 
        self.problem = problem
        self.fsispace = self.problem.w.function_space()
        self.inc = Function(self.fsispace)
        self.reuse_jacobian = reuse_jacobian
        self.max_reuse_jacobian = max_reuse_jacobian
        self.runtimedata = runtimedata
        (self.F,self.J) = (None,None)
        if self.problem.bc != None:
            [bc.homogenize() for bc in self.problem.bc]
        self.ffc_opt = {"representation": "quadrature"}
        if reduce_quadrature != 0:
            self.ffc_opt["quadrature_degree"] = reduce_quadrature
                
    def plot_current(self):
        plot = Function(self.problem.w.function_space())
        
    def solve(self,method = "lu",inc_plot = False, runtimedata = False, t = 0.0):
        """
        runtimedata
            - object of type NewtonSolverRunTimeData
        """
        
        #Do the whole solve and return result
        if self.runtimedata:
            self.subloc = sp.FSISubSpaceLocator(self.fsispace)
            self.runtimedata = MyNewtonSolverRunTimeData(t)
        self.itr = 0
        self.E = self.tol + 100
        self.jacobian_itr = 0
        self.res = [] 
        
        while self.E > self.tol:
            self.step(self.tol,method = method, inc_plot = inc_plot)
        return self.res

    def step(self,tol,method = "lu",inc_plot = False):
        """Do one Newton iteration"""
        info("Executing Newton Iteration")
        
        #Build Residual 
        self.build_residual()            
        
        #Check for convergence, F(u) should be close to 0.
        #Get the discrete 2 norm of the increment
##        self.E = np.max(self.F.array())
        self.lastresidual = self.E
        self.E = np.linalg.norm(self.F.array(),ord = 2)
        info(str((self.itr,self.E)))

        if self.runtimedata != "False":
            #get the L2 norm and max norm for each residual function
            for s in self.subloc.spaces.keys():
                vec = self.F.array()[self.subloc.spacebegins[s]:self.subloc.spaceends[s]]
                el2 = np.linalg.norm(vec,ord = 2)
                emax = np.max(vec)
                self.runtimedata.residuals["l2"][s].append(el2)
                self.runtimedata.residuals["max"][s].append(emax)
        if self.E < tol:
            return

        #Rebuild jacobian if neccessary (for example if a blow up starts)
        if self.reuse_jacobian == False or \
        self.jacobian_itr == self.max_reuse_jacobian or\
        self.E > self.lastresidual:
            self.build_jacobian()
            self.jacobian_itr = 0

        #Check to see if the norm is nan, in that case assume 
        #the matrix is singular.
        if np.isnan(self.E):
            raise NanError(self.inc,self.J,self.F)  
        
        if inc_plot == True:
            for i in range(2):
                plot(self.inc.split()[i], title = "Increment %i"%i,mode = "displacement")
            interactive()

        #Linear Solve
        self.linear_solve()

        self.problem.w.vector()[:] += self.inc.vector()
        self.res += [(self.itr,self.E)]
        self.jacobian_itr += 1
        self.itr +=1

        if self.itr > self.itrmax:
            R = Function(self.inc.function_space())
            R.vector()[:] = self.F
            raise NewtonConverganceError(self.itrmax, R)

    def linear_solve(self):
        """Dolfin/PETSc linear solve"""
        timings.startnext("PETSc linear solve")
        info("PETSc Linear Solve")
        self.linsolver.solve(self.inc.vector(),-self.F)        
        timings.stop("PETSc linear solve")         
##        #Benchmark
##        import fsinewton.utils.solver_benchmark as bench
##        bench.solve(self.J,self.inc.vector(),-self.F,benchmark = True)
##        exit()           
        
    def apply_ident_bc(self):
        """Use ident_zeros and apply BC"""
        #Deactivate dead DOFS
        self.J.ident_zeros()
        #Apply BC
        if self.problem.bc != None:
            [bc.apply(self.J) for bc in self.problem.bc]
    
    def build_jacobian(self):
        """Assemble Jacobian"""
        info("Assembling Jacobian")
        
        #If buffered matrix add the variable part to the buffered part.
        if self.problem.J_buff is not None:
            timings.startnext("Copy Buffered Jacobian")
            self.J = self.problem.J_buff.copy()

            timings.startnext("Jacobian Assembly")
            self.J = assemble(self.problem.j, tensor = self.J,
                              cell_domains = self.problem.cell_domains,
                              interior_facet_domains = self.problem.interior_facet_domains,
                              exterior_facet_domains = self.problem.exterior_facet_domains,
                              reset_sparsity = False,
                              add_values = True,
                              form_compiler_parameters = self.ffc_opt)
            timings.stop("Jacobian Assembly")
        else:
            #No buffering just assemble
            timings.startnext("Jacobian Assembly")
            self.J = assemble(self.problem.j, tensor = self.J,
                              cell_domains = self.problem.cell_domains,
                              interior_facet_domains = self.problem.interior_facet_domains,
                              exterior_facet_domains = self.problem.exterior_facet_domains,
                              form_compiler_parameters = self.ffc_opt)
            timings.stop("Jacobian Assembly")
        #Give the Jacobian it's BC.
        self.apply_ident_bc()
        #Create an LU Solver
        # Create linear solver and factorize matrix
        self.linsolver = LUSolver(self.J)
        self.linsolver.parameters["reuse_factorization"] = True
        
    def build_residual(self):
        """Assemble Residual"""
        timings.startnext("Residual assembly")
        info("Residual Assembly")
        self.F = assemble(self.problem.f, tensor = self.F,
             cell_domains = self.problem.cell_domains,
             interior_facet_domains = self.problem.interior_facet_domains,
             exterior_facet_domains = self.problem.exterior_facet_domains)
        [bc.apply(self.F) for bc in self.problem.bc]
        timings.stop("Residual assembly")
        
##        dofs = self.problem.spaces.restricteddofs["U_S"] + self.problem.spaces.restricteddofs["P_S"]
##        print self.F.array()[dofs]
####        print self.F.array()
##        exit()

class MyNewtonSolverNumpy(MyNewtonSolver):
    """Newton Solver using numpy linear algebra and fsi space restriction"""
    def build_jacobian(self):
        #Build the normal jacobian
        MyNewtonSolver.build_jacobian(self)
        #Now Turn it into a numpy array and remove excess dofs
        
        self.J_np = self.J.array()
        dofs = self.problem.spaces.usefuldofs
            
##        print "Array size before dof removal",len(self.J_np)
##        print "Condition number before dof removal",np.linalg.cond(self.J_np)
##        print "Norm before dof removal",np.linalg.norm(self.J_np,2)
        
        #Remove the uneccessary parts of the matrix.
        self.J_np = self.J_np[dofs].transpose()[dofs].transpose()
        
##        print "Array size after dof removal",len(self.J_np)
##        print "Condition number after dof removal",np.linalg.cond(self.J_np)
##        print "Norm after dof removal",np.linalg.norm(self.J_np,2)

    def linear_solve(self):
        """
        Solve Ax = b with numpy and also with removal of uneccesary rows
        using the spaces object
        """
        timings.startnext("Numpy linear solve")
        info("Numpy Linear Solve")
        dofs = self.problem.spaces.usefuldofs

        #Original np array
        b = -self.F.array()
        #New np array
        b_np = b[dofs]
        #Solve the system
        x_np = np.linalg.solve(self.J_np,b_np)

        #map the solution back to the bigger space
        x = np.zeros(len(b))
        x[dofs] = x_np

##        #Create a Vector
        x_vec = Vector(len(x))
        x_vec[:] = x
##        
        self.inc.vector()[:] = x
        timings.stop("Numpy linear solve")

class NanError(Exception):
    """Exception class for an NAN result in a Newton Solver"""
    def __init__(self,inc,J,F):
        #increment function
        self.inc = inc
        #Jacobian
        self.J = J.array()
        #residual
        self.F = F.array()
        self.mess = "Nan in newton solver increment vector"
        #Default zero tolerance for the matrix doctor
        self.zTOL = 1.0e-5
        
    def __mess__(self):
        return repr(self.mess)
    
    def analysis(self):
        sl = FSISubSpaceLocator(self.inc.function_space())
        fluidend = sl.spaceends["M_U"]
        strucend = sl.spaceends["V_S"]
        
        info("\n Jacobian Determinant \n" + \
              str(np.linalg.det(self.J)) + \
              "Block determinants \n" + \
               "\nFF " + str(np.linalg.det(self.J[:fluidend,:fluidend])) + \
               "\nSS " + str(np.linalg.det(self.J[fluidend:strucend,fluidend:strucend])) + \
               "\nMM " + str(np.linalg.det(self.J[strucend:,strucend:])) + \
               "\n Norm of residual vector \n" + \
               str(np.linalg.norm(self.F,ord = 2))
                )
    
class NewtonConverganceError(Exception):
    """Exception class for a failure to converge in a newton solver"""
    def __init__(self, iterations,inc):
        self.iterations = iterations
        self.mess = "Newton Solver failed to converge in " + str(iterations) + "iterations"
        self.inc = inc #The last computed Newton increment
        self.cTOL = 1.0e-14
        
    def __mess__(self):
        return repr(self.mess)
    
    def stuck_dofs(self,cTOL = 1.0e-8):
        """Returns all DOFS that were "Stuck" in the solve and have value > cTOL"""
        dofs = {}
        for index,elem in enumerate(self.inc.vector()):
            if abs(elem) > cTOL:
                dofs[index] = elem
        return dofs
    
    def stuck_spaces(self,dofs,sublocator):
        """returns a set of subspaces where a stuck dof is present"""
        stuckspaces = set([])
        for dof in dofs:
            stuckspaces |= set([sublocator.subspace(dof)])
        return stuckspaces

    def plot_inc(self,sub, mode = None):
        mf.plot_single(self.inc,sub,"Last Increment of Newton Solver, subspace " + str(sub),mode = mode ,interact = False)

    def plot_stuck_spaces(self,stuckspaces):
        for i in stuckspaces:
            self.plot_inc(i,mode = "displacement")
        
    def report(self):
        """Report on why Newton's Method did not converge"""
        stuck_dofs = self.stuck_dofs(cTOL = self.cTOL)
        sublocator = sp.SubSpaceLocator(self.inc.function_space())
        doflist = ""                
        for dof in stuck_dofs.keys():
            doflist += "%i %i %f \n"%(dof, sublocator.subspace(dof),stuck_dofs[dof])
            
        info_blue("Convergence Failure Report %s\nNonzero increment DOfs cTOL = %f\n \
                  DOF, Space, Value\n"%(sublocator.report(),self.cTOL) + doflist)
            
        #Plot all increments of spaces where the DOFs are stuck
        self.plot_stuck_spaces(self.stuck_spaces(stuck_dofs,sublocator))
        interactive()
    
