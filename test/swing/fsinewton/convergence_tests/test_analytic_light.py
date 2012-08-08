"""
Test the solution from the FSI Newton Solver against a known analytic solution

This is the light version which does not store timeseries, but rather calculates
the time integrated L2 norm at each refinement level. This way less data is generated
and large tests can be carried out using bigblue. No plots can be made afterwards with
a different script.

This file should be run from the command line as a script with the following options
1.type of test
2. order of struc and mesh function spaces
3. type of boundary condition
4. begin convergence level
5. end convergence level
6. solve (True/False)

to get a log of the test use tee, for example
    python test_analytic.py fsi 0 3 | tee fsilog.py
"""


__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import io
import os
from dolfin import *

#Switch here to change analytic solutions
##import fsinewton.problems.analytic.newtonanalytic as nana
##import fsinewton.problems.analytic.right_hand_sides as rhs
##PROBLEMFOLDER = "convergence"

import fsinewton.problems.analytic_new.right_hand_sides_new as rhs
import fsinewton.problems.analytic_new.newtonanalytic_new as nana
PROBLEMFOLDER = "convergence_new"


import fsinewton.problems.analytic.analytic as ana
import fsinewton.solver.solver_fsinewton as sfsi
import fsinewton.utils.misc_func as mf
import matplotlib.pyplot as plt
from fsinewton.utils.output import FSIPlotter
from fsinewton.solver.default_params import solver_params
from fsinewton.utils.timings import timings
import cPickle

class AnalyticFSISolution(object):
    "Contains the fsi analytical solution data"
    def __init__(self):
        
        #Create Expressions
        self.EU_F = Expression(rhs.cpp_U_F)
        self.EP_F = Expression(rhs.cpp_P_F)
        self.EU_S = Expression(rhs.cpp_U_S)
        self.EU_M = Expression(rhs.cpp_U_M)

        #Set the constants
        self.EU_F.C = ana.C
        self.EP_F.C = ana.C
        self.EU_S.C = ana.C
        self.EU_M.C = ana.C
    
    def update(self,t):
        self.EU_F.t = t
        self.EP_F.t = t
        self.EU_S.t = t
        self.EU_M.t = t

class zTestAnalytic(object):
    """Analytic FSI tester"""
    def setup_class(self):
        #Create an analytical solution solver to compare to.
        self.anasol = AnalyticFSISolution()
        self.plotstyles = {"U_F":'bD',"P_F":'gp',"U_S":'r*',"P_S":'co',"U_M":'mv'}

    def convergence_test(self,test,elem_order,bctype,start_refine,end_refine,solve):
        "Generate plots showing L2 convergence"

        #Set element order of struc and mesh
        solver_params["V_S"]["deg"] = elem_order
        solver_params["Q_S"]["deg"] = elem_order
        solver_params["V_M"]["deg"] = elem_order
        
        if test == "fsi":
            problemclass = nana.NewtonAnalytic
            poplist = []
            
        elif test == "fluid":
            problemclass = nana.NewtonAnalyticFluid
            poplist = ["U_S","U_M"]
            
        elif test == "struc":
            problemclass = nana.NewtonAnalyticStruc
            poplist = ["U_F","P_F","U_M"]
        
        elif test == "mesh":
            problemclass = nana.NewtonAnalyticMesh
            poplist = ["U_F","P_F","U_S"]

        elif test == "fluidstruc":
            problemclass = nana.NewtonAnalyticFluidStruc
            poplist = ["U_M"]
            
        elif test == "strucmesh":
            problemclass = nana.NewtonAnalyticStrucMesh
            poplist = ["U_F","P_F"]
        
        elif test == "meshfluid":
            problemclass = nana.NewtonAnalyticMeshFluid
            poplist = ["U_S"]
        else:
            raise Exception("bad input")
        title = "%s %s %s "%(test,elem_order,bctype)

        info_blue("Running Analytic convergence test in the %s variables with %s BC \
                   from refinmenet level %i to %i"%(test,bctype,start_refine,end_refine))

        #Path of results relative to tests folder
        basepath = "../results/%s/%sdegree%i/" %(PROBLEMFOLDER,bctype,elem_order)
        datapath = basepath + test + "data/"
        
        #initialize lists and dictionaries
        L2errors = []
        
        #Solve the problems and plot data
        for i in range(start_refine,end_refine + 1):
            #Create output file if necessary
            if not os.path.exists(datapath): os.makedirs(datapath + "refinement" + str(i))

            problem = problemclass(num_refine = i, bctype = bctype)
                
            store = datapath + "refinement" + str(i)
            solver_params["solve"] = False
            solver_params["store"] = False
            solver_params["linear_solve"] = "PETSc"
            solver_params["runtimedata"]["fsisolver"] = store
            if not os.path.exists(store + "/newtonsolverdata"):os.makedirs(store + "/newtonsolverdata")
            solver_params["runtimedata"]["newtonsolver"] = store + "/newtonsolverdata"
            solver_params["bigblue"] = True
            solver_params["stress_coupling"]== "forward"
            solver_params["newtonsoltol"] = 1.0e-12
            
            solver = sfsi.FSINewtonSolver(problem,solver_params)
            L2_errors,integrated_L2_errors,times = self.get_data(solver,problem.end_time())
            timings.reset()

            #Get rid of unwanted functions
            for x in poplist:
                integrated_L2_errors.pop(x)

            #pickle the convergence data
            if not os.path.exists(datapath): os.makedirs(store)
            
            errorfile = open(store + "/errordata","w")
            spaceerrorfile = open(store + "/spaceerrordata","w")
            meshsizefile = open(store + "/meshsize","w")
            timesfile = open(store + "/times","w")
            
            cPickle.dump(integrated_L2_errors,errorfile)
            cPickle.dump(problem.singlemesh.hmin(),meshsizefile)
            cPickle.dump(L2_errors,spaceerrorfile)
            cPickle.dump(times,timesfile)
            
            errorfile.close()
            meshsizefile.close()
            spaceerrorfile.close()
                            
    def get_data(self,solver,end_time):
        """Time step with the solver and make a list of L2 errors"""
        L2_errors = {"U_F":[],"P_F":[],"U_S":[],"U_M":[]}
        
        meshes = {"U_F":solver.problem.fluidmesh,
                  "P_F":solver.problem.fluidmesh,
                  "U_S":solver.problem.strucmesh,
                  "P_S":solver.problem.strucmesh,
                  "U_M":solver.problem.fluidmesh}
        
        anasols = {"U_F":self.anasol.EU_F,
                   "P_F":self.anasol.EP_F,
                   "U_S":self.anasol.EU_S,
                   "U_M":self.anasol.EU_M}
        
        femsols = {"U_F":solver.U1_F,
                   "P_F":solver.P1_F,
                   "U_S":solver.U1_S,
                   "U_M":solver.U1_M}

        #not actually solving, just insuring a newtonsolver exists.
        solver.solve()
        solver.prebuild_jacobians()

        #Time =0 data
        times = [solver.t]
        for func in L2_errors.keys():
            L2_errors[func].append(errornorm(anasols[func],femsols[func],mesh = meshes[func]) )

        #Time > 0 data
        while solver.t < end_time - DOLFIN_EPS:
            solver.__time_step__()
            times.append(solver.t)
            self.anasol.update(solver.t)
            for func in L2_errors.keys():
                L2_errors[func].append(errornorm(anasols[func],femsols[func],mesh = meshes[func]) )

        #Output a timings reports
        info(timings.report_str())
        
        #call solve in order to get the post processing data
        solver.solve()

        #time integrate the errors
        integrated_L2_errors = {}
        for f in L2_errors.keys():
            k = solver.dt
            #todo calculate k at each time step
            midpoints = [(L2_errors[f][i] + L2_errors[f][i-1])*k/2 for i in range(1,len(L2_errors[f]))]
            integrated_L2_errors[f] = sum(midpoints)
                
        return L2_errors,integrated_L2_errors,times
        
if __name__ == "__main__":
    print "Type of test all,fluid,struc,mesh,fluidstruc,strucmesh,meshfluid, order, \
           bc, refinements start, refinements end"
    
    import sys
    if len(sys.argv) > 7:
        print ("Usage: python %s [all|fluid|struc|mesh|fluidstruc|strucmesh|meshfluid], \
                elem_order, bctype[normal,dirichlet,neumann],start_refine [int], end_refine [int], \
                optional(solve) [t/f]"% sys.argv[0])
        exit()
    test = sys.argv[1]
    elem_order = int(sys.argv[2])
    bctype = sys.argv[3]
    start_refine = int(sys.argv[4])
    end_refine = int(sys.argv[5])
    if len(sys.argv) == 7 and sys.argv[6] == "f":
        solve = False
    else:
        solve = True
    
    t = zTestAnalytic()
    t.setup_class()
    t.convergence_test(test,elem_order,bctype,start_refine,end_refine,solve)
