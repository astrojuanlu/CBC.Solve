"""
Test the solution from the FSI Newton Solver against a known analytic solution
This file should be run from the command line as a script with the following options
1.type of test
2. order of struc and mesh function spaces
3. type of boundary condition (normal,dirichlet,neumann)
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
import demo.swing.analytic.newtonanalytic as nana
import demo.swing.analytic.right_hand_sides as rhs
PROBLEMFOLDER = "convergence"

import demo.swing.analytic.analytic as ana
import cbc.swing.fsinewton.solver.solver_fsinewton as sfsi
import cbc.swing.fsinewton.solver.solver_fsinewton as fnew
import cbc.swing.fsinewton.utils.misc_func as mf
import matplotlib.pyplot as plt
from cbc.swing.parameters import fsinewton_params
from cbc.swing.fsinewton.utils.timings import timings
from test_analytic_plot import create_convergenceplots,plot_L2_errors,save_loglogdata, \
                                save_convergencedata,plot_lmerror,save_lmerror,plot_timings

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

    def convergence_test(self,test,elem_order,bctype,start_refine,end_refine,solve):
        "Generate plots showing L2 convergence"

        #Set element order of struc and mesh
        fsinewton_params["C_S"]["deg"] = elem_order
        fsinewton_params["V_S"]["deg"] = elem_order
        fsinewton_params["C_F"]["deg"] = elem_order
        #Mini element
##        fsinewton_params["V_F"]["deg"] = 1
##        fsinewton_params["B_F"]["deg"] = 3
##        fsinewton_params["B_F"]["elem"] = "Bubble"
                
        if test == "fsi":
            problemclass = nana.NewtonAnalytic
            poplist = []
            
        elif test == "fluid":
            problemclass = nana.NewtonAnalyticFluid
            poplist = ["D_S","D_F"]
            
        elif test == "struc":
            problemclass = nana.NewtonAnalyticStruc
            poplist = ["U_F","P_F","D_F"]
        
        elif test == "mesh":
            problemclass = nana.NewtonAnalyticMesh
            poplist = ["U_F","P_F","D_S"]

        elif test == "fluidstruc":
            problemclass = nana.NewtonAnalyticFluidStruc
            poplist = ["D_F"]
            
        elif test == "strucmesh":
            problemclass = nana.NewtonAnalyticStrucMesh
            poplist = ["U_F","P_F"]
        
        elif test == "meshfluid":
            problemclass = nana.NewtonAnalyticMeshFluid
            poplist = ["D_S"]
        else:
            raise Exception("bad input")
        title = "%s %s %s "%(test,elem_order,bctype)
        storefolder = test + "data/"

        info_blue("Running Analytic convergence test in the %s variables with %s BC \
                   from refinmenet level %i to %i"%(test,bctype,start_refine,end_refine))

        #Path of results relative to tests folder
        folderpath = "results/%s/%sdegree%i/" %(PROBLEMFOLDER, bctype, elem_order)
        filepath =  folderpath + test

        fsinewton_params["solve"] = solve
        fsinewton_params["linear_solve"] = "PETSc"
        fsinewton_params["bigblue"]= False
        fsinewton_params["stress_coupling"] = "forward"
        fsinewton_params["jacobian"] = "manual"
        fsinewton_params["optimization"]["simplify_jacobian"] = False
        fsinewton_params["optimization"]["reuse_jacobian"] = True
        
        #initialize lists and dictionaries
        integrated_L2errors = []
        xaxis = []
        L2errorsperfunc = {}
        Lagrangemult_error = {"L_U":[],"L_D":[]}
        timingdata = {"Jacobian assembly":[],"Residual assembly":[],"Linear solve":[]}
        numvertex = []
        
        #Create output file
        if not os.path.exists(folderpath + storefolder): os.makedirs(folderpath + storefolder)
        
        for i in range(start_refine,end_refine + 1):
            problem = problemclass(num_refine = i, bctype = bctype)   
            store = folderpath + storefolder + "refinement" + str(i)
            fsinewton_params["store"] = store
            fsinewton_params["reuse_jacobian"] = True
            
            if solve != False:
                fsinewton_params["runtimedata"]["fsisolver"] = store
                if not os.path.exists(store + "/newtonsolverdata"):
                     os.makedirs(store + "/newtonsolverdata")
                fsinewton_params["runtimedata"]["newtonsolver"] = store + "/newtonsolverdata"

            solver = sfsi.FSINewtonSolver(problem,fsinewton_params)
            solver.solve()
            xaxis.append(solver.problem.singlemesh.hmin())     

            #Generate L2 error and L2 error integrated in time.
            L2error,integrated_L2_error,times = self.__solution_error(solver)
            integrated_L2errors.append(integrated_L2_error)
                        
            #rearrange to get a sequence of errors per function
            for k in integrated_L2errors[0].keys():
                L2errorsperfunc[k] = [integrated_L2errors[i][k] for i in range(len(integrated_L2errors))]

            #Get rid of unwanted functions
            for x in poplist:
                L2errorsperfunc.pop(x)
                L2error.pop(x)

            #plot L2error vs time
            plot_L2_errors(L2error,times,store + "/L2errors",title + "ref %i"%i)
                
            #Save the convergence data
            save_convergencedata(L2errorsperfunc,folderpath + storefolder + test,
                                 start_refine, i)
         
            #Output slopes of the loglog plots relative to the previous points
            functions = ["U_F","P_F","D_S","D_F"]
            save_loglogdata(L2errorsperfunc,folderpath + storefolder + test,xaxis,functions)

            #Plot the loglog convergence plot
            create_convergenceplots(filepath, integrated_L2errors, L2errorsperfunc,title,xaxis)

            #Store Lagrange multiplier data
            newlmerrors = self.__lagrangemult_errors(solver)
            for k in Lagrangemult_error:
                Lagrangemult_error[k].append(newlmerrors[k])
            functions = ["L_U","L_D"]
            save_lmerror(Lagrangemult_error,folderpath + storefolder + test,start_refine,i)    
            save_loglogdata(Lagrangemult_error,folderpath + storefolder + test + "lm",xaxis,functions)
            
            #Plot Lagrange Multiplier data
            title = "FSI Interface Continuity, p = %i"%elem_order
            plot_lmerror(folderpath + storefolder + "/LMerrors",Lagrangemult_error,"No Title",xaxis,title)

            #save timings data
            timingdata["Jacobian assembly"].append(timings.gettime("Jacobian Assembly"))
            timingdata["Residual assembly"].append(timings.gettime("Residual assembly"))
            timingdata["Linear solve"].append(timings.gettime("PETSc linear solve"))
            numvertex.append(solver.problem.singlemesh.num_vertices())
            #plot timings data
            plot_timings(timingdata,numvertex,folderpath + storefolder)

            timings.reset()
                
    def __solution_error(self,solver):
        "Return the L2 error for solutions compared to the analytic fsi"

        L2errors,times = self.__generate_L2errordic(solver)
        integrated_L2errors = {}

        #Use the midpoint rule to get a time integrated error
        for f in L2errors.keys():
            integrated_L2errors[f] = self.midpoint_rule(solver.dt,L2errors[f])
        print "dt = ",solver.dt
        print integrated_L2errors
        return L2errors, integrated_L2errors,times

    def __lagrangemult_errors(self,solver):
        """Return time integrated Lagrange multiplier error"""
        return {"L_U":self.midpoint_rule(solver.dt,solver.runtimedata.fluidlm),
                "L_D":self.midpoint_rule(solver.dt,solver.runtimedata.meshlm)}

    def midpoint_rule(self,k,values):
        """Returns midpoint rule time integration for constant interval size k"""
        midpoints = [(values[i] + values[i-1])*k/2 for i in range(1,len(values))]
        return sum(midpoints)
    
    def __generate_L2errordic(self,solver):
        "Generate a list of L2 errors for each time step"
        solutions = solver.storage.timeseries
        times = sorted(solutions["U_F"].vector_times())
        L2errors = {"U_F":[],"P_F":[],"D_S":[],"D_F":[]}
        meshes = {"U_F":solver.problem.fluidmesh,"P_F":solver.problem.fluidmesh,
                  "D_S":solver.problem.strucmesh,"P_S":solver.problem.strucmesh,
                  "D_F":solver.problem.fluidmesh}
        anasols = {"U_F":self.anasol.EU_F,"P_F":self.anasol.EP_F,"D_S":self.anasol.EU_S,
                   "D_F":self.anasol.EU_M}
        dummyfuncs = solver.spaces.create_fsi_functions()
        dummyfuncs = {"U_F":dummyfuncs[0],"P_F":dummyfuncs[1],"D_S":dummyfuncs[3],"D_F":dummyfuncs[5]}
        
        for t in times: 
            self.anasol.update(t)
            for func in dummyfuncs.keys():
                solutions[func].retrieve(dummyfuncs[func].vector(),t)
                L2errors[func].append(errornorm(anasols[func],dummyfuncs[func],mesh = meshes[func]))
        return L2errors,times

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
