"""Statistics and run time data from the FSI newton Solver """

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import os
from matplotlib.backends.backend_pdf import PdfPages

NEWTONITRFILE = "newtonitr"
MESHLMFILE = "meshlm"
FLUIDLMFILE = "fluidlm"
TIMESFILE = "times"
CONDFILE = "jacobian_cond"

class FsiRunTimeData(object):
    """Runtime data handeling and plot generation for the fsi newton solver"""
    def __init__(self,solver):
        self.solver = solver
        self.times = []
        #number of newton iterations
        self.newtonitr = []
        #precision of fluid lagrange multiplier
        self.fluidlm = []
        #precision of mesh langrange multiplier
        self.meshlm = []
        
        #Jacobian condition numbers key = (timestep,itrnumber), val = condition number
        self.j_cond = {}

        #Data regarding the performance of the MyNewtonsolver
        self.newtonsolverdata = []
        
    def store_fluid_lm(self,U_F,U_S,fsicoord):
        """
        Stores the maximum relative difference of U_F and P_S evaluated
        at the fsi interface mesh coordinates
        """
        #L2 Norm
##        dFSI = self.solver.problem.dFSI
##        domains = self.solver.problem.fsiboundfunc
##        self.fluidlm.append(assemble(inner(U_F-U_S,U_F-U_S)('+')*dFSI,interior_facet_domains = domains ))

        #Relative Error
        self.fluidlm.append(self.relative_error(U_F,U_S,fsicoord))

    def store_mesh_lm(self,D_S,D_F,fsicoord):
        """
        Stores the maximum relative difference of U_M and U_S evaluated
        at the fsi interface mesh coordinates
        """
        #L2 Norm
##        dFSI = self.solver.problem.dFSI
##        domains = self.solver.problem.fsiboundfunc
##        self.meshlm.append(assemble(inner(D_S-D_F,D_S-D_F)('+')*dFSI,interior_facet_domains = domains ))
        #Relative Error
        self.meshlm.append(self.relative_error(D_S,D_F,fsicoord))                           

    def store_cond_numbers(self,timestep,itrs,condnums):
        """Store the jacobian condition numbers"""
        pass

    def relative_error(self,f1,f2,coords):
        """calculate the max relative error of f1 and f2 at the coordinates"""
        v1 = [f1(coord) for coord in coords]
        v2 = [f2(coord) for coord in coords]
        diffs = [np.linalg.norm(x - y,2) for x,y in zip(v1,v2)]
        lengths1 = [np.linalg.norm(v,2) for v in v1]
        lengths2 = [np.linalg.norm(v,2) for v in v2]
        relativediffs = [d / max(l1,l2) for d,l1,l2 in zip(diffs,lengths1,lengths2)]
        return sum(relativediffs)/len(relativediffs)

    def plot_newtonitr(self,filepath):
        """Output the newtoniterations data"""
        if not os.path.exists(filepath): os.makedirs(filepath)
        newitr = open(filepath + "/newtonitr.txt","w")
        newitr.write("Times %s\n\n Newton Iterations %s"
                     %(str(self.times),str(self.newtonitr)))
        newitr.close()
        pdf = PdfPages(filepath + "/newtoniter")
        plt.figure()
        ax = plt.gca()
        ax.grid()
        plt.plot(self.times,self.newtonitr,"bD",linestyle = '-')
        plt.ylabel("Number of Iterations")
        plt.xlabel("time")
        plt.title("Newton Iterations per time step")
        plt.savefig(pdf, format = 'pdf')
        pdf.close()

    def plot_lm(self,filepath):
        """Plot the lagrange multiplier runtime data and write it to file"""
        if not os.path.exists(filepath): os.makedirs(filepath)
        lmdata = open(filepath + "/lmprecision.txt","w")
        lmdata.write("Times %s\n\n Velocity LM %s \n\n Displacement LM %s \n\n"
                     %(str(self.times),str(self.fluidlm),str(self.meshlm)))
        lmdata.close()
        
        plt.figure()
        plt.plot(self.times,self.fluidlm,'bD',linestyle = '-', label = "U_F - U_S")
        plt.plot(self.times,self.meshlm,'gp',linestyle = '-', label = "D_F - D_S")
        plt.ylabel("Max relative error on fsi interface verticies")
        plt.xlabel("time")
        plt.title("Precision of the lagrange multiplier conditions")
        plt.legend(loc = 0)
        plt.savefig(filepath + "/lmprecision")

    def pickle(self,path):
        if not os.path.exists(path): os.makedirs(path)
        itrfile = open(path + "/" + NEWTONITRFILE,"w")
        fluidlmfile = open(path + "/" + FLUIDLMFILE,"w")
        meshlmfile = open(path + "/" + MESHLMFILE,"w")
        timesfile = open(path + "/" + TIMESFILE,"w")
        condfile = open(path + "/" + CONDFILE,"w")
        
        cPickle.dump(self.newtonitr,itrfile)
        cPickle.dump(self.fluidlm,fluidlmfile)
        cPickle.dump(self.meshlm,meshlmfile)
        cPickle.dump(self.times,timesfile)
        cPickle.dump(self.j_cond,condfile)

        condfile.close()
        fluidlmfile.close()
        meshlmfile.close()
        timesfile.close()

    def plot_cond(self,norm,path,title):
        """Plot the condition numbers of the jacobian matricies"""
        plt.figure()
        itrs = self.j_cond.keys()
        condnum = [self.j_cond[k] for k in itrs]
        
        plt.plot(itrs,condnum,'b1',linestyle = '-')
        plt.ylabel("Condition Number of Jacobian")
        plt.xlabel("Newton Iteration")
        plt.title(title)
        plt.savefig(path)
        
    def store_newtonsolverdata(self,path, mode = "plot"):
        for nsdata in self.newtonsolverdata:
            if mode == "plot":
                nsdata.plot_convergence(path)
            else:
                nsdata.pickle(path)
            

if __name__ == "__main__":
    #Run a script to gather the newtonitr and lm data and plot it.
    import sys
    if len(sys.argv) < 2:
        raise Exception("Please enter a file path whose fsi runtimedata you want to generate")
    else:
        path = sys.argv[1]
        F = FsiRunTimeData()
        itrfile = open(path + "/" + NEWTONITRFILE,"r")
        fluidlmfile = open(path + "/" + FLUIDLMFILE,"r")
        meshlmfile = open(path + "/" + MESHLMFILE,"r")
        timesfile = open(path + "/" + TIMESFILE,"r")
        
        F.newtonitr = cPickle.load(itrfile)
        F.fluidlm = cPickle.load(fluidlmfile)
        F.meshlm = cPickle.load(meshlmfile)
        F.times = cPickle.load(timesfile)
        
        F.plot_newtonitr(path)
        F.plot_lm(path)
