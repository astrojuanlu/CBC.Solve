"""
Statistics and run time data from the MyNewtonSolver for the fsi problem.
The convergence of the residuals of the various fsi functions are considered.
"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import matplotlib.pyplot as plt
import numpy as np
import cPickle
import os

from test.swing.fsinewton.convergence_tests.test_analytic_plot import PLOTSTYLES
from matplotlib.backends.backend_pdf import PdfPages
RESIDUALSFILE = "residuals"
class MyNewtonSolverRunTimeData(object):
    """Run time data for the behaviour of the mynewtonsolver over one time step"""

    def __init__(self,time):
        #iterations
        self.itr = []
        self.time = time
        funcdic = {"U_F":[],"P_F":[],"L_U":[],"D_S":[],"U_S":[],"D_F":[],
                   "L_D":[]}
        import copy
        #Residuals
        self.residuals = {"l2":copy.deepcopy(funcdic),"max":copy.deepcopy(funcdic)}     

    def pickle(self,path):
        residualsfile = open(path + "/" + RESIDUALSFILE,"w")
        cPickle.dump(self.residuals,residualsfile)
        residualsfile.close()

    def plot_convergence(self,path):
        """plot the convergence of the mynewtonsolver in the residual of the norms"""
        for norm in self.residuals.keys():
            title = "Residuals time = %f"%(self.time)
            time = str(self.time).replace(".","dot")
            newpath = path + "/%s"%norm
            if not os.path.exists(newpath):os.makedirs(newpath)
            newpath = newpath + "/res%s"%time
            pdf = PdfPages(newpath)
            plt.figure()
            ax = plt.gca()
            ax.set_yscale('log')
            ax.grid()
            for f in self.residuals[norm].keys():   
                    itrs = range(len(self.residuals[norm][f]))
                    plt.plot(itrs,self.residuals[norm][f],PLOTSTYLES[f],label = f, linestyle = '-')
            plt.ylabel("Residual %s error"%norm)
            plt.xlabel("Newton Iteration")
            #plt.title(title)
            plt.legend(loc=0)
            plt.savefig(pdf,format = 'pdf')
            pdf.close()

#TODO Measure the difference in successive jacobian in terms of norm. 
