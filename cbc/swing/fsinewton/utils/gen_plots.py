"""Generate all plots used in Bigblue analytical FSI testing"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.swing.fsinewton.tests.run_job import arguments,bctypes,elem_orders,start,stop
from dolfin import info_blue
import os
import sys

RUNTIMEPATH = "runtimedata.py"
CONVPLOTPATH = "../tests/test_analytic_plot.py"


if __name__ == "__main__":
    #Allow for a user defined start and stop point
    if len(sys.argv) > 1:
        start = sys.argv[1]
        stop = sys.argv[2]

    for elem_order in elem_orders:
        for bctype in bctypes:
            for argument in arguments:
                
                case = "%s %s %s %s %s"%(argument,elem_order,bctype,start,stop)
                info_blue("Generating data for %s"%case)
                info_blue("")
                
                #Plot the runtimedata
                for i in range(int(start),int(stop) + 1):
                    datapath = "../results/convergence/%sdegree%s/%sdata/refinement%i"%(bctype,elem_order,argument,i)    
                    try:
                        os.system("python %s %s"%(RUNTIMEPATH,datapath))
                        info_blue("Runtime data generated!")
                    except:
                        info_blue("Error, couldn't generate the runtimedata of %s"%datapath)
                #Plot the convergence data
                try:
                    os.system("python %s %s"%(CONVPLOTPATH,case))
                    info_blue("Plot data generated!")
                    info_blue("")
                except:
                    info_blue("Could not run %s %s"%(CONVPLOTPATH,case))
                    info_blue("")
            
