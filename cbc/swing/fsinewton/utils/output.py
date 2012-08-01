"""Module for the storage and plotting of FSI functions"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"
from dolfin import *
import fsinewton.utils.misc_func as mf

class FSIPlotter(object):
    """Continuous plotting of fsi functions"""
    def __init__(self,U1):
        (U1_F,P1_F,L1_F,U1_S,P1_S,U1_M,L1_M) = U1.split()

        self.plotfunctions = [U1_F,P1_F,L1_F,U1_S,P1_S,U1_M]
        
        self.plottitles = ["Fluid Velocity","Fluid Pressure","Fluid LM",
                           "Structure Displacement","Structure Velocity","Mesh Displacement"]
        self.plotmodes = [None,None,None,"displacement",None,"displacement"]

        #Pass the arguments to the continuous plotting object
        self.plotter = ContinuousPlotter(self.plotfunctions,self.plottitles,self.plotmodes)

    def plot(self):
        self.plotter.plot(self.plotfunctions)

    def plotfinal(self,U1):
        mf.plot_single(U1,0,title = "Final Fluid Velocity")
        mf.plot_single(U1,3,title = "Final Structure Displacement", mode = "displacement")
        mf.plot_single(U1,5,title = "Final Mesh Displacement", mode = "displacement")

class ContinuousPlotter(object):
    """Class that allows realtime plotting with viper"""
    def __init__(self,functions,titles,modes):
        """Copy the functions locally and store the parameters"""
        self.localfunctions = [mf.extract_subfunction(f) for f in functions]
        self.title = titles
        self.modes = modes
        
    def plot(self,functions):
        """Update the functions and plot them"""
        for locfunc,newfunc,title,mode in zip(self.localfunctions,functions,self.title,self.modes):
            locfunc.assign(newfunc)
            if mode is None:
                plot(locfunc,autoposition=False, title = title)
            else:
                plot(locfunc,autoposition=False, title = title, mode = mode)


class FSIStorer(object):
    """Store FSI functions in a persistant form"""
    def __init__(self,store):
        import os
        if not os.path.exists("%s/timeseries/"%store):
                     os.makedirs("%s/timeseries/"%store)
        self.timeseries = {"U_F":TimeSeries("%s/timeseries/U_F"%store),
                           "P_F":TimeSeries("%s/timeseries/P_F"%store),
                           "L_F":TimeSeries("%s/timeseries/L_F"%store),
                           "U_S":TimeSeries("%s/timeseries/U_S"%store),
                           "P_S":TimeSeries("%s/timeseries/P_S"%store),
                           "U_M":TimeSeries("%s/timeseries/U_M"%store),
                           "L_M":TimeSeries("%s/timeseries/L_M"%store)}
        
        self.vtkfiles =  {"U_F":File("%s/vtk/U_F.pvd"%store),
                          "P_F":File("%s/vtk/P_F.pvd"%store),
                          "L_F":File("%s/vtk/L_F.pvd"%store),
                          "U_S":File("%s/vtk/U_S.pvd"%store),
                          "P_S":File("%s/vtk/P_S.pvd"%store),
                          "U_M":File("%s/vtk/U_M.pvd"%store),
                          "L_M":File("%s/vtk/L_M.pvd"%store)} 
    
    def store_solution(self,U,t):
        """Store the current solution in a dictionary with the time as the key"""

        Usave = {"U_F":mf.extract_subfunction(U.split()[0]),
                 "P_F":mf.extract_subfunction(U.split()[1]),
                 "L_F":mf.extract_subfunction(U.split()[2]),
                 "U_S":mf.extract_subfunction(U.split()[3]),
                 "P_S":mf.extract_subfunction(U.split()[4]),
                 "U_M":mf.extract_subfunction(U.split()[5]),
                 "L_M":mf.extract_subfunction(U.split()[6])}

        #loop over every FSI Function and save a timeseries and VTK file
        for k in self.timeseries.keys():
            self.timeseries[k].store(Usave[k].vector(),t)           
            self.vtkfiles[k] << Usave[k],t
