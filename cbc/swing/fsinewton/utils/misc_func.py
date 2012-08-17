"""A collection of Utility Functions used originally by the FSI Newton Solver"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"
from dolfin import *
import numpy as np

def extract_subfunction(f):
    """
    Create a new stand alone function out of a component function in
    a mixed space if necessary.
    """
    try:
        f_out = Function(f.function_space().collapse())
        f_out.assign(f)
        return f_out
    except:
        raise Exception("misc_func.extract_subfunction failure")

def assign_to_region(f,value,meshfunc,domainnums,V = None,exclude = None):
    """Give the function f the value over the subdomain, excluding the DOFS exclude"""
    if V is None:
        V = f.function_space()
    if value is None:
        value = Function(V)
    for domain in domainnums:
        bc = DirichletBC(V,value,meshfunc,domain)
        apply_to(bc,f,exclude = exclude)
    
def apply_to(bc,f,exclude = None):
    """Apply a DirichletBC to a Function, exluding the DOFs exclude"""
    #Get the dofvals out of the BC excluding those from exclude 
    dofvals = bc.get_boundary_values()
    if not exclude == None: 
        dofvals = dict((k, v) for k, v in dofvals.items()if k not in exclude)
    #Apply the dofvals
    for dof in dofvals:
        f.vector()[dof] = dofvals[dof]
        
def L2error(a,b,dx = dx,cell_domains = None ):
    """Returns the L2 error between the two functions a and b"""
    a = inner(a - b,a - b)*dx
    A = sqrt(abs(assemble(a,cell_domains = cell_domains))) 
    return A

def plot_single(function,component,title = None,mode = None, interact = False):
    """Plot a single function in a mixed function"""
    singlefunc = function.split()[component]
    plotfunc = Function(function.function_space().sub(component).collapse())
    plotfunc.assign(singlefunc)
    if mode == None:
        plot(plotfunc, title = title)
    else:
        plot(plotfunc, title = title,mode = mode)
    if interact == True:
        interactive()

def nonzero_dofs(function,cTOL = 1.0e-2):
    """Returns the dofs with values bigger than TOL """
    dofs = {}
    for index,elem in enumerate(function.vector()):
        if elem > cTOL:
            dofs[index] = elem
    return dofs          

def save_function(function,name):
    """
    The function is saved as a file name.pvd under the folder ~/results.
    It can be viewed with paraviewer or mayavi
    """
    file = File("results/"+ name + ".pvd")
    file << function

def diffvector_report(v1,v2,TOL):
    """report on the difference between two vectors sensitive to TOL"""
    DiffV = v1 - v2
    res = []
    for i,elem in enumerate(DiffV):
        if elem > TOL:
            res.append((i,elem))
            print (i,elem)
    if res == []:
        print "No differences over TOL",TOL
        res.append("No differences over TOL")
    return res

def diffmatrix_report(m1,m2,TOL):
    DiffM = m1 - m2
    res = []
    for i,row in enumerate(DiffM):
        for j,elem in enumerate(row):
            if elem > TOL:
                print (i,j,elem)
                res.append((i,j,elem))
    if res == []:
        print "No differences over TOL",TOL
        res.append("No differences over TOL")
    return res

class NotZeroTester(object):
    def checknotzero(self,F,desc = "",dx = dx,interior_facet_domains = None, restrict = False ):
        #Create a dummy vector with 1 in all entries
        V= F.function_space() 
        d = V.mesh().topology().dim()
        exp_ones = ["1" for i in range(d)]
        v = interpolate(Expression(exp_ones),V)
        if restrict == True:
            form = inner(F,v)('-')*dx
        else:
            form = inner(F,v)*dx
        val = assemble(form, interior_facet_domains = interior_facet_domains)
        assert not near(val,0.0),"Programm Error, function " \
                   + desc + " is close to 0"

def plot_bc(solver,bc,conflict_index = None):
    """Display a function with value -1 on fluid, 1 on structure, and 0 on the given BC"""

    f_all = Function(solver.fsispace)

    #Generate Structure BC to mark the domains
    strucbc = DirichletBC(solver.Q_F,1,solver.problem.strucdomain)
    #Generate a BC in the same region as the given BC
    ourbc = DirichletBC(solver.Q_F,-1,bc.user_sub_domain())

    #Apply bc to f
    [apply_to(bc,f_all) for bc in [strucbc,ourbc]]

    title = "Fluid = 0, Structure = 1,BC = -1"
    
    #If a conflict_index given, assign a value 
    if not conflict_index == None:      
        f_all.vector()[conflict_index] = 3
        title += " conflict with initial condition = 3"
    #Extract the subfunction (Pressure space)
    f = f_all[2]
    print "expecting a plot now"
    #Plot
    plot(f, title = title)
    interactive()

def build_doftionary(V):
    """
    Builds the boundary data dictionary
        doftionary key- dofnumber, value - coordinates
    """

   # d = V.mesh().topology().dim()
    dm = V.dofmap()
   ## boundarydofs = self.get_boundary_dofs(self.V)
    mesh = V.mesh()

    #It is very import that this vector has the right length
    #It holds the local dof numbers associated to a facet
    facetdofs = np.zeros(dm.num_facet_dofs(),dtype=np.uintc)

    #Initialize dof-to-normal dictionary
    doftionary = {}
    #Loop over boundary facets
    for cell in cells(mesh):
        #create one cell (since we have CG)
        cellobj = Cell(mesh,cell.index())
        #Local to global map
        globaldofcell = dm.cell_dofs(cell.index())

        #######################################
        #Find  Dof Coordinates
        #######################################
        celldofcord = dm.tabulate_coordinates(cellobj)
        for locind,dof in enumerate(globaldofcell):
            doftionary[dof] = celldofcord[locind]
    return doftionary


