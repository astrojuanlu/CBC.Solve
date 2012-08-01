"""Functions and Spaces for the Monolithic FSI Problem"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import fsinewton.utils.misc_func as mf
import fsinewton.problems.base as pfsi

NUM_SPACES = 7  #The number of spaces in the FSI mixed formulation

class FSISpaces(object):
    """Class for the management of FSI FunctionSpaces, Functions, and the associated data"""
    
    def __init__(self,problem,params):

        self.problem = problem
        self.params = params

        #Fluid and structure domains, it is assumed that the structure includes fsidofs,
        #while the fluid does not
        self.fluid = problem.fluiddomain
        self.structure = problem.strucdomain
        
        self.fsispace,(self.V_F,self.Q_F,self.L_F,self.V_S,self.Q_S,self.V_M,self.L_M), \
        (self.V_FC,self.Q_FC,self.L_FC,self.V_SC,self.Q_SC,self.V_MC,self.L_MC) = self.__create_fsi_functionspace()
        self.subloc = FSISubSpaceLocator(self.fsispace)

        #Dofs that lie on the fsi boundary
        self.fsidofs = {"U_F":self.__fsi_dofs(self.V_F),
                        "P_F":self.__fsi_dofs(self.Q_F),
                        "L_F":self.__fsi_dofs(self.L_F),
                        "U_S":self.__fsi_dofs(self.V_S),
                        "P_S":self.__fsi_dofs(self.Q_S),
                        "U_M":self.__fsi_dofs(self.V_M),
                        "L_M":self.__fsi_dofs(self.L_M), 
                        "fsispace":self.__fsi_dofs(self.fsispace)}

        #Mesh Coordinates of the FSI Boundary
        doftionary = mf.build_doftionary(self.fsispace)
        self.fsimeshcoord = [doftionary[k] for k in self.fsidofs["P_F"]]


        #Restricted dof's
        self.restricteddofs = {"U_F":self.__removedofs("U_F",self.structure),
                               "P_F":self.__removedofs("P_F",self.structure),
                               "L_F":self.fsidofs["L_F"],
                               "U_S":self.__restrict(self.V_S,self.structure),
                               "P_S":self.__restrict(self.Q_S,self.structure),
                               "U_M":self.__removedofs("U_M",self.structure),
                               "L_M":self.fsidofs["L_M"]}

        self.usefuldofs = []
        #add the fsi dofs back to the fluid domain equations and make a list of all
        #useful dofs
##        print "FSi dofs VF"
##        print len(self.fsidofs["V_F"])
##        print
##        exit()
        
        for space in ["U_F","P_F","U_M"]:
            self.restricteddofs[space] += self.fsidofs[space]
            
        for space in self.restricteddofs.keys():
            self.usefuldofs += self.restricteddofs[space]
        
        assert len(self.usefuldofs) == len(set(self.usefuldofs)),\
               "error in usefuldof creation,some dofs are double counted" 
        
        #Get a subspace locator object
        self.subloc = FSISubSpaceLocator(self.fsispace)

    def unpack_function(self,f):
        """
        unpack a function f in FSI space into a list of subfunctions using components []
        in order to avoid the problems associated with Function.split()
        """
        [U_F,P_F,L_F,U_S,P_S,U_M,L_M] = [as_vector((f[0],f[1])),f[2], as_vector((f[3],f[4])),as_vector((f[5],f[6])),
                                      as_vector((f[7],f[8])),as_vector((f[9],f[10])),as_vector((f[11],f[12]))]
        return [U_F,P_F,L_F,U_S,P_S,U_M,L_M]
    
    def __restrict(self,space,domain):
        """Returns the dofs of the function space over the domain"""
        try:
            zero = Function(space)
        except RuntimeError:
            #Maybe the space just needs to be collapsed
            zero = Function(space.collapse())
            
        bc = DirichletBC(space,zero,domain)
        return bc.get_boundary_values().keys()

    def __removedofs(self,spacename,domain):
        """Removes the dofs of the given domain from the function space"""
        dofs = range(self.subloc.spacebegins[spacename],self.subloc.spaceends[spacename])

        remove = self.__restrict(self.subloc.spaces[spacename],domain)
        offset = self.subloc.spacebegins[spacename]
        remove = [r + offset for r in remove]


        for d in remove:
            dofs.remove(d)

        return dofs
        
    def __create_fsi_functionspace(self):
        """Return the mixed function space of all variables"""
        mesh = self.problem.mesh
            
        V_F = VectorFunctionSpace(mesh, self.params["V_F"]["elem"], self.params["V_F"]["deg"]) #Ana ord 2
        Q_F = FunctionSpace(mesh, self.params["Q_F"]["elem"], self.params["Q_F"]["deg"])       #Ana ord 3
        L_F = VectorFunctionSpace(mesh, self.params["L_F"]["elem"], self.params["L_F"]["deg"]) 
        V_S = VectorFunctionSpace(mesh, self.params["V_S"]["elem"], self.params["V_S"]["deg"]) #Ana ord 2
        Q_S = VectorFunctionSpace(mesh, self.params["Q_S"]["elem"], self.params["Q_S"]["deg"]) #Ana ord 2
        V_M = VectorFunctionSpace(mesh, self.params["V_M"]["elem"], self.params["V_M"]["deg"]) #Ana ord 3ll
        L_M = VectorFunctionSpace(mesh, self.params["L_M"]["elem"], self.params["L_M"]["deg"])
        
        fsispace = MixedFunctionSpace([V_F,Q_F,L_F,V_S,Q_S,V_M,L_M])
        
        return fsispace, tuple([fsispace.sub(i) for i in range(NUM_SPACES)]), \
               (V_F,Q_F,L_F,V_S,Q_S,V_M,L_M)

    def create_fsi_functions(self):
        """Create independant subfunctions of FSI Space"""
        u_F = Function(self.V_FC)
        p_F = Function(self.Q_FC)
        l_F = Function(self.L_FC)
        u_S = Function(self.V_SC)
        p_S = Function(self.Q_SC)
        u_M = Function(self.V_SC)
        l_M = Function(self.L_MC)
        return [u_F,p_F,l_F,u_S,p_S,u_M,l_M]

    def __fsi_dofs(self, fspace = None ):
        """Generate the Dofs on the FSI Boundary"""
        if fspace is None:
            fspace = self.fsispace
        try:
            zero = Function(fspace)
        except RuntimeError:
            #Maybe the space just needs to be collapsed
            zero = Function(fspace.collapse())
##            import numpy as np
##            zero.vector()[:] = np.ones(len(zero.vector().array()))
        BC = DirichletBC(fspace,zero,self.problem.fsiboundfunc,pfsi.FSI_BOUND)

##        #Plot the BC
##        newfunc = Function(fspace.collapse())
##        BC.apply(newfunc.vector())
##        plot(newfunc,mode = "displacement")
##        interactive()
##        exit()
        
        return BC.get_boundary_values().keys()

class SubSpaceLocator(object):
    """Give the subspace of a given DOF in a mixed space"""
    def __init__(self,mixedspace):
        """Initialize a list of last Dofs in a subspace"""
        dim_subs = [mixedspace.sub(i).dim() for i in range(mixedspace.num_sub_spaces())]
        self.final_dofs = []
        cum = 0
        for dim in dim_subs:
            cum +=dim
            self.final_dofs.append(cum)

    def subspace(self,dof):
        #Return the subspace number of the dof
        for num,finaldof in enumerate(self.final_dofs):
            if dof < finaldof:
                return num
            
    def report(self):
        """Output a report of where subspaces begin and end"""
        report = "Report of Subspaces and Dofs\n"
        for num,finaldof in enumerate(self.final_dofs):
            if num == 0:
                report += "Subspace " + str(num) + " begin " + str(num) + " end " + str(self.final_dofs[0]) + "\n"
            else:
                report += "Subspace " + str(num) + " begin " + str(self.final_dofs[num-1]) + " end " + str(finaldof) + "\n"
        report += "\n"
        return report
    
class FSISubSpaceLocator(SubSpaceLocator):
    """FSI version of the SubspaceLocator with convenience"""
    def __init__(self,fsispace):
        super(FSISubSpaceLocator,self).__init__(fsispace)
        
        self.spaces = {"U_F":fsispace.sub(0).collapse(),
                       "P_F":fsispace.sub(1).collapse(),
                       "L_F":fsispace.sub(2).collapse(),
                       "U_S":fsispace.sub(3).collapse(),
                       "P_S":fsispace.sub(4).collapse(),
                       "U_M":fsispace.sub(5).collapse(),
                       "L_M":fsispace.sub(6).collapse()}
        
        self.spacebegins = {"U_F":0,
                            "P_F":self.final_dofs[0],
                            "L_F":self.final_dofs[1],
                            "U_S":self.final_dofs[2],
                            "P_S":self.final_dofs[3],
                            "U_M":self.final_dofs[4],
                            "L_M":self.final_dofs[5]}

        self.spaceends = {"U_F":self.final_dofs[0],
                          "P_F":self.final_dofs[1],
                          "L_F":self.final_dofs[2],
                          "U_S":self.final_dofs[3],
                          "P_S":self.final_dofs[4],
                          "U_M":self.final_dofs[5],
                          "L_M":self.final_dofs[6]}
