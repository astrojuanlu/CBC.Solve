"""Functions and Spaces for the Monolithic FSI Problem"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import cbc.swing.fsinewton.utils.misc_func as mf

NUM_SPACES = 7  #The number of spaces in the FSI mixed formulation

class FSISpaces(object):
    """Class for the management of FSI FunctionSpaces, Functions, and the associated data"""
    
    def __init__(self,problem,params):

        self.problem = problem
        self.params = params

        self.fsispace,(self.V_F,self.Q_F,self.M_U,self.C_S,self.V_S,self.C_F,self.M_D), \
        (self.V_FC,self.Q_FC,self.M_UC,self.C_SC,self.V_SC,self.C_FC,self.M_DC) = self.__create_fsi_functionspace(problem.singlemesh)
        self.subloc = FSISubSpaceLocator(self.fsispace)
        #Dofs that lie on the fsi boundary
        self.fsidofs = {"U_F":self.__fsi_dofs(self.V_F),
                        "P_F":self.__fsi_dofs(self.Q_F),
                        "L_U":self.__fsi_dofs(self.M_U),
                        "D_S":self.__fsi_dofs(self.C_S),
                        "U_S":self.__fsi_dofs(self.V_S),
                        "D_F":self.__fsi_dofs(self.C_F),
                        "L_D":self.__fsi_dofs(self.M_D), 
                        "fsispace":self.__fsi_dofs(self.fsispace)}

        #Mesh Coordinates of the FSI Boundary
        doftionary = mf.build_doftionary(self.fsispace)
        self.fsimeshcoord = [doftionary[k] for k in self.fsidofs["P_F"]]

        #Restricted dof's
        cellfunc = self.problem.meshfunctions["cell"]
        strucdomains = self.problem.domainnums["structure"]
        fluiddomains = self.problem.domainnums["fluid"]
        
        self.restricteddofs = {"U_F":self.__removedofs("U_F",cellfunc,strucdomains),
                               "P_F":self.__removedofs("P_F",cellfunc,strucdomains),
                               "L_U":self.fsidofs["L_U"],
                               "D_S":self.__restrict(self.C_S,cellfunc,strucdomains),
                               "U_S":self.__restrict(self.V_S,cellfunc,strucdomains),
                               "D_F":self.__removedofs("D_F",cellfunc,strucdomains),
                               "L_D":self.fsidofs["L_D"]}
        self.usefuldofs = []
        
##        for space in ["U_F","P_F","D_F"]:
##            self.restricteddofs[space] += self.fsidofs[space]
        
        for space in self.restricteddofs.keys():
            self.usefuldofs += self.restricteddofs[space]
        
        assert len(self.usefuldofs) == len(set(self.usefuldofs)),\
               "error in usefuldof creation,some dofs are double counted"
        #Get a subspace locator object
        self.subloc = FSISubSpaceLocator(self.fsispace)
        #Report on the composition
        self.system_composition_report(self.fsidofs)

    def system_composition_report(self,fsidofs):
        """Report on the number and type of DOF's """
        
        #Fluid dimension
        blah, blah,(V_F,Q_F,M_U,C_S,V_S,C_F,M_D)=  self.__create_fsi_functionspace(self.problem.fluidmesh)
        fluiddofcount = V_F.dim() + Q_F.dim() + C_F.dim() - len(fsidofs["U_F"]) - len(fsidofs["P_F"]) - len(fsidofs["D_F"])

        #Struc dimension
        blah, blah,(V_F,Q_F,M_U,C_S,V_S,C_F,M_D)=  self.__create_fsi_functionspace(self.problem.strucmesh)
        structuredofcount = V_S.dim() + C_F.dim() - len(fsidofs["U_S"]) - len(fsidofs["D_S"])

        #Get the number of FSI dofs
        fsidofcount= len(fsidofs["fsispace"])

        #Add the missing lagrange multiplier Dofs for a DG0 formulation
        if self.params["M_D"]["deg"] == 0:
            if self.params["Q_F"]["deg"] != 1:
                warning("Pressure space degree must equal 1 to get the correct system composition")
            fsidofcount += 2*(len(fsidofs["P_F"]) - 1)
        if self.params["M_U"]["deg"] == 0:
            if self.params["Q_F"]["deg"] != 1:
                warning("Pressure space degree must equal 1 to get the correct system composition")
            fsidofcount += 2*(len(fsidofs["P_F"]) - 1)

        doftotal = float(fluiddofcount + structuredofcount + fsidofcount)
            
        info_blue("FSI System Composition")
        info("Total dofs = %i"%doftotal)
        info("Fluid Struc Interface")
        info("%i \t %i \t %i"%(fluiddofcount,structuredofcount,fsidofcount))
        info("%f %% \t %f %% \t %f %%"%(fluiddofcount*100.0/doftotal,structuredofcount*100.0/doftotal, \
                                        fsidofcount*100.0/doftotal))
        
    def unpack_function(self,f):
        """
        unpack a function f in FSI space into a list of subfunctions using components []
        in order to avoid the problems associated with Function.split()
        """
        return split(f)
        
    
    def __restrict(self,space,cellfunc,strucdomains):
        """Returns the dofs of the function space over the domain"""
        try:
            zero = Function(space)
        except RuntimeError:
            #Maybe the space just needs to be collapsed
            zero = Function(space.collapse())
        doflist = []
        for domain in strucdomains:
            bc = DirichletBC(space,zero,cellfunc,domain)
            doflist += bc.get_boundary_values().keys()
        return doflist

    def __removedofs(self,spacename,cellfunc,strucdomains):
        """Removes the dofs of the given domain from the function space"""
        dofs = range(self.subloc.spacebegins[spacename],self.subloc.spaceends[spacename])

        remove = self.__restrict(self.subloc.spaces[spacename],cellfunc,strucdomains)
        offset = self.subloc.spacebegins[spacename]
        remove = [r + offset for r in remove]

        for d in remove: 
            dofs.remove(d)
        return dofs
        
    def __create_fsi_functionspace(self,mesh):
        """Return the mixed function space of all variables"""
        d = mesh.topology().dim()   

        V_F = VectorFunctionSpace(mesh, self.params["V_F"]["elem"], self.params["V_F"]["deg"])
        if self.params["B_F"]["elem"] != "None":
            info("Enriching Fluid Velocity space with element %s order %s"%(self.params["B_F"]["elem"],self.params["B_F"]["deg"]))
            V_F = V_F + VectorFunctionSpace(mesh,self.params["B_F"]["elem"], self.params["B_F"]["deg"])   
        Q_F = FunctionSpace(mesh, self.params["Q_F"]["elem"], self.params["Q_F"]["deg"])      
        M_U = VectorFunctionSpace(mesh, self.params["M_U"]["elem"], self.params["M_U"]["deg"]) 
        C_S = VectorFunctionSpace(mesh, self.params["C_S"]["elem"], self.params["C_S"]["deg"]) 
        V_S = VectorFunctionSpace(mesh, self.params["V_S"]["elem"], self.params["V_S"]["deg"]) 
        C_F = VectorFunctionSpace(mesh, self.params["C_F"]["elem"], self.params["C_F"]["deg"]) 
        M_D = VectorFunctionSpace(mesh, self.params["M_D"]["elem"], self.params["M_D"]["deg"])

        fsispace = MixedFunctionSpace([V_F,Q_F,M_U,C_S,V_S,C_F,M_D])        
        return fsispace, tuple([fsispace.sub(i) for i in range(NUM_SPACES)]), \
               (V_F,Q_F,M_U,C_S,V_S,C_F,M_D)

    def create_fsi_functions(self):
        """Create independant subfunctions of FSI Space"""
        U_F = Function(self.V_FC)
        P_F = Function(self.Q_FC)
        L_U = Function(self.M_UC)
        D_S = Function(self.C_SC)
        U_S = Function(self.V_SC)
        D_F = Function(self.C_FC)
        L_D = Function(self.M_DC)
        return [U_F,P_F,L_U,D_S,U_S,D_F,L_D]

    def __fsi_dofs(self, fspace = None ):
        """Generate the Dofs on the FSI Boundary"""
        if fspace is None:
            fspace = self.fsispace
        try:
            zero = Function(fspace)
        except RuntimeError:
            #Maybe the space just needs to be collapsed
            zero = Function(fspace.collapse())
        dofs = []
        for bound in self.problem.interiorboundarynums["FSI_bound"]: 
            BC = DirichletBC(fspace,zero,
                             self.problem.meshfunctions["interiorfacet"],bound)
            dofs += BC.get_boundary_values().keys()
        return dofs

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
                       "L_U":fsispace.sub(2).collapse(),
                       "D_S":fsispace.sub(3).collapse(),
                       "U_S":fsispace.sub(4).collapse(),
                       "D_F":fsispace.sub(5).collapse(),
                       "L_D":fsispace.sub(6).collapse()}
        
        self.spacebegins = {"U_F":0,
                            "P_F":self.final_dofs[0],
                            "L_U":self.final_dofs[1],
                            "D_S":self.final_dofs[2],
                            "U_S":self.final_dofs[3],
                            "D_F":self.final_dofs[4],
                            "L_D":self.final_dofs[5]}

        self.spaceends = {"U_F":self.final_dofs[0],
                          "P_F":self.final_dofs[1],
                          "L_U":self.final_dofs[2],
                          "D_S":self.final_dofs[3],
                          "U_S":self.final_dofs[4],
                          "D_F":self.final_dofs[5],
                          "L_D":self.final_dofs[6]}
