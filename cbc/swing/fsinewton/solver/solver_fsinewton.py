"This Module has functions used to solve the primary FSI problem"
"using Newtons Method"

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
import numpy as np
import cbc.common as ccom
import residualforms as rf
import jacobianforms as jfor
import cbc.swing.fsinewton.utils.misc_func as mf
from boundary_conditions import FSIBC
from spaces import FSISpaces
from mynewtonsolver import MyNonlinearProblem,MyNewtonSolver, \
                                            NewtonConverganceError,NanError, \
                                            MyNewtonSolverNumpy
from cbc.swing.fsinewton.utils.output import FSIPlotter, FSIStorer
from cbc.swing.parameters import fsinewton_params
from cbc.swing.fsinewton.utils.runtimedata import FsiRunTimeData
from cbc.swing.fsinewton.utils.timings import timings

class FSINewtonSolver(ccom.CBCSolver):
    """A Monolithic Newton Solver for FSI problems"""
    def __init__(self,problem,params = fsinewton_params):
        
        timings.startnext("Fsi Newton Solver init")
        info_blue("Initializing FSI Newton Solver")
        info("Using params \n" + str(params) )
        #Initialize base class
        ccom.CBCSolver.__init__(self)
        self.problem = problem
        self.params = params
        #Define the various helper objects of the fsinewton solver
        self.spaces = FSISpaces(problem,params)
        self.fsibc = FSIBC(problem,self.spaces)

        #Define mixed Functions
        self.U0 = self.__initial_state()
        self.U1 = Function(self.spaces.fsispace)

        #Define Subfunction references to the mixed functions
        (self.U1_F,self.P1_F,self.L1_U,self.D1_S,self.U1_S,self.D1_F,self.L1_D) = self.U1.split()
        (self.U0_F,self.P0_F,self.L0_U,self.D0_S,self.U0_S,self.D0_F,self.L0_D) = self.U0.split()

        #Define list tensor references to the mixed funtions
        self.U1list = self.spaces.unpack_function(self.U1)
        self.U0list = self.spaces.unpack_function(self.U0)

        #Define Mixed Trial and Test Functions
        self.IU = TrialFunctions(self.spaces.fsispace)
        self.V = TestFunctions(self.spaces.fsispace)

        #Define time relevant variables
        self.dt = self.problem.initial_step()
        self.kn = Constant(self.dt)
        self.t = 0.0
        
        #Define Time Descretized Functions
        self.Umid,self.Udot = self.time_discreteU(self.U1list,self.U0list,self.kn)
        self.IUmid,self.IUdot = self.time_discreteI(self.IU,self.kn)
        
        #Initialize any Body forces if present
        self.__init_forces()

        #Define Forms and buffered part of the jacobian matrix.
        self.r,self.j,self.j_buff = self.create_forms()
        self.runtimedata = FsiRunTimeData(self)
        timings.stop("Fsi Newton Solver init")

    def prepare_solve(self):
        """Setup helper objects for a solve"""        
        
        #Init the plotter if necessary
        if self.params["plot"]:
            self.plotter = FSIPlotter(self.U1)
        #Init Storage
        if self.params["store"] != False:
            self.storage = FSIStorer(self.params["store"])
        
        #Define nonlinear problem for newton solver.
        self.nonlinearproblem = \
        MyNonlinearProblem(self.r, self.U1, self.fsibc.bcallI,
                           self.j,J_buff = self.j_buff,
                           cell_domains = self.problem.meshfunctions["cell"],
                           interior_facet_domains = self.problem.meshfunctions["interiorfacet"],
                           exterior_facet_domains= self.problem.meshfunctions["exteriorfacet"],
                           spaces = self.spaces)
        
        #Create a Newton Solver object
        self.newtonsolver = MyNewtonSolver(self.nonlinearproblem,
                                           itrmax = self.params["newtonitrmax"],
                                           reuse_jacobian = self.params["optimization"]["reuse_jacobian"],
                                           max_reuse_jacobian = self.params["optimization"]["max_reuse_jacobian"],
                                           runtimedata = self.params["runtimedata"]["newtonsolver"],
                                           tol = self.params["newtonsoltol"],
                                           reduce_quadrature =  self.params["optimization"]["reduce_quadrature"] )
         
        info_blue("Newton Solver Tolerance is %s"%self.newtonsolver.tol)
        self.prebuild_jacobians()
        
    def solve(self):
        """Solve the FSI problem over time"""
        self.prepare_solve()
        
        #Time Loop
        info(" ".join(["\n Solving FSI problem",self.problem.__str__() ,"with Newton's method \n"]))
        if self.params["solve"] != False:
            
            #Store the initial value if necessary
            if self.params["store"] != False:
                self.storage.store_solution(self.U0,self.t)
                
            while self.t < self.problem.end_time() - DOLFIN_EPS:
                ret = self.time_step()
                if self.params["store"] != False:
                    self.storage.store_solution(self.U1,self.t)
        self.post_processing()

    def prebuild_jacobians(self):
        """Buffer or assemble jacobians if neccessary"""
        #Build the buffered jacobian if necessary
        if self.params["jacobian"] == "buff":
            self.nonlinearproblem.J_buff = self.assemble_J_buff()
        
        #Prebuild step jacobian if necessary
        if self.params["optimization"]["reuse_jacobian"] == True:
            self.newtonsolver.build_jacobian()

    def assemble_J_buff(self):
        """Assembles the buffered jacobian"""
        info("Assembling Buffered Jacobian")
        timings.startnext("Buffered Jacobian assembly")
        J_buff = assemble(self.j_buff,
                          cell_domains = self.problem.cellfunc,
                          interior_facet_domains = self.problem.fsiboundfunc,
                          exterior_facet_domains = self.problem.extboundfunc)
        timings.stop("Buffered Jacobian assembly")
        return J_buff

    def time_step(self):
        """Newton solve for the values of the FSI system at the next time level"""
        
        #update the body forces
        self.__update_forces(self.t)

        #Update the time
        self.t += self.dt
        info("\n t = %f"%self.t)
        
        #Initial guess is previous time step value
        self.U1.vector()[:] = self.U0.vector()

        #Apply initial guess BC (not homogeneous)
        for bc in self.fsibc.bcallU1_ini:
            bc.apply(self.U1.vector())

        try:
            #Call newton solve and store the last iteration for testing
            self.last_itr = self.newtonsolver.solve(t = self.t)                    
            info("Newton Solver Converged in %i iterations"%(len(self.last_itr)))

            #store fsisolver runtimedata
            if self.params["runtimedata"]["fsisolver"] != False:
                #Save the number of iterations and lagrange multiplier
                #precision for later ploting
                
                self.runtimedata.times.append(self.t)
                self.runtimedata.newtonitr.append(len(self.last_itr))
                self.runtimedata.store_fluid_lm(self.U1_F,self.U1_S,self.spaces.fsimeshcoord)
                self.runtimedata.store_mesh_lm(self.D1_F,self.D1_S,self.spaces.fsimeshcoord)

            #store newtonsolverruntimedata
            if self.params["runtimedata"]["newtonsolver"]:
                self.runtimedata.newtonsolverdata.append(self.newtonsolver.runtimedata)
            
            #Assign the new time step value to U0
            self.U0.vector()[:] = self.U1.vector()

            #Plot if necessary
            if self.params["plot"]:
                self.plotter.plot()
            
        except NewtonConverganceError as NCE:
            #Print some analysis of why convergence failed
            NCE.report()
            raise 

        except NanError as NE:
            #Print analysis of why the Nan happened
            ne.zTOL = 1.0e-5
            print NE.mess
            NE.analysis()
            raise

    def __init_forces(self):
        """Create a list of all present forces"""
        self.G_S = self.problem.structure_boundary_traction_extra()
        self.F_F = self.problem.fluid_body_force()
        self.F_S = self.problem.structure_body_force()
        self.F_M = self.problem.mesh_right_hand_side()
        self.G_F = self.problem.fluid_velocity_neumann_values()
        self.G_F_FSI = self.problem.fluid_fsi_stress()
        self.forces = [f for f in [self.G_S,self.F_F,self.F_S,self.F_M, self.G_F,self.G_F_FSI]\
                       if f is not None and f != []]

    def __update_forces(self,t):
        for f in self.forces:
            f.t = t + self.dt
            
        #Update functions connected to the problem if possible.
        try:
            self.problem.update(t, t + self.dt , self.dt)
        except:
            raise Exception("Update of functions failed")


    def create_forms(self):
        """Generate FSI residual and jacobian"""
        
        #Material Paramter Dictionary
        matparams = {"mu_F": self.problem.fluid_viscosity(),
                     "rho_F": self.problem.fluid_density(),
                     "mu_S": self.problem.structure_mu(),
                     "lmbda_S": self.problem.structure_lmbda(),
                     "rho_S": self.problem.structure_density(),
                     "mu_M": self.problem.mesh_mu(),
                     "lmbda_M": self.problem.mesh_lmbda() }

        info("Using material parameters\n" + str(matparams))

        #Turn the numbers into dolfin constants
        for k in matparams:
            matparams[k] = Constant(matparams[k])
        

        #Normals Dictionary
        normals = {"N_F":FacetNormal(self.problem.fluidmesh), \
                   "N_S":FacetNormal(self.problem.strucmesh)}
        
        #Measures dictionary
        measures = self.problem.measures

        #Forces dictionary
        forces = {"F_F":self.F_F,
                  "F_S":self.F_S,
                  "F_M":self.F_M,
                  "G_S":self.G_S,
                  "G_F":self.G_F,
                  "G_F_FSI":self.G_F_FSI}

        #Define full FSI residual and store block residuals for testing
        r,self.blockresiduals = rf.fsi_residual(self.U1list,self.Umid,self.Udot, 
                                                self.V,matparams,measures,forces,
                                                normals,self.params)
        #Calculcate Jacobian forms
        if self.params["jacobian"] == "auto":
            info("Using Automatic Jacobian")
            j_buff = None
            j = derivative(r,self.U1)
        else:
        #Not Automatic so manual calculation
            j,j_buff =  jfor.fsi_jacobian(self.IU,self.IUdot,self.IUmid,self.U1list,
                        self.Umid,self.Udot,self.V,self.V,
                         matparams,measures,forces,normals,self.params)
            if self.params["jacobian"] == "buff":
                info("Using Buffered Jacobian")
            elif self.params["jacobian"] == "manual":
                info("Using Manual Jacobian")
                j += j_buff 
                j_buff = None
            else:
                raise Exception("only auto, buff, and manual are possible jacobian parameters")
        return r,j,j_buff

    def assemble_J_buff(self):
        """Assembles the buffered jacobian"""
        info("Assembling Buffered Jacobian")
        timings.startnext("Buffered Jacobian assembly")
        J_buff = assemble(self.j_buff,
                          cell_domains = self.problem.meshfunctions["cell"],
                          interior_facet_domains = self.problem.meshfunctions["interiorfacet"],
                          exterior_facet_domains = self.problem.meshfunctions["exteriorfacet"]  )
        timings.stop("Buffered Jacobian assembly")
        return J_buff

    def __initial_state(self):
        "Get the initial state of the fsi system by inserting values from subspace functions"
        info_blue("Creating initial conditions")
        #Generate a zerovector with same dim as mesh
        d = self.problem.singlemesh.topology().dim()
        zerovec = ["0.0" for i in range(d)]

        #Take the initial data in a dictionary in whatever form it may be
        ini_data    = {"U_F":self.problem.fluid_velocity_initial_condition,\
                       "P_F":self.problem.fluid_pressure_initial_condition,\
                       "D_S":self.problem.struc_displacement_initial_condition,\
                       "U_S":self.problem.struc_velocity_initial_condition,\
                       "D_F":self.problem.mesh_displacement_initial_condition}

        spaces = self.spaces.subloc.spaces

        #Try to get all initial data as a dolfin function.
        for funcname in ini_data.keys():
            success = "Initial data created for " + funcname
            fail = "Warning, could not create initial condition for "+funcname+ " default value is 0"
            try:
                ini_data[funcname] = ini_data[funcname]()
                #Try the CBCSolver initial function method
                ini_data[funcname] = self.create_initial_condition(ini_data[funcname],spaces[funcname])
                
                #Try to strings into expressions and interpolate them
                if isinstance(ini_data[funcname], basestring):
                    ini_data[funcname] = interpolate(Expression(ini_data[funcname]),spaces[funcname])
                    info(success)

                #If a list or tuple
                elif type(ini_data[funcname]) == type([]) or type(ini_data[funcname]) == type(()):
                    #First assume there are strings in the tuple already
                    try:
                        ini_data[funcname] = interpolate(Expression(ini_data[funcname]),spaces[funcname])
                        info(success)
                    except:
                        #If this doesn't work interpolate as constant
                        ini_data[funcname] = interpolate(Constant(ini_data[funcname]),spaces[funcname])
                        info(success)
    
                #If already an expression interpolate it.
                elif isinstance(ini_data[funcname],Expression):
                    ini_data[funcname] = interpolate(ini_data[funcname],spaces[funcname])
                    info(success)

                #If a function try to project it.
                elif isinstance(ini_data[funcname],Function):
                    ini_data[funcname] = project(ini_data[funcname],spaces[funcname])
                elif ini_data[funcname] is not None:
                    warning(fail)
            except:
                warning(fail)
                ini_data[funcname] = None
        
        U0 = Function(self.spaces.fsispace)

        #insert the data into U0
        for funcname in ini_data.keys():
            if ini_data[funcname] is not None:
                print funcname
                U0.vector()[self.spaces.subloc.spacebegins[funcname]: \
                            self.spaces.subloc.spaceends[funcname]] = \
                ini_data[funcname].vector()[:]
            
        fsi_dofs = self.spaces.fsidofs["fsispace"]
        cellfunc = self.problem.meshfunctions["cell"]
        strucdomains = self.problem.domainnums["structure"]
        fluiddomains = self.problem.domainnums["fluid"]
        
        #Zero out fluid variables outside of their domain.
        mf.assign_to_region(U0,zerovec,cellfunc,strucdomains,V = self.spaces.V_F,exclude = fsi_dofs)
        mf.assign_to_region(U0,"0.0",cellfunc,strucdomains,V = self.spaces.Q_F,exclude = fsi_dofs)
        mf.assign_to_region(U0,zerovec,cellfunc,strucdomains,V = self.spaces.C_F,exclude = fsi_dofs)

        #Zero out structure variables outside of their domain
        mf.assign_to_region(U0,zerovec,cellfunc,fluiddomains,V = self.spaces.C_S,exclude = fsi_dofs)
        mf.assign_to_region(U0,zerovec,cellfunc,fluiddomains,V = self.spaces.V_S,exclude = fsi_dofs)
        return U0
        
    def time_discreteU(self,U1,U0,kn):
        Umid = tuple([(x+y)*0.5 for x,y in zip(U1,U0)])
        Udot = tuple([(x-y)*(1/kn) for x,y in zip(U1,U0)])  
        return Umid,Udot                                            
        
    def time_discreteI(self,IU,kn):
        IUmid = tuple([x*0.5 for x in IU])
        IUdot = tuple([x/kn for x in IU])  
        return IUmid,IUdot

    def post_processing(self):
        #Write a report of the timings
        info(timings.report_str())
        
        if self.params["runtimedata"]["fsisolver"] != "False":
            if  self.params["bigblue"] == False:
                
                #Create plots with matplotlibs
                self.runtimedata.plot_newtonitr(self.params["runtimedata"]["fsisolver"])
                self.runtimedata.plot_lm(self.params["runtimedata"]["fsisolver"])
                info("Total number of newton iterations is %i"%sum(self.runtimedata.newtonitr))
                
            elif self.params["bigblue"] == True:
                #cPickle data for later plotting 
                self.runtimedata.pickle(self.params["runtimedata"]["fsisolver"])

        if self.params["runtimedata"]["newtonsolver"] != "False":
            if self.params["bigblue"] == False: mode = "plot"
            else:mode = "store"
            
            self.runtimedata.store_newtonsolverdata(path = self.params["runtimedata"]["newtonsolver"],
                                                    mode = mode)
