__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import parameters,Parameters, File, info
from utils import date
import os


# Set DOLFIN parameters
parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["log_level"] = DEBUG
parameters["form_compiler"]["representation"] = "quadrature"
##parameters["form_compiler"]["optimize"] = True

def default_parameters():
    "Return default values for solver parameters."

    p = Parameters("solver_parameters");

    p.add("solve_primal", True)
    p.add("primal_solver", "Newton") #Newton or fixpoint

    p.add("solve_dual", True)
    p.add("estimate_error", True)
    p.add("plot_solution", False)
    p.add("save_solution", True)
    p.add("save_series", True)
    p.add("uniform_timestep", False)
    p.add("uniform_mesh", False)
    p.add("dorfler_marking", True)
    p.add("global_storage", False)
    p.add("structure_element_degree", 1)
    p.add("mesh_element_degree", 1)
    p.add("max_num_refinements", 100)
    p.add("tolerance", 0.001)
    p.add("iteration_tolerance", 1e-12)
    p.add("initial_timestep", 0.05)
    p.add("num_initial_refinements", 0)
    p.add("maximum_iterations", 1000)
    p.add("num_smoothings", 50)
    p.add("w_h", 0.45)
    p.add("w_k", 0.45)
    p.add("w_c", 0.1)
    p.add("marking_fraction", 0.5)
    p.add("refinement_algorithm", "regular_cut")
    p.add("crossed_mesh", False)
    p.add("use_exact_solution", False)
    p.add("output_directory", "unspecified")
    p.add("description", "unspecified")
    p.add(default_fsinewtonsolver_parameters())

    # Hacks
    p.add("fluid_solver", "ipcs")
    #
    q = Parameters("dualsolver")
    q.add("timestepping","FE") #CG1 BE or FE
    q.add("fluid_domain_time_discretization","end-point")
    p.add(q)
    return p

#These are relevant if p["primal_solver"] = Newton
def default_fsinewtonsolver_parameters():
    p = Parameters("FSINewtonSolver")

    #Fluid velocity
    vf = Parameters("V_F")
    vf.add("deg",2)
    vf.add("elem","CG")
    p.add(vf)

    #Optional fluid velocity enrichment space for mini element
    #In two dimension use Bubble 3 element, in 3 dimensions Bubble 4.
    bf = Parameters("B_F")
    bf.add("deg","None")
    bf.add("elem","None")
    p.add(bf)

    #Fluid pressure
    qf = Parameters("Q_F")
    qf.add("deg",1)
    qf.add("elem","CG")
    p.add(qf)

    #Velocity Lagrange multiplier
    mu = Parameters("M_U")
    mu.add("deg",0)
    mu.add("elem","DG")
    p.add(mu)

    #Structure displacement
    cs = Parameters("C_S")
    cs.add("deg",1)
    cs.add("elem","CG")
    p.add(cs)

    #Structure velocity
    vs = Parameters("V_S")
    vs.add("deg",1)
    vs.add("elem","CG")
    p.add(vs)

    #Fluid domain displacement
    cf = Parameters("C_F")
    cf.add("deg",1)
    cf.add("elem","CG")
    p.add(cf)

    #Displacement Lagrange Multiplier
    md = Parameters("M_D")
    md.add("deg",0)
    md.add("elem","DG")
    p.add(md)

    p.add("solve",True)
    #Local plotting and storage by the FSINewtonSolver
    p.add("store",False)
    p.add("plot",False)
    
    rndata = Parameters("runtimedata")
    rndata.add("fsisolver","False")
    rndata.add("newtonsolver","False")
    p.add(rndata)

    #Time schemes
    #This parameter refers to how the fluid domain mapping is
    #time discretized in the Navier Stokes equation
    p.add("fluid_domain_time_discretization","end-point") #Alternative is "mid-point"

    #Run time optimization
    opt = Parameters("optimization")
    opt.add("reuse_jacobian",True)
    opt.add("simplify_jacobian",False)
    opt.add("max_reuse_jacobian",30)
    opt.add("reduce_quadrature",0) #0 means no reduction, i >0 means reduce to order i.
    p.add(opt)
    
    p.add("jacobian","buff") # "manual", "auto", "buff"
    p.add("newtonitrmax",100)
    
    ##################################
    #This parameter is only necessary for the problem class NewtonFSI
    p.add("newtonsoltol",1.0e-13)
    ##################################
    p.add("bigblue",False)
    return p
fsinewton_params = default_fsinewtonsolver_parameters().to_dict()

def set_output_directory(parameters, problem, case):
    "Set and create output directory"

    # Set output directory
    if parameters["output_directory"] == "unspecified":
        if case is None:
            parameters["output_directory"] = "results-%s-%s" % (problem, date())
        else:
            parameters["output_directory"] = "results-%s-%s" % (problem, str(case))

    # Create output directory
    dir = parameters["output_directory"]
    if not os.path.exists(dir):
        os.makedirs(dir)

def read_parameters():
    """Read parametrs from file specified on command-line or return
    default parameters if no file is specified"""

    # Read parameters from the command-line
    import sys
    p = default_parameters()
    try:
        file = File(sys.argv[1])
        file >> p
        info("Parameters read from %s." % sys.argv[1])
    except:
        info("No parameters specified, using default parameters.")

    # Set output directory
    if p["output_directory"] == "unspecified":
        p["output_directory"] = "results-%s"%date()

    return p

def store_parameters(parameters):
    "Store parameters to file and return filename"

    # Save to file application_parameters.xml
    file = File("application_parameters.xml")
    file << parameters

    # Save to file <output_directory>/application_parameters.xml
    filename = "%s/application_parameters.xml" % parameters["output_directory"]
    file = File(filename)
    file << parameters

    return filename
