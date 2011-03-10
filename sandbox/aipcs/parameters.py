__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import Parameters, File, info
from utils import date

def default_parameters():
    "Return default values for solver parameters."

    p = Parameters("solver_parameters");

    p.add("solve_primal", True)
    p.add("solve_dual", True)
    p.add("estimate_error", True)
    p.add("plot_solution", False)
    p.add("save_solution", True)
    p.add("save_series", True)
    p.add("uniform_timestep", False)
    p.add("uniform_mesh", False)
    p.add("dorfler_marking", False)
    p.add("global_storage", False)

    p.add("max_num_refinements", 100)
    p.add("tolerance", 0.001)
    p.add("initial_timestep", 0.05)
    p.add("num_initial_refinements", 0)
    p.add("maximum_iterations", 1000)
    p.add("w_h", 0.45)
    p.add("w_k", 0.45)
    p.add("w_c", 0.1)
    p.add("marking_fraction", 0.3)
    p.add("refinement_algorithm", "regular_cut")
    p.add("crossed_mesh", False)

    p.add("output_directory", "unspecified")
    p.add("description", "unspecified")
    p.add("goal_functional", 0)

    return p

# Note handling of reading/writing parameters below. Both read/write
# set the output directory and store parameters. This is needes so
# that we can both run demos directly and from run scripts.

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
        p["output_directory"] = "results-%s" % date()

    # Save to file <output_directory>/application_parameters.xml
    filename = "%s/application_parameters.xml" % p["output_directory"]
    file = File(filename)
    file << p

    return p

def store_parameters(p, problem, case):
    "Store parameters to file and return filename"

    # Set output directory
    if p["output_directory"] == "unspecified":
        if case is None:
            p["output_directory"] = "results-%s-%s" % (problem, date())
        else:
            p["output_directory"] = "results-%s-%s" % (problem, str(case))

    # Save to file application_parameters.xml
    file = File("application_parameters.xml")
    file << p

    # Save to file <output_directory>/application_parameters.xml
    filename = "%s/application_parameters.xml" % p["output_directory"]
    file = File(filename)
    file << p

    return filename

