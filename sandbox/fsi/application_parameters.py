__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import Parameters, File
from utils import date

def application_parameters():
    "Return default values for application parameters."

    application_parameters = Parameters("application_parameters")

    application_parameters.add("end_time", 0.25)
    application_parameters.add("dt", 0.01)
    application_parameters.add("TOL", 1e-12)
    application_parameters.add("w_h", 0.45)
    application_parameters.add("w_k", 0.45)
    application_parameters.add("w_c", 0.1)
    application_parameters.add("fraction", 0.5)
    application_parameters.add("mesh_scale", 1)
    application_parameters.add("mesh_alpha", 1.0)
    application_parameters.add("fixed_point_tol", 1e-12)

    application_parameters.add("solve_primal",     True)
    application_parameters.add("solve_dual",       True)
    application_parameters.add("estimate_error",   True)
    application_parameters.add("dorfler_marking",  True)
    application_parameters.add("uniform_timestep", False)
    application_parameters.add("convergence_test", False)

    return application_parameters

def store_application_parameters(p):
    "Store application parameters to file"

    # Save to file "application_parameters.xml"
    file = File("application_parameters.xml")
    file << p

    # Save to file based on current date
    file = File("application_parameters-%s.xml" % date())
    file << p
