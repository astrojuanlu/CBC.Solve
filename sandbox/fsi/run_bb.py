import os
from dolfin_utils.pjobs import *
from application_parameters import *
from utils import *

#os.environ['PYTHONPATH'] = "..:../..:" + os.environ['PYTHONPATH']

# Use for quick testing on local machine
submit = lambda command, *args, **kwargs : os.system(command)

# Define problem
problem = "modified_pressure_driven_cavity"
#problem = "channel_with_flap"                # FIXME: Update to new interface
#problem = "driven_cavity_free_bottom"        # FIXME: Update to new interface
#problem = "driven_cavity_fixed_bottom"       # FIXME: Update to new interface
#problem = "leaky_cavity_free_bottom"         # FIXME: Update to new interface
#problem = "leaky_cavity_fixed_bottom"        # FIXME: Update to new interface

# Set parameters
p = application_parameters()
p["end_time"] = 0.25
p["dt"]  = 0.036
p["TOL"] = 0.0001
p["w_h"] = 0.45
p["w_k"] = 0.45
p["w_c"] = 0.1
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["dorfler_marking"]  = False
p["uniform_timestep"] = False
p["convergence_test"] = False
p["fraction"] = 0.5
p["mesh_scale"] = 1
p["mesh_alpha"] = 1.0
p["fixed_point_tol"] = 1e-12
store_application_parameters(p)

# Clean old data and submit
os.system("./clean.sh")
submit("python %s.py" % problem, nodes=1, ppn=8,  keep_environment=True, walltime=24*1000)
