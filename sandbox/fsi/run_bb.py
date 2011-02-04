import os
from dolfin_utils.pjobs import *
from parameters import *
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
p = default_parameters()
p["solve_primal"] = False
p["solve_dual"] = False
p["estimate_error"] = True
p["dorfler_marking"]  = False
p["uniform_timestep"] = False
p["tolerance"] = 0.1
p["fixedpoint_tolerance"] = 1e-8
p["w_h"] = 0.45
p["w_k"] = 0.45
p["w_c"] = 0.1
p["marking_fraction"] = 0.5
p["initial_timestep"] = 0.01
store_parameters(p)

# Clean old data (does not work if we want to solve primal/dual separately)
#os.system("./clean.sh")

# Submit job
submit("python %s.py" % problem, nodes=1, ppn=8,  keep_environment=True, walltime=24*1000)
