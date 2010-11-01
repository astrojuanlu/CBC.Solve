import os
from time import sleep
from dolfin_utils.pjobs import submit
os.environ['PYTHONPATH'] = "..:../..:" + os.environ['PYTHONPATH']
days = 20

# Define problem names
problem_names = ["channel_with_flap", "driven_cavity_free_bottom", "driven_cavity_fixed_bottom" , "leaky_cavity_free_bottom", "leaky_cavity_fixed_bottom"]

#-----------------------------------------------
# problem_names[0] = channel_with_flap
# problem_names[1] = driven_cavity_free_bottom
# problem_names[2] = driven_cavity_fixed_bottom
# problem_names[3] = leaky_cavity_free_bottom
# problem_names[4] = leaky_cavity_fixed_bottom
#-----------------------------------------------

# Define problem
problem = problem_names[1]

# Define TOL parameters
TOL = 0.1
w_h = 0.5
w_k = 0.9
w_c = 0.1

# Define mesh parameters
d_f = 0.65  # Dorfler fraction
m_a = 1.0   # Mesh constant 
ny  = 20   

# Define time parameters
dt  = 0.04
T   = 0.5

# Define and submit job (and clear old data)
clean = ("./clean.sh")
jobs =("python"+" "+ str(problem)+".py"+" "+"--w_h"+" "+ str(w_h)+" "+"--w_k"+" "+ str(w_k) +" "+"--w_c"+" "+ str(w_c) +" "+"--T"+" "+ str(T)  +" "+"--dt"+" "+ str(dt)  +" "+"--dorfler_fraction"+" "+ str(d_f) +" "+"--mesh_alpha"+" "+ str(m_a)+" "+"--adaptive_tolerance"+" "+ str(TOL))
submit(clean, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
sleep(1)
submit(jobs, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
