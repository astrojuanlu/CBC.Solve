import os
from time import sleep
from dolfin_utils.pjobs import submit
os.environ['PYTHONPATH'] = "..:../..:" + os.environ['PYTHONPATH']
days = 20

# Define problem names
problem_names = ["paper1", "channel_with_flap", "driven_cavity_free_bottom", "driven_cavity_fixed_bottom" , "leaky_cavity_free_bottom", "leaky_cavity_fixed_bottom"]

#-----------------------------------------------
# problem_names[0] = paper1
# problem_names[1] = channel_with_flap * 
# problem_names[2] = driven_cavity_free_bottom *
# problem_names[3] = driven_cavity_fixed_bottom *
# problem_names[4] = leaky_cavity_free_bottom *
# problem_names[5] = leaky_cavity_fixed_bottom *
#-----------------------------------------------

# FIXME: Fix problems marked as * to the new interface

# Define problem
problem = problem_names[0]

# Define solver options
solve_primal     = True
solve_dual       = False
estimate_error   = False
uniform_timestep = True

# Define TOL parameters
TOL = 0.1
w_h = 0.1
w_k = 0.1
w_c = 0.1

# Define mesh parameters
d_f = 0.5  # Dorfler fraction
m_a = 1.0   # Mesh constant 
ny  = 30

# Define time parameters
end_time = 8.33333333
dt  = 0.0233333333333


# Define and submit job (and clear old data)
clean = ("./clean.sh")
jobs =("python"+" "+ str(problem)+".py"+" "+"--solve_primal" + " " + str(solve_primal) + " " + "--solve_dual" + " " + str(solve_dual) + " " + "--estimate_error" + " " + str(estimate_error) + " " + "--uniform_timestep" + " " + str(uniform_timestep) + " " + "--TOL" + " " + str(TOL) + " " + "--w_h" + " " + str(w_h) + " " + "--w_k" + " " + str(w_k) + " " + "--w_c" + " " + str(w_c) + " " + "--dorfler_fraction" + " " + str(d_f) + " " + "--mesh_alpha" + " " + str(m_a) + " " + "--ny" + " " + str(ny) + " " + "--end_time" + " " + str(end_time) + " " + "--dt" + " " + str(dt))
submit(clean, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
sleep(1)
submit(jobs, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
