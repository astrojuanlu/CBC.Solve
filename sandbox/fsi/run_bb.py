import os
from time import sleep
from dolfin_utils.pjobs import submit
os.environ['PYTHONPATH'] = "..:../..:" + os.environ['PYTHONPATH']
days = 20

# Define problem names
problem_names = ["pressure_driven_cavity", "channel_with_flap", "driven_cavity_free_bottom", "driven_cavity_fixed_bottom" , "leaky_cavity_free_bottom", "leaky_cavity_fixed_bottom"]

#-----------------------------------------------
# problem_names[0] = pressure_driven_cavity
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
solve_dual       = True
estimate_error   = True
uniform_timestep = False
dorfler_marking  = False

# Define TOL parameters
TOL = 0.1
w_h = 0.00000001
w_k = 0.1
w_c = 0.1
fp_tol = 1e-12

# Define mesh parameters
fraction = 0.5  
m_a = 1.0   # Mesh constant 
ny  = 3

# Define time parameters
end_time = 0.04
dt  = 0.01

# Define and submit job (and clear old data)
clean = ("./clean.sh")
jobs =("python"+" "+ str(problem)+".py"+" "+"--solve_primal" + " " + str(solve_primal) + " " + "--solve_dual" + " " + str(solve_dual) + " " + "--estimate_error" + " " + str(estimate_error) + " " + "--uniform_timestep" + " " + str(uniform_timestep) + " " + "--TOL" + " " + str(TOL) + " " + "--w_h" + " " + str(w_h) + " " + "--w_k" + " " + str(w_k) + " " + "--w_c" + " " + str(w_c) + " " + "--dorfler_marking" + " " + str(dorfler_marking) + " " + "--mesh_alpha" + " " + str(m_a) + " " + "--ny" + " " + str(ny) + " " + "--end_time" + " " + str(end_time) + " " + "--dt" + " " + str(dt) + " " + "--fixed_point_tol" + " " + str(fp_tol) + " " + "--fraction" + " " + str(fraction))
submit(clean, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
sleep(1)
submit(jobs, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)



# OLD VERSION
# import os
# from time import sleep
# from dolfin_utils.pjobs import submit
# os.environ['PYTHONPATH'] = "..:../..:" + os.environ['PYTHONPATH']
# days = 20

# # Define problem names
# problem_names = ["pressure_driven_cavity", "channel_with_flap", "driven_cavity_free_bottom", "driven_cavity_fixed_bottom" , "leaky_cavity_free_bottom", "leaky_cavity_fixed_bottom"]

# #-----------------------------------------------
# # problem_names[0] = pressure_driven_cavity
# # problem_names[1] = channel_with_flap * 
# # problem_names[2] = driven_cavity_free_bottom *
# # problem_names[3] = driven_cavity_fixed_bottom *
# # problem_names[4] = leaky_cavity_free_bottom *
# # problem_names[5] = leaky_cavity_fixed_bottom *
# #-----------------------------------------------

# # FIXME: Fix problems marked as * to the new interface

# # Define problem
# problem = problem_names[0]

# # Define solver options
# solve_primal     = True
# solve_dual       = True
# estimate_error   = True
# uniform_timestep = False
# dorfler_marking = False

# # Define TOL parameters
# TOL = 0.1
# w_h = 0.1
# w_k = 0.1
# w_c = 0.1
# fp_tol = 1e-12

# # Define mesh parameters
# fraction = 0.5  
# m_a = 1.0   # Mesh constant 
# ny  = 1

# # Define time parameters
# end_time = 0.25
# dt  = 0.01

# # Define and submit job (and clear old data)
# clean = ("./clean.sh")
# jobs =("python"+" "+ str(problem)+".py"+" "+"--solve_primal" + " " + str(solve_primal) + " " + "--solve_dual" + " " + str(solve_dual) + " " + "--estimate_error" + " " + str(estimate_error) + " " + "--uniform_timestep" + " " + str(uniform_timestep) + " " + "--TOL" + " " + str(TOL) + " " + "--w_h" + " " + str(w_h) + " " + "--w_k" + " " + str(w_k) + " " + "--w_c" + " " + str(w_c) + " " + "--dorfler_marking" + " " + str(dorfler_marking) + " " + "--mesh_alpha" + " " + str(m_a) + " " + "--ny" + " " + str(ny) + " " + "--end_time" + " " + str(end_time) + " " + "--dt" + " " + str(dt) + " " + "--fixed_point_tol" + " " + str(fp_tol) + " " + "--fraction" + str(fraction))
# submit(clean, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
# sleep(1)
# submit(jobs, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
