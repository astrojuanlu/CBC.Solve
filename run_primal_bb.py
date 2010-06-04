import os
from time import sleep
from dolfin_utils.pjobs import submit
if ".." not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] = "..:" + os.environ['PYTHONPATH']

days = 20
jobs = []

# # Parameters
nx = [80, 160, 320, 640]
dt = [0.05]
T  = [0.2]
smooth = [20]

# Loop jobs                                                                                                                                                                 
for i in range(len(nx)):
    for j in range(len(dt)):
        for k in range(len(T)):
            for l in range(len(smooth)):
                jobs =(("python sandbox/fsi/primal.py" + " " + "nx="+ str(nx[i]) + " " + "dt=" + str(dt[j]) + " " +  "T=" + str(T[k]) + " " + "smooth="+ str(smooth[l])))
                submit(jobs, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
                sleep(10)
                
