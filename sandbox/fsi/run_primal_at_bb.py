from time import sleep
import os
from dolfin_utils.pjobs import submit
from time import *
if ".." not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] = "..:" + os.environ['PYTHONPATH']

days = 20
jobs = []

# Parameters                                                                                                                                                                
nx =  [80,160,240,320,400,480,560,640,720,800,880,960,1040,1120,1200,1280,1360,1440,1520,1600,1680]
dt = [0.0353553390593,0.0176776695296,0.0117851130197,0.00883883476474,0.00707106781176,0.00589255650964,0.00505076272259,0.00441941738212,0.00392837100623,0.00353553390556,0.00321412173232,0.00294627825444,0.00271964146573,0.00252538136094,0.00235702260337,0.00220970869076,0.00207972582644,0.00196418550266,0.00186080731834,0.00176776695241,0.00168358757349]
mesh_smooth = [10, 30, 50]

# Loop jobs                                                                                                                                                                 
for i in range(len(nx)):
    for j in range(len(mesh_smooth)):
        jobs.append(("python sandbox/fsi/primal.py " + str(nx[i]) + " "+ str(dt[i])+ " "+ str(mesh_smooth[j])))
        submit(jobs, nodes=1, ppn=2,  keep_environment=True, walltime=24*days, dryrun=False)
        sleep(1000)

