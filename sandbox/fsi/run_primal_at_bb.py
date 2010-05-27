from time import sleep

import os
from dolfin_utils.pjobs import submit
from time import *
if ".." not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] = "..:" + os.environ['PYTHONPATH']

days = 10


jobs = []

nx = [40,80]
dt = [0.1, 0.2]


for i in range(len(nx)):
    for j in range(len(dt)):
        jobs = []
        
        jobs.append(("python sandbox/fsi/primal.py " + str(nx[i]) + " "+ str(dt[j])+ ">&"+ "logfile"+ str(nx[i]) + str(dt[j])))

        submit(jobs, nodes=1, ppn=2,  keep_environment=True, walltime=24*days, dryrun=False)
        sleep(10)
