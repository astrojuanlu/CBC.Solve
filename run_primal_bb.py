import os
from time import sleep
from dolfin_utils.pjobs import submit
if ".." not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] = "..:" + os.environ['PYTHONPATH']

days = 20
jobs = []

# # Parameters
nx = [80,120,160,200,240,280,320,360,400,440,480,520,560,600,640]
dt = [0.0141421356237,0.0094280904158,0.00707106781185,0.00565685424946,0.00471404520786,0.00404061017818,0.0035355339059,0.00314269680521,0.0028284271247,0.00257129738607,0.00235702260386,0.00217571317278,0.00202030508904,0.00188561808308,0.00176776695285]
T  = [1.0]
smooth = [50]

# Loop jobs                                                                                                                                                                 
for i in range(len(nx)):
   # for j in range(len(dt)):
    for k in range(len(T)):
        for l in range(len(smooth)):
            jobs =(("python sandbox/fsi/primal.py" + " " + "nx="+ str(nx[i]) + " " + "dt=" + str(dt[i]) + " " +  "T=" + str(T[k]) + " " + "smooth="+ str(smooth[l])))
            submit(jobs, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)
            sleep(10)
