import os
from dolfin_utils.pjobs import submit
os.environ['PYTHONPATH'] = "..:../..:" + os.environ['PYTHONPATH']
days = 20





jobs =(("python driven_cavity_free_bottom.py --T 0.06"))




submit(jobs, nodes=1, ppn=8,  keep_environment=True, walltime=24*days, dryrun=False)

