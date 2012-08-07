"""
This Module contains default parameters for the fsi newton solver
    - reuse_jacobian
        Prevents the jacobian from being reassembled at every newton iteration.
    - max_reuse_jacobian
        Forces reassembly of jacobian after every nth iteration.
            Note: At the start of a time step a reassembly happens anyway.
    - jacobian_forms
        "auto"   - Use the dolfin Derivative() to the symbolic derivative of the residual forms.
        "manual" - Use a combined Jacobian from jacobianforms_buffered.py and jacobianforms_step.py
        "buff"   - Buffer the jacobian from jacobianforms_buffered.py and add the jacobian from
                   jacobianforms_step.py during time steping.
    - bigblue
        If running on bigblue plots should not be generated with matplotlib, instead
        data should be pickled.
    - linear_solve
        np (better condition numbers) or PETSc (better memoory management)
    - Stress coupling
        'forward' structure gets stress from fluid
        'backward' fluid gets stress from structure
    -newtonsoltol
        newtonsolver tolerance
"""
__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import Parameters,parameters
# Set DOLFIN parameters
parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["log_level"] = DEBUG
parameters["form_compiler"]["representation"] = "quadrature"
##parameters["form_compiler"]["optimize"] = True

def default_fsinewtonsolver_parameters():
    p = Parameters("FSINewtonSolver_parameters")
    
    vf = Parameters("V_F")
    vf.add("deg",2)
    vf.add("elem","CG")
    p.add(vf)
    
    qf = Parameters("Q_F")
    qf.add("deg",1)
    qf.add("elem","CG")
    p.add(qf)
    
    lf = Parameters("L_F")
    lf.add("deg",0)
    lf.add("elem","DG")
    p.add(lf)
    
    vs = Parameters("V_S")
    vs.add("deg",1)
    vs.add("elem","CG")
    p.add(vs)
    
    qs = Parameters("Q_S")
    qs.add("deg",1)
    qs.add("elem","CG")
    p.add(qs)
    
    vm = Parameters("V_M")
    vm.add("deg",1)
    vm.add("elem","CG")
    p.add(vm)
    
    lm = Parameters("L_M")
    lm.add("deg",0)
    lm.add("elem","DG")
    p.add(lm)
    
    p.add("stress_coupling","forward")
    p.add("solve",True)
    p.add("store",False)
    p.add("plot",False)
    p.add("linear_solve","PETSc")
    
    rndata = Parameters("runtimedata")
    rndata.add("fsisolver",False)
    rndata.add("newtonsolver",False)
    p.add(rndata)
    
    p.add("jacobian","auto")
    p.add("reuse_jacobian",True)
    p.add("max_reuse_jacobian",20)
    p.add("newtonitrmax",100)
    p.add("newtonsoltol",1.0e-13)
    p.add("bigblue",False)
    return p
solver_params = default_fsinewtonsolver_parameters().to_dict()
