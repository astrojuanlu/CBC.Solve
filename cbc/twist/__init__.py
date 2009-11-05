from dolfin import *
from kinematics import *
from material_models import LinearElastic, StVenantKirchhoff, MooneyRivlin, neoHookean, Isihara, Biderman, GentThomas
from hyperelasticity_problems import StaticHyperelasticityProblem, HyperelasticityProblem

# Optimise compilation of forms
parameters.optimize = True
