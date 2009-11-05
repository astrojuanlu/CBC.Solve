__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from cbc.twist.kinematics import *
from sys import exit

class MaterialModel():
    """Base class for all hyperelastic material models """

    def __init__(self, parameters):
        self.num_parameters = 0
        self.parameters = parameters
        self.P = 0
        self.kinematic_measure = ""
        self.model_info()
        self._num_parameters_check()

    def model_info(self):
        pass

    def _num_parameters_check(self):
        if len(self.parameters) != self.num_parameters:
            print "Please provide the correct number of parameters (%s) for this material model" % self.num_parameters
            exit(2)

    def _parameters_as_functions(self, u):
        parameters = []
        for i in range(0, len(self.parameters)):
            # Handle parameters passed as numbers (floats, ints etc.)
            # differently from those passed as DOLFIN objects
            # (expressions, functions and constant etc.)
            parameter_type = str(self.parameters[i].__class__)
            if "dolfin" in parameter_type:
                parameters.append(self.parameters[i])
            else:
                print "Converting given numerical parameter to DOLFIN Constant"
                parameters.append(Constant(u.function_space().mesh(), self.parameters[i]))
        return parameters
    
    def _construct_local_kinematics(self, u):
        self.I = SecondOrderIdentity(u)
        self.epsilon = InfinitesimalStrain(u)
        self.F = DeformationGradient(u)
        self.J = Jacobian(u)
        self.C = RightCauchyGreen(u)
        self.E = GreenLagrangeStrain(u)
        self.b = LeftCauchyGreen(u)
        self.e = EulerAlmansiStrain(u)
        [self.I1, self.I2, self.I3] = CauchyGreenInvariants(u)
        [self.I1bar, self.I2bar] = IsochoricCauchyGreenInvariants(u)
        
    def strain_energy(self, parameters):
        pass

    def SecondPiolaKirchhoffStress(self, u):
        self._construct_local_kinematics(u)
        psi = self.strain_energy(MaterialModel._parameters_as_functions(self, u))

        if self.kinematic_measure == "InfinitesimalStrain":
            epsilon = self.epsilon
            S = diff(psi, epsilon)
        elif self.kinematic_measure == "RightCauchyGreen":
            C = self.C
            S = 2*diff(psi, C)            
        elif self.kinematic_measure ==  "GreenLagrangeStrain":
            E = self.E
            S = diff(psi, E)
        elif self.kinematic_measure == "CauchyGreenInvariants":
            I = self.I; C = self.C
            I1 = self.I1; I2 = self.I2; I3 = self.I3
            gamma1 = diff(psi, I1) + I1*diff(psi, I2)
            gamma2 = -diff(psi, I2)
            gamma3 = I3*diff(psi, I3)
            S = 2*(gamma1*I + gamma2*C + gamma3*inv(C))
        elif self.kinematic_measure == "IsochoricCauchyGreenInvariants":
            I = self.I; Cbar = self.Cbar
            I1bar = self.I1bar; I2bar = self.I2bar; J = self.J
            gamma1bar = diff(psibar, I1bar) + I1bar*diff(psibar, I2bar)
            gamma2bar = -diff(psibar, I2bar)
            Sbar = 2*(gamma1bar*I + gamma2bar*C_bar)
            #FIXME: This process needs to be completed

        return S

    def FirstPiolaKirchhoffStress(self, u):
        S = self.SecondPiolaKirchhoffStress(u)
        F = self.F
        P = F*S
        
        if self.kinematic_measure == "InfinitesimalStrain":
            return S
        else:
            return P

class LinearElastic(MaterialModel):
    """Defines the strain energy function for a linear elastic
    material"""

    def model_info(self):
        self.num_parameters = 2
        self.kinematic_measure = "InfinitesimalStrain"

    def strain_energy(self, parameters):
        epsilon = self.epsilon
        [mu, lmbda] = parameters    
        return lmbda/2*(tr(epsilon)**2) + mu*tr(epsilon*epsilon)

class StVenantKirchhoff(MaterialModel):
    """Defines the strain energy function for a St. Venant-Kirchhoff
    material"""

    def model_info(self):
        self.num_parameters = 2
        self.kinematic_measure = "GreenLagrangeStrain"

    def strain_energy(self, parameters):
        E = self.E
        [mu, lmbda] = parameters
        return lmbda/2*(tr(E)**2) + mu*tr(E*E)

class MooneyRivlin(MaterialModel):
    """Defines the strain energy function for a (two term)
    Mooney-Rivlin material"""        

    def model_info(self):
        self.num_parameters = 2
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        I1 = self.I1
        I2 = self.I2

        [C1, C2] = parameters
        return C1*(I1 - 3) + C2*(I2 - 3)

class neoHookean(MaterialModel):
    """Defines the strain energy function for a neo-Hookean
    material"""

    def model_info(self):
        self.num_parameters = 1
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        I1 = self.I1

        [half_nkT] = parameters
        return half_nkT*(I1 - 3)

class Isihara(MaterialModel):
    """Defines the strain energy function for an Isihara material"""

    def model_info(self):
        self.num_parameters = 3
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        I1 = self.I1
        I2 = self.I2

        [C10, C01, C20] = parameters
        return C10*(I1 - 3) + C01*(I2 - 3) + C20*(I2 - 3)**2

class Biderman(MaterialModel):
    """Defines the strain energy function for a Biderman material"""

    def model_info(self):
        self.num_parameters = 4
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        I1 = self.I1
        I2 = self.I2

        [C10, C01, C20, C30] = parameters
        return C10*(I1 - 3) + C01*(I2 - 3) + C20*(I2 - 3)**2 + C30(I1 - 3)**3

class GentThomas(MaterialModel):
    """Defines the strain energy function for a Gent-Thomas
    material"""

    def model_info(self):
        self.num_parameters = 2
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        I1 = self.I1
        I2 = self.I2

        [C1, C2] = parameters
        return C1*(I1 - 3) + C2*ln(I2/3)
