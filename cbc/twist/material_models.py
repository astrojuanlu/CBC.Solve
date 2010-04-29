__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from material_model_base import MaterialModel

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
