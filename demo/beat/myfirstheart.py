from cbc.beat import *

class BoundaryStimulus(Expression):
    def eval(self, values, x):
        t = self.t
        if x[0] == 0.0 and x[1] == 1.0 and t > 0.1 and t < 0.2:
            values[0] = 10.0
        else:
            values[0] = 0.0

class FitzHughNagumo(CardiacCellModel):

    def __init__(self, epsilon, gamma, alpha):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def F(self, v, s):
        return self.epsilon*(v - self.gamma*s)

    def I(self, v, s):
        return v*(v - self.alpha)*(1 - v) - s

class MyFirstHeart(Heart):

    def mesh(self):
        n = 10
        return Rectangle(0, 0, 4, 2, 2*n, n)

    def end_time(self):
        return 2.0

    def boundary_current(self):
        return BoundaryStimulus()

    def conductivities(self):

        M_i = lambda v: 0.3*v
        M_ie = lambda v: 0.5*v

        return (M_i, M_ie)

# Define cell model
cell = FitzHughNagumo(epsilon=0.01, gamma=0.5, alpha=0.1)

# Define heart
heart = MyFirstHeart(cell)

# Define solver (with time-step)
simulate = FullyImplicit(dt=0.01)

# Simulate heart
simulate(heart)
