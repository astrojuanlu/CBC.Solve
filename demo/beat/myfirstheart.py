from cbc.beat import *

class BoundaryStimulus(Expression):
    def eval(self, values, x):
        t = self.t
        if x[0] == 0.0 and x[1] == 0.0 and t > 0.01 and t < 0.1:
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
        m = Mesh("heart.xml.gz")
        return refine(m)

    def end_time(self):
        return 1.0

    def boundary_current(self):
        return BoundaryStimulus()

    def conductivities(self):
        g = Expression("0.1/(sqrt(pow(x[0] - 0.15, 2) + pow(x[1], 2)) + 0.1)")
        M_i = lambda v: g*v
        M_ie = lambda v: 1.5*g*v
        return (M_i, M_ie)

# Define cell model
cell = FitzHughNagumo(epsilon=0.01, gamma=0.5, alpha=0.1)

# Define heart
heart = MyFirstHeart(cell)

# Define solver (with time-step)
simulate = FullyImplicit(dt=0.01)

# Simulate heart
simulate(heart)
