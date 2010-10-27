"""

   v_t + Stuff(v, u) + I(v, s) = 0
         OtherStuff(v, u, I_e) = 0
   s_t - F(v, s) = 0



Fitz-Hugh-Nagumo

"""


def F(v, s):

    epsilon =  0.01
    gamma = 0.5
    return epsilon*(v - gamma*s)

def I(v, s):

    alpha = 0.1
    return v*(v - alpha)*(1 - v) - s
