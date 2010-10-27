"""
   v_t + Stuff(v, u) + I(v, s) = 0
         OtherStuff(v, u, I_e) = 0
   s_t - F(v, s) = 0

"""

from dolfin import error

class CardiacCellModel:

    def F(v, s):
        error("Must define F = F(v, s)")

    def I(v, s):
        error("Must define I = I(v, s)")
