__author__ = "Marie E. Rognes"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-04-27
# Last changed: 2012-05-02

from dolfin import *
from time import sleep

cpp_U_S = """
class U_S : public Expression
{
public:

  U_S() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double Y = xx[1];

    values[0] = C*Y*t*(1 - Y);
    values[1] = 0.0;
  }

  double C;
  double t;

};
"""

cpp_P_S = """
class P_S : public Expression
{
public:

  P_S() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double Y = xx[1];

    values[0] = C*Y*(1 - Y);
    values[1] = 0.0;
  }

  double C;
  double t;

};
"""


cpp_U_M = """
class U_M : public Expression
{
public:

  U_M() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = C*X*Y*t*(1 - Y);
    values[1] = 0.0;
  }

  double C;
  double t;

};
"""

cpp_P_M = """
class P_M : public Expression
{
public:

  P_M() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = C*X*Y*(1 - Y);
    values[1] = 0.0;
  }

  double C;
  double t;

};
"""


cpp_u_F = """
class u_F : public Expression
{
public:

  u_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double y = xx[1];

    values[0] = C*y*(1 - y);
    values[1] = 0.0;
  }

  double C;
  double t;

};
"""

cpp_p_F = """
class p_F : public Expression
{
public:

  p_F() : Expression(), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double y = xx[1];

    values[0] = C*(1 - 2*y);
  }

  double C;
  double t;

};
"""


cpp_f_F = """
class f_F : public Expression
{
public:

  f_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    values[0] = 2*C;
    values[1] = -2*C;
  }

  double C;
  double t;

};
"""


cpp_F_S = """
class F_S : public Expression
{
public:

  F_S() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double Y = xx[1];

    values[0] = 2*C*t + 12*pow(C, 3)*pow(t,3)
               - 48*Y*pow(C, 3)*pow(t, 3) + 48*pow(C, 3)*pow(Y, 2)*pow(t,3);
    values[1] = 8*pow(C, 2)*pow(t, 2) - 16*Y*pow(C, 2)*pow(t, 2);
  }

  double C;
  double t;

};
"""

cpp_F_M = """
class F_M : public Expression
{
public:

  F_M() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = C*X*Y + 2*C*X*t - C*X*pow(Y,2);
    values[1] = -3*C*t + 6*C*Y*t;
  }

  double C;
  double t;

};
"""

cpp_g_F = """
class g_F : public Expression
{
public:

  g_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double y = xx[1];

    values[0] = C*(1 - 2*y);
    values[1] = - C + 2*C*y;
  }

  double C;
  double t;

};
"""

cpp_G_S0 = """
class G_S0 : public Expression
{
public:

  G_S0() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = C - 2*C*Y + 2*pow(C, 2)*pow(t, 2) + X*t*pow(C, 2) - 8*Y*pow(C, 2)*pow(t, 2) + 8*pow(C, 2)*pow(Y, 2)*pow(t, 2) - 4*X*Y*t*pow(C, 2) + 4*X*t*pow(C, 2)*pow(Y, 2);
    values[1] = -C + C*t + 2*C*Y - 2*C*Y*t - X*t*pow(C, 2) - 4*X*t*pow(C, 2)*pow(Y, 2) + 4*X*Y*t*pow(C, 2);

  }

  double C;
  double t;

};
"""


if __name__ == "__main__":

    # Instantiate expressions
    C = 1.0
    u_F = Expression(cpp_u_F)
    p_F = Expression(cpp_p_F)
    U_S = Expression(cpp_U_S)
    U_M = Expression(cpp_U_M)
    P_S = Expression(cpp_P_S)
    P_M = Expression(cpp_P_M)
    f_F = Expression(cpp_f_F)
    F_S = Expression(cpp_F_S)
    F_M = Expression(cpp_F_M)
    g_F = Expression(cpp_g_F)
    G_S0 = Expression(cpp_G_S0)
    u_F.C = C
    p_F.C = C
    U_S.C = C
    U_M.C = C
    P_S.C = C
    P_M.C = C
    f_F.C = C
    F_S.C = C
    F_M.C = C
    G_S0.C = C
    g_F.C = C

    # Functions used for plotting expressions
    n = 16
    omega_F = UnitSquare(n, n)
    Omega_M = UnitSquare(n, n)
    Omega_S = Rectangle(1.0, 0.0, 2.0, 1.0, n, n)
    V_F = VectorFunctionSpace(omega_F, "Lagrange", 2)
    Q_F = FunctionSpace(omega_F, "Lagrange", 1)
    V_S = VectorFunctionSpace(Omega_S, "Lagrange", 2)
    V_M = VectorFunctionSpace(Omega_M, "Lagrange", 2)
    _u_F = Function(V_F)
    _p_F = Function(Q_F)
    _U_S = Function(V_S)
    _U_M = Function(V_M)
    _P_S = Function(V_S)
    _P_M = Function(V_M)
    _f_F = Function(V_F)
    _F_S = Function(V_S)
    _F_M = Function(V_S)
    _g_F = Function(V_F)
    _G_S0 = Function(V_S)

    # Animate solutions
    T = 0.5
    t = 0.0
    dt = 0.01

    while t < T:

        print t

        u_F.t = t
        p_F.t = t
        U_S.t = t
        U_M.t = t
        f_F.t = t
        F_S.t = t
        F_M.t = t
        g_F.t = t
        G_S0.t = t

        _u_F.interpolate(u_F)
        _p_F.interpolate(p_F)
        _U_S.interpolate(U_S)
        _U_M.interpolate(U_M)
        _f_F.interpolate(f_F)
        _F_S.interpolate(F_S)
        _F_M.interpolate(F_M)
        _g_F.interpolate(g_F)
        _G_S0.interpolate(G_S0)

        plot(_u_F, title="u_F", autoposition=False)
        plot(_p_F, title="p_F", autoposition=False)
        plot(_U_S, title="U_S", autoposition=False)
        plot(_U_M, title="U_M", autoposition=False)
        plot(_P_S, title="P_S", autoposition=False)
        plot(_P_M, title="P_M", autoposition=False)
        plot(_f_F, title="f_F", autoposition=False)
        plot(_F_S, title="F_S", autoposition=False)
        plot(_F_M, title="F_M", autoposition=False)
        plot(_g_F, title="g_F", autoposition=False)
        plot(_G_S0, title="G_S0", autoposition=False)

        t += dt

        sleep(0.1)

    interactive()
