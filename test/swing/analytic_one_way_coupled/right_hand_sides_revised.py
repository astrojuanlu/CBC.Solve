__author__ = "Marie E. Rognes"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-04-27
# Last changed: 2012-04-28

from dolfin import *
from time import sleep

cpp_u_F = """
class u_F : public Expression
{
public:

  u_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];

    values[0] = 0.0;
    values[1] = 2*pi*C*x*(1 - x)*cos(pi*t)*sin(pi*t);
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
    const double x = xx[0];
    values[0] = - pow(C, 2)*pow(sin(pi*t), 2)*(1 - 2*x);
  }

  double C;
  double t;

};
"""

cpp_U_S = """
class U_S : public Expression
{
public:

  U_S() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = 0.0;
    values[1] = C*X*pow(sin(pi*t), 2)*(1 - X)*sin(pi*Y);
  }

  double C;
  double t;

};
"""

cpp_U_M = cpp_U_S

cpp_f_F = """
class f_F : public Expression
{
public:

  f_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];

    values[0] = 2*pow(C,2)*pow(sin(pi*t), 2);
    values[1] = 4*pi*C*cos(pi*t)*sin(pi*t) \
                - 2*C*x*pow(pi, 2)*pow(sin(pi*t), 2) \
                - 2*C*pow(pi, 2)*pow(x, 2)*pow(cos(pi*t), 2) \
                + 2*C*x*pow(pi, 2)*pow(cos(pi*t), 2) \
                + 2*C*pow(pi, 2)*pow(x, 2)*pow(sin(pi*t), 2);
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
    const double x = xx[0];
    const double y = xx[1];

    values[0] = 0.0;
    values[1] = C*x*2*sin(pi*t)*cos(pi*t)*pi*(1 - x)*sin(pi*y);
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
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = -3*pi*C*pow(sin(pi*t), 2)*cos(pi*Y) + 6*pi*C*X*pow(sin(pi*t), 2)*cos(pi*Y) + 8*pow(C, 2)*pow(sin(pi*Y), 2)*pow(sin(pi*t), 4) - 16*X*pow(C, 2)*pow(sin(pi*Y), 2)*pow(sin(pi*t), 4) + X*pow(pi, 2)*pow(C, 2)*pow(sin(pi*Y), 2)*pow(sin(pi*t), 4) - 6*pow(pi, 2)*pow(C, 2)*pow(X, 3)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 4) - 3*X*pow(pi, 2)*pow(C, 2)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 4) - 3*pow(pi, 2)*pow(C, 2)*pow(X, 2)*pow(sin(pi*Y), 2)*pow(sin(pi*t), 4) + 2*pow(pi, 2)*pow(C, 2)*pow(X, 3)*pow(sin(pi*Y), 2)*pow(sin(pi*t), 4) + 9*pow(pi, 2)*pow(C, 2)*pow(X, 2)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 4);
    values[1] = 2*C*pow(sin(pi*t), 2)*sin(pi*Y) - 200*C*pow(pi, 2)*pow(X, 2)*pow(cos(pi*t), 2)*sin(pi*Y) - 196*C*X*pow(pi, 2)*pow(sin(pi*t), 2)*sin(pi*Y) - 8*pi*pow(C, 2)*pow(sin(pi*t), 4)*cos(pi*Y)*sin(pi*Y) + 196*C*pow(pi, 2)*pow(X, 2)*pow(sin(pi*t), 2)*sin(pi*Y) + 200*C*X*pow(pi, 2)*pow(cos(pi*t), 2)*sin(pi*Y) - 72*pow(pi, 2)*pow(C, 3)*pow(X, 3)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 6)*sin(pi*Y) - 40*pi*pow(C, 2)*pow(X, 2)*pow(sin(pi*t), 4)*cos(pi*Y)*sin(pi*Y) - 24*pow(pi, 3)*pow(C, 2)*pow(X, 3)*pow(sin(pi*t), 4)*cos(pi*Y)*sin(pi*Y) - 18*pow(pi, 4)*pow(C, 3)*pow(X, 4)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 6)*sin(pi*Y) - 8*X*pow(pi, 2)*pow(C, 3)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 6)*sin(pi*Y) - 6*pow(pi, 4)*pow(C, 3)*pow(X, 6)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 6)*sin(pi*Y) + 6*pow(pi, 4)*pow(C, 3)*pow(X, 3)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 6)*sin(pi*Y) + 12*pow(pi, 3)*pow(C, 2)*pow(X, 2)*pow(sin(pi*t), 4)*cos(pi*Y)*sin(pi*Y) + 12*pow(pi, 3)*pow(C, 2)*pow(X, 4)*pow(sin(pi*t), 4)*cos(pi*Y)*sin(pi*Y) + 18*pow(pi, 4)*pow(C, 3)*pow(X, 5)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 6)*sin(pi*Y) + 36*pow(pi, 2)*pow(C, 3)*pow(X, 4)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 6)*sin(pi*Y) + 40*pi*X*pow(C, 2)*pow(sin(pi*t), 4)*cos(pi*Y)*sin(pi*Y) + 44*pow(pi, 2)*pow(C, 3)*pow(X, 2)*pow(cos(pi*Y), 2)*pow(sin(pi*t), 6)*sin(pi*Y) + 12*pow(C, 3)*pow(sin(pi*Y), 3)*pow(sin(pi*t), 6) - 48*X*pow(C, 3)*pow(sin(pi*Y), 3)*pow(sin(pi*t), 6) + 48*pow(C, 3)*pow(X, 2)*pow(sin(pi*Y), 3)*pow(sin(pi*t), 6) - 10*pow(pi, 2)*pow(C, 3)*pow(X, 2)*pow(sin(pi*Y), 3)*pow(sin(pi*t), 6) - 8*pow(pi, 2)*pow(C, 3)*pow(X, 4)*pow(sin(pi*Y), 3)*pow(sin(pi*t), 6) + 2*X*pow(pi, 2)*pow(C, 3)*pow(sin(pi*Y), 3)*pow(sin(pi*t), 6) + 16*pow(pi, 2)*pow(C, 3)*pow(X, 3)*pow(sin(pi*Y), 3)*pow(sin(pi*t), 6);

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
    const double pi2 = pow(pi, 2);

    values[0] = - pi*C*pow(sin(pi*t), 2)*(3 - 6*X)*cos(pi*Y);
    values[1] = C*pow(sin(pi*t), 2)*(2 - 4*pi2*pow(X, 2) + 4*X*pi2)*sin(pi*Y) \
                + 2*pi*C*X*(1 - X)*cos(pi*t)*sin(pi*Y)*sin(pi*t);
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
    const double x = xx[0];

    values[0] = 2*pi*C*cos(pi*t)*sin(pi*t) - 4*pi*C*x*cos(pi*t)*sin(pi*t);
    values[1] =  pow(C, 2)*pow(sin(pi*t), 2)*(1 - 2*x);
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
    const double C2 = pow(C, 2);
    const double X2 = pow(X, 2);
    const double sinpt2 = pow(sin(pi*t), 2);
    values[0] = - 2*pi*C*cos(pi*t)*sin(pi*t) + 4*pi*C*X*cos(pi*t)*sin(pi*t) \
                + C*sinpt2 - 2*C*X*sinpt2;
    values[1] = - C2*sinpt2 + 2*X*C2*sinpt2 \
                + 2*C2*pow(sin(pi*t), 4) - 8*X*C2*pow(sin(pi*t), 4) \
                + 8*C2*X2*pow(sin(pi*t), 4);
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
    f_F = Expression(cpp_f_F)
    F_S = Expression(cpp_F_S)
    F_M = Expression(cpp_F_M)
    g_F = Expression(cpp_g_F)
    G_S0 = Expression(cpp_G_S0)
    u_F.C = C
    p_F.C = C
    U_S.C = C
    U_M.C = C
    f_F.C = C
    F_S.C = C
    F_M.C = C
    G_S0.C = C
    g_F.C = C

    # Functions used for plotting expressions
    n = 16
    omega_F = Rectangle(0.0, 0.5, 1.0, 1.0, n, n)
    Omega_M = Rectangle(0.0, 0.5, 1.0, 1.0, n, n)
    Omega_S = Rectangle(0.0, 0.0, 1.0, 0.5, n, n)
    V_F = VectorFunctionSpace(omega_F, "Lagrange", 1)
    Q_F = FunctionSpace(omega_F, "Lagrange", 1)
    V_S = VectorFunctionSpace(Omega_S, "Lagrange", 1)
    V_M = VectorFunctionSpace(Omega_M, "Lagrange", 1)
    _u_F = Function(V_F)
    _p_F = Function(Q_F)
    _U_S = Function(V_S)
    _U_M = Function(V_M)
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

        #plot(_u_F, title="u_F", autoposition=False)
        #plot(_p_F, title="p_F", autoposition=False)
        #plot(_U_S, title="U_S", autoposition=False)
        #plot(_U_M, title="U_M", autoposition=False)
        # plot(_f_F, title="f_F", autoposition=False)
        plot(_F_S, title="F_S", autoposition=False)
        # plot(_F_M, title="F_M", autoposition=False)
        # plot(_g_F, title="g_F", autoposition=False)
        # plot(_G_S0, title="G_S0", autoposition=False)

        t += dt

        sleep(0.1)

    interactive()
