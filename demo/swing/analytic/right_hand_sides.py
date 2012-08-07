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

    values[0] = C*Y*(1 - Y)*(1 - cos(t));
    values[1] = 0;
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

    values[0] = C*Y*(1 - Y)*sin(t);
    values[1] = 0;
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

    values[0] = C*X*Y*(1 - Y)*(1 - cos(t));
    values[1] = 0;
  }

  double C;
  double t;

};
"""

#Due to the nice analytical solution we get U_F = u_F
cpp_U_F = """
class U_F : public Expression
{
public:

  U_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double Y = xx[1];

    values[0] = C*Y*(1 - Y)*sin(t);
    values[1] = 0;
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

    values[0] = C*y*(1 - y)*sin(t);
    values[1] = 0;
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

    values[0] = 2*C*(1 - x)*sin(t);
  }

  double C;
  double t;

};
"""

cpp_P_F = """
class P_F : public Expression
{
public:

  P_F() : Expression(), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = 2*C*(-C*X*Y*(1 - Y)*(1 - cos(t)) + 1 - X)*sin(t);
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
    const double y = xx[1];

    values[0] = C*y*(1 - y)*cos(t);
    values[1] = 0;
  }

  double C;
  double t;

};
"""

#Here also F_F = f_F
cpp_F_F = """
class F_F : public Expression
{
public:

  F_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double Y = xx[1];

    values[0] = C*Y*(1 - Y)*cos(t);
    values[1] = 0;
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

    values[0] = 12*pow(C, 3)*(1 - cos(t))*pow((1 - 2*Y)*(1 - cos(t)), 2)
              + 100*C*Y*(1 - Y)*cos(t) + 2*C*(1 - cos(t));
    values[1] = 8*pow(C, 2)*(1 - 2*Y)*(2*(1 - cos(t)) - pow(sin(t), 2));
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

    values[0] = C*X*(-pow(Y, 2)*sin(t) + Y*sin(t) - 2*cos(t) + 2);
    values[1] = 3*C*(-2*Y*cos(t) + 2*Y + cos(t) - 1);
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
    const double y = xx[1];

    values[0] = 2*C*(1 - x)*sin(t);
    values[1] = -C*(1 - 2*y)*sin(t);
  }

  double C;
  double t;

};
"""

#If we set X = 0 this should be the same as g_F
cpp_G_F =  """
class G_F : public Expression
{
public:

  G_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = 2*C*(-C*X*Y*(1 - Y ) *(1- cos(t)) + 1 - X)*sin(t);
    values[1] = -C*(1 - 2*Y)*sin(t);
  }

  double C;
  double t;

};
"""

cpp_G_F_FSI =  """
class G_F_FSI : public Expression
{
public:

  G_F_FSI() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] =  C*(6*C*X*pow(Y,2)*cos(t)
                - 6*C*X*pow(Y,2)
                - 6*C*X*Y*cos(t)
                + 6*C*X*Y
                + C*X*cos(t)
                - C*X + 2*X- 2)*sin(t);

    values[1] = C*(4*pow(C,2)*pow(X,2)*pow(Y,3)*pow(sin(t),2)
                + 8*pow(C,2)*pow(X,2)*pow(Y,3)*cos(t)
                - 8*pow(C,2)*pow(X,2)*pow(Y,3)
                - 6*pow(C,2)*pow(X,2)*pow(Y,2)*pow(sin(t),2)
                - 12*pow(C,2)*pow(X,2)*pow(Y,2)*cos(t)
                + 12*pow(C,2)*pow(X,2)*pow(Y,2) + 2*pow(C,2)*pow(X,2)*Y*pow(sin(t),2)
                + 4*pow(C,2)*pow(X,2)*Y*cos(t)
                - 4*pow(C,2)*pow(X,2)*Y - 4*C*pow(X,2)*Y*cos(t)
                + 4*C*pow(X,2)*Y + 2*C*pow(X,2)*cos(t)
                - 2*C*pow(X,2) + 4*C*X*Y*cos(t)
                - 4*C*X*Y - 2*C*X*cos(t)
                + 2*C*X - 2*Y + 1)*sin(t);
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

    values[0] = 2*pow(C, 2)*pow(2*Y*cos(t) - 2*Y - cos(t) + 1, 2)
               - C*(6*C*X*pow(Y, 2)*cos(t)
               - 6*C*X*pow(Y, 2)
               - 6*C*X*Y*cos(t)
               + 6*C*X*Y
               + C*X*cos(t)
               - C*X
               + 2*X
               - 2)*sin(t);

    values[1] = C*(
               - 4*pow(C, 2)*pow(X, 2)*pow(Y, 3)*pow(sin(t), 3)
               - 8*pow(C, 2)*pow(X, 2)*pow(Y, 3)*sin(t)*cos(t)
               + 8*pow(C, 2)*pow(X, 2)*pow(Y, 3)*sin(t)
               + 6*pow(C, 2)*pow(X, 2)*pow(Y, 2)*pow(sin(t), 3)
               + 12*pow(C, 2)*pow(X, 2)*pow(Y, 2)*sin(t)*cos(t)
               - 12*pow(C, 2)*pow(X, 2)*pow(Y, 2)*sin(t)
               - 2*pow(C, 2)*pow(X, 2)*Y*pow(sin(t), 3)
               - 4*pow(C, 2)*pow(X, 2)*Y*sin(t)*cos(t)
               + 4*pow(C, 2)*pow(X, 2)*Y*sin(t)
               + 4*C*pow(X, 2)*Y*sin(t)*cos(t)
               - 4*C*pow(X, 2)*Y*sin(t)
               - 2*C*pow(X, 2)*sin(t)*cos(t)
               + 2*C*pow(X, 2)*sin(t)
               - 4*C*X*Y*sin(t)*cos(t)
               + 4*C*X*Y*sin(t)
               + 2*C*X*sin(t)*cos(t)
               - 2*C*X*sin(t)
               + 2*Y*sin(t)
               + 2*Y*cos(t)
               - 2*Y
               - sin(t)
               - cos(t)
               + 1);
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
    P_F = Expression(cpp_P_F)
    U_S = Expression(cpp_U_S)
    U_M = Expression(cpp_U_M)
    P_S = Expression(cpp_P_S)
    f_F = Expression(cpp_f_F)
    F_S = Expression(cpp_F_S)
    F_M = Expression(cpp_F_M)
    g_F = Expression(cpp_g_F)
    G_S0 = Expression(cpp_G_S0)
    G_F_FSI = Expression(cpp_G_F_FSI)
    u_F.C = C
    p_F.C = C
    P_F.C = C
    U_S.C = C
    U_M.C = C
    P_S.C = C
    f_F.C = C
    F_S.C = C
    F_M.C = C
    G_S0.C = C
    g_F.C = C
    G_F_FSI.C = C

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
    _P_F = Function(Q_F)
    _U_S = Function(V_S)
    _U_M = Function(V_M)
    _P_S = Function(V_S)
    _f_F = Function(V_F)
    _F_S = Function(V_S)
    _F_M = Function(V_S)
    _g_F = Function(V_F)
    _G_S0 = Function(V_S)
    _G_F_FSI = Function(V_F)
    # Animate solutions
    T = 1.5
    t = 0.0
    dt = 0.01

    while t < T:

        print t

        u_F.t = t
        p_F.t = t
        P_F.t = t
        U_S.t = t
        P_S.t = t
        U_M.t = t
        f_F.t = t
        F_S.t = t
        F_M.t = t
        g_F.t = t
        G_S0.t = t
        G_F_FSI.t = t

        _u_F.interpolate(u_F)
        _p_F.interpolate(p_F)
        _P_F.interpolate(P_F)
        _U_S.interpolate(U_S)
        _P_S.interpolate(U_S)
        _U_M.interpolate(U_M)
        _f_F.interpolate(f_F)
        _F_S.interpolate(F_S)
        _F_M.interpolate(F_M)
        _g_F.interpolate(g_F)
        _G_S0.interpolate(G_S0)
        _G_F_FSI.interpolate(G_F_FSI) 

        plot(_u_F, title="u_F", autoposition=False)
        plot(_p_F, title="p_F", autoposition=False)
        plot(_P_F, title="P_F", autoposition=False)
        plot(_U_S, title="U_S", autoposition=False)
        plot(_U_M, title="U_M", autoposition=False)
        plot(_P_S, title="P_S", autoposition=False)
        plot(_f_F, title="f_F", autoposition=False)
        plot(_F_S, title="F_S", autoposition=False)
        plot(_F_M, title="F_M", autoposition=False)
        plot(_g_F, title="g_F", autoposition=False)
        plot(_G_S0, title="G_S0", autoposition=False)
        plot(_G_F_FSI,title ="G_F_FSI", autoposition = False)
        t += dt

        sleep(0.1)

    interactive()
