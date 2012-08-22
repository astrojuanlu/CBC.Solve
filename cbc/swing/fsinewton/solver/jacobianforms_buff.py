"""This module specifies the jacobian forms that only need to be assembled once"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from cbc.swing.operators import *
from cbc.twist import PiolaTransform
from cbc.swing.operators import Sigma_F as _Sigma_F
from cbc.swing.operators import Sigma_S as _Sigma_S
from cbc.swing.operators import Sigma_M as _Sigma_M
from cbc.swing.operators import F, J, I
from residualforms import sum

def fsi_jacobian_buffered(Iulist,Iudotlist,Iumidlist,Vlist,dotVlist,matparams,measures,forces,normals):
    """"
    Build the buffered jacobian forms for the full FSI problem
    including the fluid, structure and mesh equations that only
    need to be assembled once.

    Iulist   - Trial(increment) function list
             - Iu_F,Ip_F,Il_F,Iu_S,Ip_S,Iu_M,Il_M

    Iudotlist - Trial function time derivative list
              - Iu_Fmid,Ip_Fmid,Il_Fmid,Iu_Smid,Ip_Smid,Iu_Mmid,Il_Mmid

    Iumidlist - Trial function time approximation list
              - Iu_Fdot,Ip_Fdot,Il_Fdot,Iu_Sdot,Ip_Sdot,Iu_Mdot,Il_Mdot
    
    U1list   - List of current fsi variables
             - u1_F,p1_F,l1_F,u1_S,p1_S,u1_M,l1_M

    Umidlist - List of time approximated fsi variables.
             - u_Fmid,p_Fmid,l_Fmid,u_Smid,p_Smid,u_Mmid,l_Mmid

    Vlist    - List of Test functions
             - v_F,q_F,m_F,v_S,q_S,v_M,m_M

    dotVlist - List of time integrated by parts test functions used in the
             - dual problem. For the FSI jacobian dotVlist = Vlist.
               
    matparams - Dictionary of material parameters
              - mu_F,rho_F,mu_S,lmbda_S,rho_S

    measures  - Dictionary of measures
              - dxF,dxS,dxM,dsF,dsS,dFSI
               (dx = interior, ds = exterior boundary, dFSI = FSI interface)

    normals - Dictionary of outer normals
            - N_F, N_S           
    """
    info_blue("Creating Buffered Jacobian Forms")

    #Unpack Trial Functions
    Iu_F,Ip_F,Il_F,Iu_S,Ip_S,Iu_M,Il_M = Iulist
    Iu_Fmid,Ip_Fmid,Il_Fmid,Iu_Smid,Ip_Smid,Iu_Mmid,Il_Mmid = Iumidlist
    Iu_Fdot,Ip_Fdot,Il_Fdot,Iu_Sdot,Ip_Sdot,Iu_Mdot,Il_Mdot = Iudotlist

    #Unpack Test Functions
    v_F,q_F,m_F,v_S,q_S,v_M,m_M = Vlist

    #Unpack Test Functions
    dotv_F,dotq_F,dotm_F,dotv_S,dotq_S,dotv_M,dotm_M = dotVlist

    #Unpack Material Parameters
    mu_F = matparams["mu_F"]
    rho_F = matparams["rho_F"]
    mu_S = matparams["mu_S"]
    lmbda_S = matparams["lmbda_S"]
    rho_S = matparams["rho_S"]
    mu_M = matparams["mu_M"]
    lmbda_M = matparams["lmbda_M"]

    #Unpack Measures
    #Unpack lists of Measures
    dxF = measures["fluid"]
    dxS = measures["structure"]
    dsF = measures["fluidneumannbound"]
    dsS = measures["strucbound"]
    dFSI = measures["FSI_bound"]
    dsDN = measures["donothingbound"]

    #Unpack Normals
    N_F = normals["N_F"]
    N_S = normals["N_S"]

    #Unpack forces
    F_F = forces["F_F"]
    F_S = forces["F_S"]
    F_M = forces["F_M"]
    G_S = forces["G_S"]
    G_F = forces["G_F"]
    
    #Diagonal blocks
    j_S = J_BlockS(Iu_Sdot, Ip_Sdot, Iu_S, Ip_S, v_S,dotv_S, q_S, dotq_S, mu_S, lmbda_S, rho_S, dxS)
    j_FD = J_BlockFD(Iu_Mdot,Iu_M,v_M,dotv_M,mu_M,lmbda_M,dxF)

    #Interface block
    j_FSI = J_FSI(Iu_F,Iu_S,Ip_S,Iu_M,Il_M,Il_F,v_F,v_M,m_F,m_M,dFSI)
   
    return j_S + j_FD + j_FSI
    
def J_FSI(dU_F,dU_S,dP_S,dU_M,dL_M,dL_F,v_F,v_M,m_F,m_M,dFSIlist):
    """Fluid diagonal block FSI interface"""
    J_FSI = []
    for dFSI in dFSIlist:
        a = inner(m_F,dU_F)('+')*dFSI  #u_F =P_S on dSl boundary
        a += inner(v_F,dL_F)('+')*dFSI #Lagrange Multiplier
        a += -inner(m_F('+'),dP_S('+'))*dFSI
        a += inner(m_M, dU_M)('+')*dFSI 
        a += inner(v_M, dL_M)('+')*dFSI #Lagrange Multiplier
        a += -inner(m_M('+'),dU_S('+'))*dFSI
        J_FSI.append(a)
    return sum(J_FSI)


def J_BlockS(dotdU_S, dotdP_S, dU_S, dP_S, v_S,dotv_S, q_S, dotq_S, mu_S, lmbda_S, rho_S, dxSlist):
    J_SS = []
    for dxS in dxSlist:
        J_SS.append(inner(dotv_S, rho_S*dotdP_S)*dxS + inner(dotq_S, dotdU_S)*dxS - inner(q_S,dP_S)*dxS ) 
    return sum(J_SS)

def J_BlockFD(dUdot_M,dU_M,v_M,dotv_M,mu_M,lmbda_M,dxFlist):
    """Mesh diagonal block"""
    Sigma_M = _Sigma_M(dU_M, mu_M, lmbda_M)
    J_MM = []
    for dx_F in dxFlist:
        J_MM.append(inner(dotv_M, dUdot_M)*dx_F + inner(sym(grad(v_M)), Sigma_M)*dx_F)
    return sum(J_MM)
