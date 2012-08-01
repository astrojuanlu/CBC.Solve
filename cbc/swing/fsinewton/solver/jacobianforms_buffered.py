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
    dxF = measures["dxF"]
    dxS = measures["dxS"]
    dxM = measures["dxM"]
    dsF = measures["dsF"]
    dsS = measures["dsS"]
    #dsM = measures["dsM"] (not in use at the moment)
    dFSI = measures["dFSI"]

    #Unpack Normals
    N_F = normals["N_F"]
    N_S = normals["N_S"]

    #Unpack forces
    F_F = forces["F_F"]
    F_S = forces["F_S"]
    F_M = forces["F_M"]
    G_S = forces["G_S"]
    G_F = forces["G_F"]
    
    #FSI Interface conditions, should only apply to current variables.
    #################################################################
    #Diagonal blocks
    j_F2 = J_BlockFFbound(Iu_F,Il_F,v_F,m_F,dFSI,innerbound = True)
    j_M2 = J_BlockMMbound(Iu_M,Il_M,v_M,m_M,dFSI,innerbound = True)

    #Off Diagonal blocks
    j_MS = J_BlockMSbound(Iu_S,m_M,dFSI,innerbound = True)
    j_FS = J_BlockFSbound(Ip_S,m_F,dFSI,innerbound = True)
    #################################################################

    #Unpack the time approximated functions that should not appear in
    #the interface conditions
    Iu_Fmid,Ip_Fmid,Il_Fmid,Iu_Smid,Ip_Smid,Iu_Mmid,Il_Mmid = Iumidlist
    Iu_Fdot,Ip_Fdot,Il_Fdot,Iu_Sdot,Ip_Sdot,Iu_Mdot,Il_Mdot = Iudotlist

    #Decoupled equations (Diagonal block)
    #################################################################
    j_S1 = J_BlockSS(Iu_Sdot,Ip_Sdot,Iu_Smid,Ip_Smid,v_S,dotv_S,q_S,dotq_S,mu_S,lmbda_S,rho_S,dxS)
    j_M1 = J_BlockMM(Iu_Mdot,Iu_Mmid,v_M,v_M,mu_M,lmbda_M,dxM)

    #Fluid-Fluid block
    j_FF = j_F2
    #Structure-Structure block
    j_SS = j_S1 
    #Mesh-Mesh Block
    j_MM = j_M1 + j_M2

    #Fluid row
    j_F =  j_FF + j_FS 
    #Structure row
    j_S = j_SS 
    #Mesh row
    j_M =          j_MS + j_MM

    #Define Full FSI Jacobian 
    j = j_F + j_S + j_M

    return j
    
def J_BlockFFbound(dU_F,dL_F,v_F,m_F,dFSI,innerbound):
    """Fluid diagonal block FSI interface"""
    if innerbound == False:
        LM_F = inner(m_F,dU_F)*dFSI       #U_F =P_S on dFSI boundary
        LM_F += inner(v_F,dL_F)*dFSI      #Lagrange Multiplier
    else:
        LM_F = inner(m_F,dU_F)('+')*dFSI  #u_F =P_S on dSl boundary
        LM_F += inner(v_F,dL_F)('+')*dFSI #Lagrange Multiplier
    return LM_F

def J_BlockFSbound(dP_S,m_F,dFSI,innerbound):
    """Fluid structure Coupling"""
    if innerbound == False:
        C_MS = -inner(m_F,dP_S)*dFSI
    else:
        C_MS = -inner(m_F('+'),dP_S('+'))*dFSI
    return C_MS

def J_BlockSS(dotdU_S, dotdP_S, dU_S, dP_S, v_S,dotv_S, q_S, dotq_S, mu_S, lmbda_S, rho_S, dxS): 
    J_SS = inner(dotv_S, rho_S*dotdP_S)*dxS + inner(dotq_S, dotdU_S)*dxS - inner(q_S,dP_S)*dxS   
    return J_SS

def J_BlockMM(dUdot_M,dU_M,v_M,dotv_M,mu_M,lmbda_M,dx_F):
    """Mesh diagonal block"""
    Sigma_M = _Sigma_M(dU_M, mu_M, lmbda_M)
    R_M = inner(dotv_M, dUdot_M)*dx_F + inner(sym(grad(v_M)), Sigma_M)*dx_F
    return R_M

def J_BlockMMbound(dU_M,dL_M,v_M,m_M,d_FSI,innerbound):
    """Mesh diagonal block""" 
    if innerbound == False:
        C_MM = inner(m_M, dU_M)*d_FSI 
        C_MM += inner(v_M, dL_M)*d_FSI #Lagrange Multiplier
    else:
        C_MM = inner(m_M, dU_M)('+')*d_FSI 
        C_MM += inner(v_M, dL_M)('+')*d_FSI #Lagrange Multiplier
    return C_MM

def J_BlockMSbound(dU_S,m_M,d_FSI,innerbound):
    """Mesh structure coupling"""
    if innerbound == False:
        C_MS = -inner(m_M,dU_S)*d_FSI
    else:
        C_MS = -inner(m_M('+'),dU_S('+'))*d_FSI
    return C_MS
