"This module specifies the Jacobian forms for the monolithic FSI problem."

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

##Throughout this module the following notation is used.

##U_F Fluid Velocity
##P_F Fluid Pressure
##L_U Fluid lagrange multiplier that enforces kinematic continuity of fluid and structure

##D_S Structure displacement
##U_S Structure Velocity

##D_F Mesh Displacement
##L_D Mesh lagrange multiplier that enforces displacement matching with structure on FSI boundary

##Test functions are related to their trial functions by the following letter substitution.
## u-> v , p-> q, l-> m, d -> c

def fsi_jacobian(Iulist,Iudotlist,Iumidlist,U1list,Umidlist,Udotlist,Vlist,dotVlist,matparams,measures,forces,normals):
    """"
    Build the residual forms for the full FSI problem
    including the fluid, structure and mesh equations

    Iulist   - Trial(increment) function list
             - IU_F,IP_F,IL_U,ID_S,IU_S,ID_F,IL_D

    Iudotlist - Trial function time derivative list
              - IU_Fmid,IP_Fmid,IL_Umid,ID_Smid,IU_Smid,ID_Fmid,IL_Dmid

    Iumidlist - Trial function time approximation list
              - IU_Fdot,IP_Fdot,IL_Udot,ID_Sdot,IU_Sdot,ID_Fdot,IL_Ddot
    
    U1list   - List of current fsi variables
             - U1_F,P1_F,L1_U,D1_S,U1_S,D1_F,L1_D

    Umidlist - List of time approximated fsi variables.
             - U_Fmid,P_Fmid,L_Umid,D_Smid,U_Smid,D_Fmid,L_Dmid

    Vlist    - List of Test functions
             - v_F,q_F,m_U,c_S,v_S,c_F,m_D

    dotVlist - List of time integrated by parts test functions used in the
             - dual problem. For the FSI jacobian dotVlist = Vlist.
               
    matparams - Dictionary of material parameters
              - mu_F,rho_F,mu_S,lmbda_S,rho_S,mu_FD,lmbbda_M

    measures  - Dictionary of measures
              - dxF,dxS,dxM,dsF,dsS,dFSI
               (dx = interior, ds = exterior boundary, dFSI = FSI interface)

    normals - Dictionary of outer normals
            - N_F, N_S           
    """
    info_blue("Creating Jacobian Forms")

    #Unpack Functions
    U1_F,P1_F,L1_U,D1_S,U1_S,D1_F,L1_D = U1list
    u_Fdot,p_Fdot,l_Fdot,u_Sdot,p_Sdot,u_Mdot,l_Mdot = Udotlist

    #Unpack Trial Functions
    IU_F,IP_F,IL_U,ID_S,IU_S,ID_F,IL_D = Iulist

    #Unpack Test Functions
    v_F,q_F,m_U,c_S,v_S,c_F,m_D = Vlist

    #Unpack Test Functions
    dotv_F,dotq_F,dotm_U,dotc_S,dotv_S,dotc_F,dotm_D = dotVlist

    #Unpack Material Parameters
    mu_F = matparams["mu_F"]
    rho_F = matparams["rho_F"]
    mu_S = matparams["mu_S"]
    lmbda_S = matparams["lmbda_S"]
    rho_S = matparams["rho_S"]
    #To the user "M-mesh", here "FD-Fluid Domain"
    mu_FD = matparams["mu_M"]
    lmbda_FD = matparams["lmbda_M"]

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

    #Unpack the time approximated functions 
    IU_Fmid,IP_Fmid,IL_Umid,ID_Smid,IU_Smid,ID_Fmid,IL_Dmid = Iumidlist
    IU_Fdot,IP_Fdot,IL_Udot,ID_Sdot,IU_Sdot,ID_Fdot,IL_Ddot = Iudotlist
    U_Fmid,P_Fmid,L_Umid,D_Smid,U_Smid,D_Fmid,L_Dmid = Umidlist
    u_Fdot,p_Fdot,l_Fdot,u_Sdot,p_Sdot,u_Mdot,l_Mdot = Udotlist
    
    #FSI Interface conditions
    #################################################################
    #Diagonal blocks
    j_F2 = J_BlockFFbound(IU_F,IL_U,v_F,m_U,dFSI)
    j_M2 = J_BlockMMbound(ID_F,IL_D,c_F,m_D,dFSI)

    #Off Diagonal blocks
    j_FS = J_BlockFSbound(IU_S,m_U,dFSI)
    j_SF = J_BlockSFbound(IU_Fmid,IP_Fmid,D_Fmid,c_S,mu_F,N_F,dFSI)
    j_SM = J_blockSMbound(D1_F,ID_F,U1_F,P1_F,mu_F,c_S,N_S,dFSI)
    j_MS = J_BlockMSbound(ID_S,m_D,dFSI)
    #################################################################

    #Main Equations
    #################################################################
    j_F1 = J_BlockFF(IU_Fdot,IU_Fmid,IP_F,U_Fmid,u_Mdot,v_F,dotv_F,q_F,D1_F,rho_F,mu_F,N_F,dxF,dsF,G_F)
    j_S1 = J_BlockSS(ID_Sdot,IU_Sdot,ID_Smid,IU_Smid,D_Smid,U_Smid,c_S,dotc_S,v_S,dotv_S,mu_S,lmbda_S,rho_S,dxS)
    j_M1 = J_BlockMM(ID_Fdot,ID_Fmid,c_F,c_F,mu_FD,lmbda_FD,dxM)
    #################################################################

    #Fluid-mesh block, occures across all of the fluid domain.
    j_FM = J_BlockFM(U_Fmid, u_Fdot,P1_F, D1_F, ID_F, u_Mdot, ID_Fdot, v_F, dotv_F, q_F, rho_F, mu_F,N_F, dxF,dsF,G_F,F_F)
    
    #Fluid-Fluid block
    j_FF = j_F1 + j_F2
    #Structure-Structure block
    j_SS = j_S1
    #Mesh-Mesh Block
    j_MM = j_M1 + j_M2

    #Fluid row
    j_F =  j_FF + j_FS + j_FM 
    #Structure row
    j_S =  j_SF + j_SS + j_SM
    #Mesh row
    j_M =         j_MS + j_MM

    j = j_F + j_S + j_M

    return j

def dD_FSigmaF(U_M,dD_F,U_F,P_F,mu_F):
    """Derivative of Sigma_F with respect to Mesh variables"""
    ret =   J(U_M)*tr(dot(grad(dD_F), inv(F(U_M))))*dot(Sigma_F(U_F, P_F, U_M, mu_F), inv(F(U_M)).T)
    ret += - J(U_M)*dot(mu_F*(dot(grad(U_F), dot(inv(F(U_M)), dot(grad(dD_F), inv(F(U_M)))))), inv(F(U_M)).T)
    ret += - J(U_M)*dot(mu_F*(dot(inv(F(U_M)).T, dot(grad(dD_F).T, dot(inv(F(U_M)).T, grad(U_F).T )))), inv(F(U_M)).T)
    ret += - J(U_M)*dot(dot(Sigma_F(U_F, P_F, U_M, mu_F), inv(F(U_M)).T), dot(grad(dD_F).T, inv(F(U_M)).T))
    return ret

def J_BlockFF(dotdU,dU,dP,U,dotU_M,v,dotv,q,U_M,rho,mu,N_F,dxF,dsF,g_F=None):
    """Fluid Diagonal Block, Fluid Domain """
    
    #DT (without ALE term)
    A_FF =  inner(dotv, rho*J(U_M)*dotdU)*dxF                              
    A_FF +=  inner(v, rho*J(U_M)*dot(dot(grad(dU), inv(F(U_M))), U - dotU_M))*dxF  
    A_FF +=  inner(v, rho*J(U_M)*dot(grad(U), dot(inv(F(U_M)), dU)))*dxF

    #div Sigma_F
    A_FF +=  inner(grad(v), J(U_M)*mu*dot(grad(dU), dot(inv(F(U_M)), inv(F(U_M)).T)))*dxF     
    A_FF +=  inner(grad(v), J(U_M)*mu*dot(inv(F(U_M)).T, dot(grad(dU).T, inv(F(U_M)).T)))*dxF 
    A_FF += -inner(grad(v), J(U_M)*dP*inv(F(U_M)).T)*dxF                                       

    #div U_F (incompressibility)
    A_FF +=  inner(q, div(J(U_M)*dot(inv(F(U_M)), dU)))*dxF

    #Do nothing BC if in use.
    if g_F is None or g_F == []:
        A_FF  += -inner(v, dot(J(U_M)*mu*dot(inv(F(U_M)).T, dot(grad(dU).T, inv(F(U_M)).T)), N_F))*dsF
        A_FF  +=  inner(v, J(U_M)*dP*dot(I, dot(inv(F(U_M)).T, N_F)))*dsF

    return A_FF

def J_BlockFFbound(dU_F,dL_F,v_F,m_U,dFSI):
    """Fluid diagonal block FSI interface"""
    Lm_U = inner(m_U,dU_F)('+')*dFSI  #u_F =P_S on dSl boundary
    Lm_U += inner(v_F,dL_F)('+')*dFSI #Lagrange Multiplier
    return Lm_U

def J_BlockFSbound(dP_S,m_U,dFSI):
    """Fluid structure Coupling"""
    C_MS = -inner(m_U('+'),dP_S('+'))*dFSI
    return C_MS

def J_BlockFM(U, dotU, P, U_M, dD_F,dotU_M, dotdD_F, v_F,dotv, q, rho, mu,N_F, dxF,ds_F,g_F = None,F_F = None):
    """Fluid mesh coupling"""
    
    #DT  
    A_FM =  inner(v_F, rho*J(U_M)*tr(dot(grad(dD_F), inv(F(U_M))))*dotU)*dxF
    A_FM +=  inner(v_F, rho*J(U_M)*tr(dot(grad(dD_F), inv(F(U_M))))*dot(grad(U), dot(inv(F(U_M)), U - dotU_M)))*dxF
    A_FM += -inner(v_F,rho*J(U_M)*dot((dot(grad(U), dot(inv(F(U_M)), \
             dot(grad(dD_F), inv(F(U_M)))))), U - dotU_M ))*dxF
    A_FM += -inner(v_F, rho*J(U_M)*dot(grad(U), dot(inv(F(U_M)),dotdD_F)))*dxF

    #SigmaF
    A_FM += inner(grad(v_F),dD_FSigmaF(U_M,dD_F,U,P,mu))*dxF

    #Div U_F (incompressibility)
    A_FM +=  inner(q, div(J(U_M)*tr(dot(grad(dD_F), inv(F(U_M))))*dot(inv(F(U_M)), U)))*dxF
    A_FM += -inner(q, div(J(U_M)*dot(dot(inv(F(U_M)), grad(dD_F)), dot(inv(F(U_M)), U))))*dxF

    ##Add the terms for the Do nothing boundary if necessary
    if g_F is None:
         #Derivative of do nothing tensor with J factored out
        dSigma  =  tr(grad(dD_F)*inv(F(U_M)))*(mu*inv(F(U_M)).T*grad(U).T - P*I)*inv(F(U_M)).T
        dSigma += -mu*inv(F(U_M)).T*grad(dD_F).T*inv(F(U_M)).T*grad(U).T*inv(F(U_M)).T
        dSigma += -(mu*inv(F(U_M)).T*grad(U).T - P*I)*inv(F(U_M)).T*grad(dD_F).T*inv(F(U_M)).T

        #Add the J                           
        dSigma = J(U_M)*dSigma        
        A_FM += -inner(v_F,dot(dSigma,N_F))*ds_F

    #If a fluid body force has been specified, it will end up here. 
    if F_F is not None:
        A_FM += -inner(v_F,J(U_M)*tr(dot(grad(dD_F),inv(F(U_M))))*F_F)*dxF
    return A_FM

def J_BlockSS(dotdD_S, dotdU_S, dD_S, dU_S, D_S, U_S, c_S,dotc_S, v_S, dotv_S, mu_S, lmbda_S, rho_S, dxS): 
    "Structure diagonal block"
    F_S = grad(D_S) + I                 #I + grad U_s
    E_S = 0.5*(F_S.T*F_S - I)           #Es in the book
    dE_S = 0.5*(grad(dD_S).T*F_S + F_S.T*grad(dD_S))#Derivative of Es wrt to US in the book
    dUsSigma_S = grad(dD_S)*(2*mu_S*E_S + lmbda_S*tr(E_S)*I) + F_S*(2*mu_S*dE_S + lmbda_S*tr(dE_S)*I)

    J_SS = inner(dotc_S, rho_S*dotdU_S)*dxS + inner(grad(c_S), dUsSigma_S)*dxS \
           + inner(dotv_S, dotdD_S - dU_S)*dxS   
    return J_SS

def J_BlockSFbound(dU_F,dP_F,U_M,c_S,mu_F,N_F,dFSI):
    "Structure fluid coupling"
    Sigma_F = PiolaTransform(_Sigma_F(dU_F, dP_F, U_M, mu_F), U_M)
    A_SF = -(inner(dot(Sigma_F('+'),N_F('-')),c_S('-')))*dFSI
    return A_SF

def J_blockSMbound(U_M,dD_F,U_F,P_F,mu_F,c_S,N_S,dFSI):
    """Structure mesh coupling"""
    A_SM = -inner(c_S('-'),dot(dD_FSigmaF(U_M,dD_F,U_F,P_F,mu_F)('+'),N_S('-')))*dFSI
    return A_SM

def J_BlockMM(dUdot_FD,dU_FD,c_F,dotc_F,mu_FD,lmbda_FD,dx_F):
    """Mesh diagonal block"""
    Sigma_FD = _Sigma_M(dU_FD, mu_FD, lmbda_FD)
    R_M = inner(dotc_F, dUdot_FD)*dx_F + inner(sym(grad(c_F)), Sigma_FD)*dx_F
    return R_M

def J_BlockMMbound(dD_F,dL_D,c_F,m_D,d_FSI):
    """Mesh diagonal block""" 
    C_MM = inner(m_D, dD_F)('+')*d_FSI 
    C_MM += inner(c_F, dL_D)('+')*d_FSI #Lagrange Multiplier
    return C_MM

def J_BlockMSbound(dD_S,m_D,d_FSI):
    """Mesh structure coupling"""
    C_MS = -inner(m_D('+'),dD_S('+'))*d_FSI
    return C_MS
