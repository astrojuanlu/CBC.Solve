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
    info_blue("Creating Jacobian Forms")

    #Unpack Functions
    u1_F,p1_F,l1_F,u1_S,p1_S,u1_M,l1_M = U1list
    u_Fdot,p_Fdot,l_Fdot,u_Sdot,p_Sdot,u_Mdot,l_Mdot = Udotlist

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

    #Unpack the time approximated functions 
    Iu_Fmid,Ip_Fmid,Il_Fmid,Iu_Smid,Ip_Smid,Iu_Mmid,Il_Mmid = Iumidlist
    Iu_Fdot,Ip_Fdot,Il_Fdot,Iu_Sdot,Ip_Sdot,Iu_Mdot,Il_Mdot = Iudotlist
    u_Fmid,p_Fmid,l_Fmid,u_Smid,p_Smid,u_Mmid,l_Mmid = Umidlist
    u_Fdot,p_Fdot,l_Fdot,u_Sdot,p_Sdot,u_Mdot,l_Mdot = Udotlist
    
    #FSI Interface conditions
    #################################################################
    #Diagonal blocks
    j_F2 = J_BlockFFbound(Iu_F,Il_F,v_F,m_F,dFSI,innerbound = True)
    j_M2 = J_BlockMMbound(Iu_M,Il_M,v_M,m_M,dFSI,innerbound = True)

    #Off Diagonal blocks
    j_FS = J_BlockFSbound(Ip_S,m_F,dFSI,innerbound = True)
    j_SF = J_BlockSFbound(Iu_Fmid,Ip_Fmid,u_Mmid,v_S,mu_F,N_F,dFSI,innerbound = True)
    j_SM = J_blockSMbound(u1_M,Iu_M,u1_F,p1_F,mu_F,v_S,N_S,dFSI,innerbound = True)
    j_MS = J_BlockMSbound(Iu_S,m_M,dFSI,innerbound = True)
    #################################################################

    #Main Equations
    #################################################################
    j_F1 = J_BlockFF(Iu_Fdot,Iu_Fmid,Ip_F,u_Fmid,u_Mdot,v_F,dotv_F,q_F,u1_M,rho_F,mu_F,N_F,dxF,dsF,G_F)
    j_S1 = J_BlockSS(Iu_Sdot,Ip_Sdot,Iu_Smid,Ip_Smid,u_Smid,p_Smid,v_S,dotv_S,q_S,dotq_S,mu_S,lmbda_S,rho_S,dxS)
    j_M1 = J_BlockMM(Iu_Mdot,Iu_Mmid,v_M,v_M,mu_M,lmbda_M,dxM)
    #################################################################

    #Fluid-mesh block, occures across all of the fluid domain.
    j_FM = J_BlockFM(u_Fmid, u_Fdot,p1_F, u1_M, Iu_M, u_Mdot, Iu_Mdot, v_F, dotv_F, q_F, rho_F, mu_F,N_F, dxF,dsF,G_F,F_F)
    
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

def dU_MSigmaF(U_M,dU_M,U_F,P_F,mu_F):
    """Derivative of Sigma_F with respect to Mesh variables"""
    ret =   J(U_M)*tr(dot(grad(dU_M), inv(F(U_M))))*dot(Sigma_F(U_F, P_F, U_M, mu_F), inv(F(U_M)).T)
    ret += - J(U_M)*dot(mu_F*(dot(grad(U_F), dot(inv(F(U_M)), dot(grad(dU_M), inv(F(U_M)))))), inv(F(U_M)).T)
    ret += - J(U_M)*dot(mu_F*(dot(inv(F(U_M)).T, dot(grad(dU_M).T, dot(inv(F(U_M)).T, grad(U_F).T )))), inv(F(U_M)).T)
    ret += - J(U_M)*dot(dot(Sigma_F(U_F, P_F, U_M, mu_F), inv(F(U_M)).T), dot(grad(dU_M).T, inv(F(U_M)).T))
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
    if g_F is None:
##        print U_M
##        print dU
##        print N_F
##        print dsF
##        exit()
        A_FF  += -inner(v, dot(J(U_M)*mu*dot(inv(F(U_M)).T, dot(grad(dU).T, inv(F(U_M)).T)), N_F))*dsF
        A_FF  +=  inner(v, J(U_M)*dP*dot(I, dot(inv(F(U_M)).T, N_F)))*dsF

    return A_FF

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

def J_BlockFM(U, dotU, P, U_M, dU_M,dotU_M, dotdU_M, v_F,dotv, q, rho, mu,N_F, dxF,ds_F,g_F = None,F_F = None):
    """Fluid mesh coupling"""
    
    #DT  
    A_FM =  inner(v_F, rho*J(U_M)*tr(dot(grad(dU_M), inv(F(U_M))))*dotU)*dxF
    A_FM +=  inner(v_F, rho*J(U_M)*tr(dot(grad(dU_M), inv(F(U_M))))*dot(grad(U), dot(inv(F(U_M)), U - dotU_M)))*dxF
    A_FM += -inner(v_F,rho*J(U_M)*dot((dot(grad(U), dot(inv(F(U_M)), \
             dot(grad(dU_M), inv(F(U_M)))))), U - dotU_M ))*dxF
    A_FM += -inner(v_F, rho*J(U_M)*dot(grad(U), dot(inv(F(U_M)),dotdU_M)))*dxF

    #SigmaF
    A_FM += inner(grad(v_F),dU_MSigmaF(U_M,dU_M,U,P,mu))*dxF

    #Div U_F (incompressibility)
    A_FM +=  inner(q, div(J(U_M)*tr(dot(grad(dU_M), inv(F(U_M))))*dot(inv(F(U_M)), U)))*dxF
    A_FM += -inner(q, div(J(U_M)*dot(dot(inv(F(U_M)), grad(dU_M)), dot(inv(F(U_M)), U))))*dxF

    ##Add the terms for the Do nothing boundary if necessary
    if g_F is None:
         #Derivative of do nothing tensor with J factored out
        dSigma  =  tr(grad(dU_M)*inv(F(U_M)))*(mu*inv(F(U_M)).T*grad(U).T - P*I)*inv(F(U_M)).T
        dSigma += -mu*inv(F(U_M)).T*grad(dU_M).T*inv(F(U_M)).T*grad(U).T*inv(F(U_M)).T
        dSigma += -(mu*inv(F(U_M)).T*grad(U).T - P*I)*inv(F(U_M)).T*grad(dU_M).T*inv(F(U_M)).T

        #Add the J                           
        dSigma = J(U_M)*dSigma        
        A_FM += -inner(v_F,dot(dSigma,N_F))*ds_F

    #If a fluid body force has been specified, it will end up here. 
    if F_F is not None:
        A_FM += -inner(v_F,J(U_M)*tr(dot(grad(dU_M),inv(F(U_M))))*F_F)*dxF
    return A_FM

def J_BlockSS(dotdU_S, dotdP_S, dU_S, dP_S, U_S, P_S, v_S,dotv_S, q_S, dotq_S, mu_S, lmbda_S, rho_S, dxS): 
    "Structure diagonal block"
    F_S = grad(U_S) + I                 #I + grad U_s
    E_S = 0.5*(F_S.T*F_S - I)           #Es in the book
    dE_S = 0.5*(grad(dU_S).T*F_S + F_S.T*grad(dU_S))#Derivative of Es wrt to US in the book
    dUsSigma_S = grad(dU_S)*(2*mu_S*E_S + lmbda_S*tr(E_S)*I) + F_S*(2*mu_S*dE_S + lmbda_S*tr(dE_S)*I)

    J_SS = inner(dotv_S, rho_S*dotdP_S)*dxS + inner(grad(v_S), dUsSigma_S)*dxS \
           + inner(dotq_S, dotdU_S - dP_S)*dxS   
    return J_SS

def J_BlockSFbound(dU_F,dP_F,U_M,v_S,mu_F,N_F,dFSI,innerbound):
    "Structure fluid coupling"
    Sigma_F = PiolaTransform(_Sigma_F(dU_F, dP_F, U_M, mu_F), U_M)
    if innerbound == False:
        A_SF = -(inner(dot(Sigma_F,N_F),v_S))*dFSI
    else:
        A_SF = -(inner(dot(Sigma_F('+'),N_F('-')),v_S('-')))*dFSI
    return A_SF

def J_blockSMbound(U_M,dU_M,U_F,P_F,mu_F,v_S,N_S,dFSI,innerbound):
    """Structure mesh coupling"""
    if innerbound == False:
        A_SM = -inner(v_S,dot(dU_MSigmaF(U_M,dU_M,U_F,P_F,mu_F),N_S))*dFSI
    else:
        A_SM = -inner(v_S('-'),dot(dU_MSigmaF(U_M,dU_M,U_F,P_F,mu_F)('+'),N_S('-')))*dFSI
    return A_SM

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
