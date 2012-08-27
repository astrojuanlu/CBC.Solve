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
from cbc.swing.operators import F, J
from cbc.twist.kinematics import SecondOrderIdentity as I
from residualforms import sum

##Throughout this module the following notation is used.

##U_F Fluid Velocity
##P_F Fluid Pressure
##L_U Fluid lagrange multiplier that enforces kinematic continuity of fluid and structure

##D_S Structure displacement
##U_S Structure Velocity

##D_F Fluid Domain Displacement
##L_D Fluid Domain lagrange multiplier that enforces displacement matching with structure on FSI boundary

##Test functions are related to their trial functions by the following letter substitution.
## u-> v , p-> q, l-> m, d -> c

def fsi_jacobian(Iulist,Iudotlist,Iumidlist,U1list,Umidlist,Udotlist,Vlist,
                 dotVlist,matparams,measures,forces,normals,params):
    """"
    Build the jacobian forms for the full FSI problem
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
               (dx = interior, ds = exterior boundary, dFSI = FSI interface)

    normals - Dictionary of outer normals
            - N_F, N_S           
    """
    info_blue("Creating Jacobian Forms")

    #Unpack Functions
    U1_F,P1_F,L1_U,D1_S,U1_S,D1_F,L1_D = U1list

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
    F_FD = forces["F_M"]
    G_S = forces["G_S"]
    G_F = forces["G_F"]

    #Unpack the time approximated functions 
    IU_Fmid,IP_Fmid,IL_Umid,ID_Smid,IU_Smid,ID_Fmid,IL_Dmid = Iumidlist
    IU_Fdot,IP_Fdot,IL_Udot,ID_Sdot,IU_Sdot,ID_Fdot,IL_Ddot = Iudotlist
    U_Fmid,P_Fmid,L_Umid,D_Smid,U_Smid,D_Fmid,L_Dmid = Umidlist
    U_Fdot,P_Fdot,L_Udot,D_Sdot,U_Sdot,D_Fdot,L_Ddot = Udotlist    

    #Fluid Residual
    if params["fluid_domain_time_discretization"] == "end-point":
        D_Fstar = D1_F
        ID_Fstar = ID_F
    elif params["fluid_domain_time_discretization"] == "mid-point":
        D_Fstar = D_Fmid
        ID_Fstar = ID_Fmid
    else: raise Exception("Only mid-point and end-point are possible \
                          fluid_domain_time_discretization parameter values \
                          current value is %s"%solver_params["fluid_domain_time_discretization"])

    #Diagonal Blocks
    j = J_BlockF(IU_Fdot,IU_Fmid,IP_F,U_Fmid,D_Fdot,v_F,dotv_F,q_F,D_Fstar,
                   rho_F,mu_F,N_F,dxF,dsDN,dsF,G_F)
    
    j += J_BlockS(ID_Smid,D_Smid,c_S,mu_S,lmbda_S,rho_S,dxS)
    #Bufferable constant terms, including Fluid Domain Block
    j_buff = J_Bufferable(IU_F,ID_Sdot,ID_S, IU_Sdot,IU_S,IU_Smid,ID_Fdot,ID_F,
                          ID_Fmid,IL_U,IL_D,v_F,dotv_S,dotc_S,dotc_F,c_F,m_D,m_U,
                          rho_S,mu_FD,lmbda_FD,dxF,dFSI,dxS)
    
    #Interface Block
    j += J_FSI(IU_F,IU_Fmid,IP_Fmid,ID_Fmid,U_Fmid,P_Fmid,D_Fmid,c_S,mu_F,N_F,dFSI)

    #Fluid-Fluid Domain Block
    if params["optimization"]["simplify_jacobian"] == True:
        simplify = dFSI
        info("Using Simplified Jacobian")
    else:simplify = None
    j += J_BlockFFD(U_Fmid, U_Fdot,P1_F, D_Fstar, ID_Fstar, D_Fdot, ID_Fdot,
                           v_F, dotv_F, q_F, rho_F, mu_F,N_F, dxF,dsDN,simplify,dsF, G_F,F_F)
    return j,j_buff

def dD_FSigmaF(D_F,dD_F,U_F,P_F,mu_F):
    """Derivative of Sigma_F with respect to D_F"""
    ret =   J(D_F)*tr(dot(grad(dD_F), inv(F(D_F))))*dot(Sigma_F(U_F, P_F, D_F, mu_F), inv(F(D_F)).T)
    ret += - J(D_F)*dot(mu_F*(dot(grad(U_F), dot(inv(F(D_F)), dot(grad(dD_F), inv(F(D_F)))))), inv(F(D_F)).T)
    ret += - J(D_F)*dot(mu_F*(dot(inv(F(D_F)).T, dot(grad(dD_F).T, dot(inv(F(D_F)).T, grad(U_F).T )))), inv(F(D_F)).T)
    ret += - J(D_F)*dot(dot(Sigma_F(U_F, P_F, D_F, mu_F), inv(F(D_F)).T), dot(grad(dD_F).T, inv(F(D_F)).T))
    return ret

def J_BlockF(dotdU,dU,dP,U,dotD_F,v,dotv,q,D_F,rho,mu,N_F,dxFlist,dsDNlist,dsF,g_F=None):
    """Fluid Diagonal Block, Fluid Domain """
    Id = I(U)
    forms = []
    for dxF in dxFlist:
        #DT (without ALE term)
        J_FF =  inner(dotv, rho*J(D_F)*dotdU)*dxF                              
        J_FF +=  inner(v, rho*J(D_F)*dot(dot(grad(dU), inv(F(D_F))), U - dotD_F))*dxF  
        J_FF +=  inner(v, rho*J(D_F)*dot(grad(U), dot(inv(F(D_F)), dU)))*dxF

        #div Sigma_F
        J_FF +=  inner(grad(v), J(D_F)*mu*dot(grad(dU), dot(inv(F(D_F)), inv(F(D_F)).T)))*dxF     
        J_FF +=  inner(grad(v), J(D_F)*mu*dot(inv(F(D_F)).T, dot(grad(dU).T, inv(F(D_F)).T)))*dxF 
        J_FF += -inner(grad(v), J(D_F)*dP*inv(F(D_F)).T)*dxF                                       

        #div U_F (incompressibility)
        J_FF +=  inner(q, div(J(D_F)*dot(inv(F(D_F)), dU)))*dxF
        forms.append(J_FF)

    #Do nothing BC
    for dsDN in dsDNlist:
        DN_FF =  -inner(v, dot(J(D_F)*mu*dot(inv(F(D_F)).T, dot(grad(dU).T, inv(F(D_F)).T)), N_F))*dsDN
        DN_FF += inner(v, J(D_F)*dP*dot(Id, dot(inv(F(D_F)).T, N_F)))*dsDN
        forms.append(DN_FF)
    return sum(forms)

def J_BlockFFD(U, dotU, P, D_F, dD_F,dotD_F, dotdD_F, v_F,dotv, q, rho, mu,N_F,
               dxFlist,dsDNlist,dFSIlist,ds_F,g_F = None,F_F = None):
    """Fluid-Fluid Domain coupling"""
    Id = I(D_F)
    forms = []
    for dxF in dxFlist:
        #If a fluid body force has been specified, it will end up here. 
        if F_F is not None and F_F != []:
            B_F = -inner(v_F,J(D_F)*tr(dot(grad(dD_F),inv(F(D_F))))*F_F)*dxF
            forms.append(B_F)
    #Simplify the remaining terms if an FSI interface measure provided    
    if dFSIlist is not None:
        dxFlist = dFSIlist
        
    for dxF in dxFlist:
        #DT  
        J_FM =  inner(v_F, rho*J(D_F)*tr(dot(grad(dD_F), inv(F(D_F))))*dotU)
        J_FM +=  inner(v_F, rho*J(D_F)*tr(dot(grad(dD_F), inv(F(D_F))))*dot(grad(U), dot(inv(F(D_F)), U - dotD_F)))
        J_FM += -inner(v_F,rho*J(D_F)*dot((dot(grad(U), dot(inv(F(D_F)), \
                 dot(grad(dD_F), inv(F(D_F)))))), U - dotD_F ))
        J_FM += -inner(v_F, rho*J(D_F)*dot(grad(U), dot(inv(F(D_F)),dotdD_F)))

        #SigmaF
        J_FM += inner(grad(v_F),dD_FSigmaF(D_F,dD_F,U,P,mu))

        #Div U_F (incompressibility)
        J_FM +=  inner(q, div(J(D_F)*tr(dot(grad(dD_F), inv(F(D_F))))*dot(inv(F(D_F)), U)))
        J_FM += -inner(q, div(J(D_F)*dot(dot(inv(F(D_F)), grad(dD_F)), dot(inv(F(D_F)), U))))
        if dFSIlist is None:
            J_FM = J_FM*dxF
        else:
            J_FM = J_FM('+')*dxF
        forms.append(J_FM)
    return sum(forms)

    ##Add the terms for the Do nothing boundary if necessary
    for dsDN in dsDNlist:
        #Derivative of do nothing tensor with J factored out
        dSigma  =  tr(grad(dD_F)*inv(F(D_F)))*(mu*inv(F(D_F)).T*grad(U).T - P*Id)*inv(F(D_F)).T
        dSigma += -mu*inv(F(D_F)).T*grad(dD_F).T*inv(F(D_F)).T*grad(U).T*inv(F(D_F)).T
        dSigma += -(mu*inv(F(D_F)).T*grad(U).T - P*Id)*inv(F(D_F)).T*grad(dD_F).T*inv(F(D_F)).T
        #J added
        dSigma = J(D_F)*dSigma
        DN_FM = -inner(v_F,dot(dSigma,N_F))*dsDN
        forms.append(DN_FM)
    return sum(forms)

def J_BlockS(dD_S, D_S, c_S, mu_S, lmbda_S, rho_S, dxSlist): 
    "Structure diagonal block"
    Id = I(D_S)
    F_S = grad(D_S) + I(D_S)                 #I + grad U_s
    E_S = 0.5*(F_S.T*F_S - Id)           #Es in the book
    dE_S = 0.5*(grad(dD_S).T*F_S + F_S.T*grad(dD_S))#Derivative of Es wrt to US in the book
    dUsSigma_S = grad(dD_S)*(2*mu_S*E_S + lmbda_S*tr(E_S)*Id) + F_S*(2*mu_S*dE_S + lmbda_S*tr(dE_S)*Id)

    forms = []
    for dxS in dxSlist:
        forms.append(inner(grad(c_S), dUsSigma_S)*dxS)
    return sum(forms)

def J_FSI(dU_F,dU_Fmid,dP_Fmid,dD_Fmid,U_Fmid,P_Fmid,D_Fmid,c_S,mu_F,N_F,dFSIlist):
    """Derivative of the Interface Residual"""
    forms = []
    Sigma_F = PiolaTransform(_Sigma_F(dU_Fmid, dP_Fmid, D_Fmid, mu_F), D_Fmid)
    for dFSI in dFSIlist:   
        J_FSI = -(inner(c_S('-'),dot(Sigma_F('+'),N_F('-'))))*dFSI       
        J_FSI += -inner(c_S('-'),dot(dD_FSigmaF(D_Fmid,dD_Fmid,U_Fmid,P_Fmid,mu_F)('+'),N_F('-')))*dFSI
        forms.append(J_FSI)
    return sum(forms)

def J_Bufferable(dU_F,dotdD_S,dD_S, dotdU_S,dU_S,dU_Smid,dD_Fdot,dD_F,dD_Fmid,dL_U,dL_D,v_F,dotv_S,
                 dotc_S,dotc_F,c_F,m_D,m_U,rho_S,mu_FD,lmbda_FD,dx_Flist,dFSIlist,dxSlist):
    """Linear forms that only need to be assembled once"""
    #Fluid Domain diagonal block
    forms = []
    Sigma_FD = _Sigma_M(dD_Fmid, mu_FD, lmbda_FD)
    for dx_F in dx_Flist:
        J_FD = inner(dotc_F, dD_Fdot)*dx_F + inner(sym(grad(c_F)), Sigma_FD)*dx_F
        forms.append(J_FD)
    #Lagrange Multiplier conditions
    for dFSI in dFSIlist:
        J_FSI = inner(m_D, dD_F - dD_S)('+')*dFSI 
        J_FSI += inner(c_F, dL_D)('+')*dFSI #Lagrange Multiplier
        J_FSI += inner(m_U, dU_F - dU_S)('+')*dFSI
        J_FSI += inner(v_F, dL_U)('+')*dFSI #Lagrange Multiplier
        forms.append(J_FSI)
    #Structure Dt terms
    for dxS in dxSlist:
        J_S = inner(dotc_S, rho_S*dotdU_S)*dxS + inner(dotv_S, dotdD_S - dU_Smid)*dxS
        forms.append(J_S)
    return sum(forms)
