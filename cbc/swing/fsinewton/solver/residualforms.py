"This module specifies the Residual forms for the monolithic FSI problem."

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

##u_F Fluid Velocity
##p_F Fluid Pressure
##l_F Fluid lagrange multiplier that enforces kinematic continuity of fluid and structure

##u_S Structure displacement
##p_S Structure Velocity

##u_M Mesh Displacement
##l_M Mesh lagrange multiplier that enforces displacement matching with structure on FSI boundary

##Test functions are related to their trial functions by the following letter substitution.
## u-> v , p-> q, l-> m

def fsi_residual(U1list,Umidlist,Udotlist,Vlist,matparams,measures,forces,normals,solver_params):
    """"
    Build the residual forms for the full FSI problem
    including the fluid, structure and mesh equations

    U1list   - List of current fsi variables
             - u1_F,p1_F,l1_F,u1_S,p1_S,u1_M,l1_M

    Umidlist - List of time approximated fsi variables.
             - u_Fmid,p_Fmid,l_Fmid,u_Smid,p_Smid,u_Mmid,l_Mmid

    V        - List of Test functions
             - v_F,q_F,m_F,v_S,q_S,v_M,m_M
               
    matparams - Dictionary of material parameters
              - mu_F,rho_F,mu_S,lmbda_S,rho_S

    measures  - Dictionary of measures
              - dxF,dxS,dxM,dsF,dsS,dFSI
               (dx = interior, ds = exterior boundary, dFSI = FSI interface)

    forces - Dictionary of body and boundary forces
           - F_F,F_S,F_M,G_S,g_F
             (F = body force, G_S = extra FSI traction on structure, g_F = fluid boundary force)

    normals - Dictionary of outer normals
            - N_F, N_S       
    """
    info_blue("Creating residual forms")

    #Unpack the functions
    U1_F,P1_F,L1_U,D1_S,U1_S,D1_F,L1_D = U1list

    #Test Functions
    v_F,q_F,m_U,c_S,v_S,c_F,m_D = Vlist

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
    dsDN = measures["dsDN"] #do nothing fluid boundary

    #Unpack forces
    F_F = forces["F_F"]
    F_S = forces["F_S"]
    F_M = forces["F_M"]
    G_S = forces["G_S"]
    G_F = forces["G_F"]
    G_F_FSI = forces["G_F_FSI"]

    #Unpack Normals
    N_F = normals["N_F"]
    N_S = normals["N_S"]


    #Unpack the time approximations
    U_Fmid,P_Fmid,L_Umid,D_Smid,U_Smid,D_Fmid,L_Dmid = Umidlist
    U_Fdot,P_Fdot,L_Udot,D_Sdot,U_Sdot,D_Fdot,L_Ddot = Udotlist

    #Fluid Residual
    r_F = fluid_residual(U_Fdot,U_Fmid,U1_F,P1_F,v_F,q_F,mu_F,rho_F,D1_F,N_F,dxF,dsDN,dsF,F_F,D_Fdot,G_F)

    #Structure Residual
    r_S = struc_residual(D_Sdot,U_Sdot,D_Smid,U_Smid,c_S,v_S,mu_S,lmbda_S,rho_S,dxS,dsS,F_S)

    #Fluid Domain Residual
    r_FD = fluid_domain_residual(D_Fdot,D_Fmid,c_F,mu_M,lmbda_M,dxM,F_M)

    #Interface residual
    r_FSI = interface_residual(U1_F,U_Fmid,P_Fmid,D1_S,U1_S,D1_F,D_Fmid,L1_U,L1_D,v_F,c_S,
                                c_F,m_D,m_U,mu_F,N_F,dFSI,Exact_SigmaF = G_F_FSI,G_S = G_S)
    #Define full FSI residual
    r = r_F + r_S + r_FD + r_FSI

    #Store the partial residuals in a dictionary
    blockresiduals = {"r_F":r_F,"r_S":r_S,"r_FD":r_FD,"r_FSI":r_FSI}

    #return the full residual and partial residuals (for testing)
    return r,blockresiduals

def fluid_residual(Udot,U,U1_F,P,v,q,mu,rho,U_M,N,dx_F,ds_DN,ds_F,F_F,Udot_M, G_F=None):
    #ALE term present here
    Dt_U = rho*J(U_M)*(Udot + dot(grad(U),dot(inv(F(U_M)),U - Udot_M)))
        
    Sigma_F = PiolaTransform(_Sigma_F(U, P, U_M, mu), U_M)

    #DT
    R_F  = inner(v, Dt_U)*dx_F                                                                      

    #Div Sigma F
    R_F += inner(grad(v), Sigma_F)*dx_F

    #Incompressibility
    R_F += inner(q, div(J(U_M)*dot(inv(F(U_M)), U)))*dx_F                                           

    #Use do nothing BC if specified
    if ds_DN is not None:
        info("Using Do nothing Fluid BC")
        R_F += -inner(v, J(U_M)*dot((mu*inv(F(U_M)).T*grad(U).T - P*I)*inv(F(U_M)).T, N))*ds_DN
        
    #Add boundary traction (sigma dot n) to fluid boundary if specified.
    if ds_F is not None and ds_F != []:
        info("Using Fluid boundary Traction (Neumann) BC")
        R_F += - inner(G_F, v)*ds_F
        
    #Right hand side Fluid (body force)
    if F_F is not None and F_F != []:
        info("Using Fluid body force")
        R_F += -inner(v,J(U_M)*F_F)*dx_F
    return R_F

def fluid_fsibound(P_S,U_F,L_F,v_F,v_S,m_F,dFSI,innerbound):
    if innerbound == False:
        #Kinematic continuity of structure and fluid on the interface
        C_F  = inner(m_F,U_F - P_S)*dFSI
        #Lagrange Multiplier term
        C_F += inner(v_F,L_F)*dFSI
    else:
        #Kinematic continuity of structure and fluid on the interface
        C_F  = inner(m_F,U_F - P_S)('+')*dFSI
        #Lagrange Multiplier term
        C_F += inner(v_F,L_F)('+')*dFSI
    return C_F

def fluid_fsibound2(U_S,v_F,N_S,mu_S,lmbda_S,G_S,dFSI, innerbound,Exact_SigmaF = None):
    """Structure stress on fluid"""
    #Current Structure tensor
    Sigma_S = _Sigma_S(U_S, mu_S, lmbda_S)

    if innerbound == False:
        #Structure Traction on Fluid
        C_S = -(inner(dot(Sigma_S,N_S),v_F))*dFSI
    else:
        if Exact_SigmaF is None:
            #Structure Traction on Fluid
            C_S = -(inner(dot(Sigma_S('-'),N_S('+')),v_F('+')))*dFSI
            if G_S is not None:
                info("Using additional fsi boundary traction term")
                C_S += -inner(G_S('-'),v_F('-'))*dFSI
        else:
            #Prescribed fluid traction on structure
            info("Using perscribed Structure Stress on fsi boundary")
            C_S = (inner(Exact_SigmaF('+'),v_S('-')))*dFSI
    return C_S

def struc_residual(Udot_S,Pdot_S,U_S, P_S,v_S,q_S,mu_S,lmbda_S,rho_S,dx_S,ds_S,F_S):
                    
    Sigma_S = _Sigma_S(U_S, mu_S, lmbda_S)
    #Hyperelasticity equations St. Venant Kirchoff
    R_S = inner(v_S, rho_S*Pdot_S)*dx_S + inner(grad(v_S), Sigma_S)*dx_S + inner(q_S, Udot_S - P_S)*dx_S
    #Right hand side Structure (Body force)
    if F_S is not None and F_S != []:
        info("Using structure body force")
        R_S += -inner(v_S,J(U_S)*F_S)*dx_S
    return R_S

def struc_fsibound(U_F,P_F,U_M,mu_F,v_S,N_F,G_S,dFSI, innerbound, Exact_SigmaF = None):
    #Current Fluid tensor
    Sigma_F = PiolaTransform(_Sigma_F(U_F, P_F, U_M, mu_F), U_M)

    if innerbound == False:
        #Fluid Traction on structure
        C_S = -(inner(dot(Sigma_F,N_F),v_S))*dFSI
        #Optional boundary traction term
        if G_S is not None:
            C_S += -inner(G_S,v_S)*dFSI
    else:
        if Exact_SigmaF is None:
            #Calculated fluid traction on structure
            C_S = -(inner(dot(Sigma_F('+'),N_F('-')),v_S('-')))*dFSI
        else:
            #Prescribed fluid traction on structure
            info("Using perscribed Fluid Stress on fsi boundary")
            C_S = (inner(Exact_SigmaF('+'),v_S('-')))*dFSI
            
        #Optional boundary traction term
        if G_S is not None and G_S != []:
            info("Using additional fsi boundary traction term")
            C_S += inner(G_S('-'),v_S('-'))*dFSI
    return C_S

def struc_fsibound2(P_S,U_F,L_F,v_F,v_S,m_F,dFSI,innerbound):
    if innerbound == False:
        #Kinematic continuity of structure and fluid on the interface
        C_F  = inner(m_F,U_F - P_S)*dFSI
        #Lagrange Multiplier term
        C_F += inner(v_S,L_F)*dFSI
    else:
        #Kinematic continuity of structure and fluid on the interface
        C_F  = inner(m_F,P_S - U_F)('-')*dFSI
        #Lagrange Multiplier term
        C_F += inner(v_S,L_F)('-')*dFSI
    return C_F

def fluid_domain_residual(Udot_M,U_M,v_M,mu_M,lmbda_M,dx_F,F_M):
    #Mesh stress tensor
    Sigma_M = _Sigma_M(U_M, mu_M, lmbda_M)

    #Mesh equation
    R_M = inner(v_M, Udot_M)*dx_F + inner(sym(grad(v_M)), Sigma_M)*dx_F
    #Right hand side mesh (Body Force)
    if F_M is not None and F_M != []:
        info("Using mesh body force")
        R_M += -inner(v_M,F_M)*dx_F
    return R_M

def mesh_fsibound(U_S,U_M,L_M,v_M,m_M,d_FSI,innerbound):
    if innerbound == True:
        #Mesh should follow the structure displacement
        C_MS =  inner(m_M, U_M - U_S)('+')*d_FSI
        #Lagrange Multiplier
        C_MS += inner(v_M, L_M)('+')*d_FSI
    else:
        #Mesh should follow the structure displacement
        C_MS =  inner(m_M, U_M - U_S)*d_FSI
        #Lagrange Multiplier
        C_MS += inner(v_M, L_M)*d_FSI 
    return C_MS

def interface_residual(U_F,U_Fmid,P_Fmid,U_S,P_S,U_M,U_Mmid,L_F,L_M,v_F,v_S,
                       v_M,m_M,m_F,mu_F,N_F,dFSI,Exact_SigmaF,G_S):
    """Residual for interface conditions on the FSI interface"""
    #Displacement Lagrange Multiplier
    R_FSI =  inner(m_M, U_M - U_S)('+')*dFSI
    R_FSI += inner(v_M, L_M)('+')*dFSI

    #Velocity Lagrange Multiplier
    R_FSI += inner(m_F,U_F - P_S)('+')*dFSI
    R_FSI += inner(v_F,L_F)('+')*dFSI

    #Stress Continuity
    if Exact_SigmaF is None:
        #Calculated fluid traction on structure
        Sigma_F = PiolaTransform(_Sigma_F(U_Fmid, P_Fmid, U_Mmid, mu_F), U_Mmid)
        R_FSI += -(inner(dot(Sigma_F('+'),N_F('-')),v_S('-')))*dFSI
    else:
        #Prescribed fluid traction on structure
        info("Using perscribed Fluid Stress on fsi boundary")
        R_FSI += (inner(Exact_SigmaF('+'),v_S('-')))*dFSI
        
    #Optional boundary traction term
    if G_S is not None and G_S != []:
        info("Using additional fsi boundary traction term")
        R_FSI += inner(G_S('-'),v_S('-'))*dFSI
    return R_FSI
