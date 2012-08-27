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
from cbc.swing.operators import F, J
from cbc.twist.kinematics import SecondOrderIdentity as I


##Throughout this module the following notation is used.

##U_F Fluid Velocity
##P_F Fluid Pressure
##L_U Fluid Lagrange multiplier that enforces kinematic continuity of fluid and structure

##D_S Structure displacement
##U_S Structure Velocity

##D_F Fluid Domain (Mesh) Displacement
##L_D Fluid Domain (Mesh) lagrange multiplier that enforces displacement matching with structure on FSI boundary

##Test functions are related to their trial functions by the following letter substitution.
## u-> v , p-> q, l-> m , d -> c

def sum(somelist):
    res = somelist[0]
    for elem in somelist[1:]:
        res += elem
    return res
        
def fsi_residual(U1list,Umidlist,Udotlist,Vlist,matparams,measures,forces,normals,solver_params):
    """"
    Build the residual forms for the full FSI problem
    including the fluid, structure and mesh equations

    U1list   - List of current fsi variables
             - U1_F,P1_F,L1_U,D1_S,U1_S,D1_F,L1_D

    Umidlist - List of time approximated fsi variables, mid here is from the point of view of cG(1)
             - U_Fmid,P_Fmid,L_Umid,D_Smid,U_Smid,D_Fmid,L_Dmid

    V        - List of Test functions
             - v_F,q_F,m_U,c_S,v_S,c_F,m_D
               
    matparams - Dictionary of material parameters
              - mu_F,rho_F,mu_S,lmbda_S,rho_S

    measures  - Dictionary of measures

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
    #The user interface referes here to "mesh-M", wheras interally
    #these parameters refer to "fluid domain-FD"
    mu_FD = matparams["mu_M"]
    lmbda_FD = matparams["lmbda_M"]

    #Unpack lists of Measures
    dxF = measures["fluid"]
    dxS = measures["structure"]
    dsF = measures["fluidneumannbound"]
    dsS = measures["strucbound"]
    dFSI = measures["FSI_bound"]
    dsDN = measures["donothingbound"]
    
    #Unpack forces
    F_F = forces["F_F"]
    F_S = forces["F_S"]
    #Here "Mesh" is internally "fluid domain"
    F_FD = forces["F_M"]
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
    if solver_params["fluid_domain_time_discretization"] == "end-point": D_F = D1_F
    elif solver_params["fluid_domain_time_discretization"] =="mid-point":D_F = D_Fmid
    else: raise Exception("Only mid-point and end-point are possible \
                          fluid_domain_time_discretization parameter values \
                          current value is %s"%solver_params["fluid_domain_time_discretization"])
    r_F = fluid_residual(U_Fdot,U_Fmid,U1_F,P1_F,v_F,q_F,mu_F,rho_F,D_F,N_F,dxF,dsDN,dsF,F_F,D_Fdot,G_F)
    
    #Structure Residual
    r_S = struc_residual(D_Sdot,U_Sdot,D_Smid,U_Smid,c_S,v_S,mu_S,lmbda_S,rho_S,dxS,dsS,F_S)

    #Fluid Domain Residual
    r_FD = fluid_domain_residual(D_Fdot,D_Fmid,c_F,mu_FD,lmbda_FD,dxF,F_FD)

    #Interface residual
    r_FSI = interface_residual(U1_F,U_Fmid,P_Fmid,D1_S,U1_S,D1_F,D_Fmid,L1_U,L1_D,v_F,c_S,
                                c_F,m_D,m_U,mu_F,N_F,dFSI,Exact_SigmaF = G_F_FSI,G_S = G_S)
    
    #Define full FSI residual
    r = r_F + r_S + r_FD + r_FSI

    #Store the partial residuals in a dictionary
    blockresiduals = {"r_F":r_F,"r_S":r_S,"r_FD":r_FD,"r_FSI":r_FSI}

    #return the full residual and partial residuals (for testing)
    return r,blockresiduals

def fluid_residual(U_Fdot,U_F,U1_F,P_F,v_F,q_F,mu,rho,D_F,N_F,dx_Flist,ds_DNlist,ds_Flist,
                   F_F,D_Fdot, G_F=None):
    #Fluid equation in the Domain
    forms = []
    for dx_F in dx_Flist:
        Dt_U = rho*J(D_F)*(U_Fdot + dot(grad(U_F),dot(inv(F(D_F)),U_F - D_Fdot)))
            
        Sigma_F = PiolaTransform(_Sigma_F(U_F, P_F, D_F, mu), D_F)

        #DT
        R_F  = inner(v_F, Dt_U)*dx_F                                                                      

        #Div Sigma F
        R_F += inner(grad(v_F), Sigma_F)*dx_F

        #Incompressibility
        R_F += inner(q_F, div(J(D_F)*dot(inv(F(D_F)), U_F)))*dx_F

        #Right hand side Fluid (body force)
        if F_F is not None and F_F != []:
            info("Using Fluid body force")
            R_F += -inner(v_F,J(D_F)*F_F)*dx_F
        forms.append(R_F)

    #Use do nothing BC if specified
    for ds_DN in ds_DNlist:
        info("Using Do nothing Fluid BC")
        DN_F = -inner(v_F, J(D_F)*dot((mu*inv(F(D_F)).T*grad(U_F).T - P_F*I(U_F))*inv(F(D_F)).T, N_F))*ds_DN
        forms.append(DN_F)
                       
    #Add boundary traction (sigma dot n) to fluid boundary if specified.
    for ds_F in ds_Flist:
        info("Using Fluid boundary Traction (Neumann) BC")
        B_F =  -inner(G_F, v_F)*ds_F
        forms.append(B_F)
    return sum(forms)

def struc_residual(Ddot_S,Udot_S,D_S, U_S,c_S,v_S,mu_S,lmbda_S,rho_S,dx_Slist,ds_Slist,F_S):                  
    Sigma_S = _Sigma_S(D_S, mu_S, lmbda_S)
    forms = []
    for dx_S in dx_Slist:
        #Hyperelasticity equations St. Venant Kirchoff
        R_S = inner(c_S, rho_S*Udot_S)*dx_S + inner(grad(c_S), Sigma_S)*dx_S + inner(v_S, Ddot_S -U_S)*dx_S
        #Right hand side Structure (Body force)
        if F_S is not None and F_S != []:
            info("Using structure body force")
            R_S += -inner(c_S,F_S)*dx_S
        forms.append(R_S)
        
    #No Neumann terms at the moment
    return sum(forms)

def fluid_domain_residual(Ddot_F,D_F,c_F,mu_FD,lmbda_FD,dx_Flist,F_FD):
    #Fluid Domain (Mesh) stress tensor
    Sigma_FD = _Sigma_M(D_F, mu_FD, lmbda_FD)
    forms = []
    
    for dx_F in dx_Flist:
        #Fluid Domain equation
        R_FD = inner(c_F, Ddot_F)*dx_F + inner(sym(grad(c_F)), Sigma_FD)*dx_F
        #Right hand side mesh (Body Force)
        if F_FD is not None and F_FD != []:
            info("Using fluid domain body force")
            R_FD += -inner(c_F,F_FD)*dx_F
        forms.append(R_FD)
    return sum(forms)

def interface_residual(U_F,U_Fmid,P_Fmid,D_S,U_S,D_F,D_Fmid,L_U,L_D,v_F,c_S,
                       c_F,m_D,m_U,mu_F,N_F,dFSIlist,Exact_SigmaF,G_S):
    """Residual for interface conditions on the FSI interface"""
    forms = []
    for dFSI in dFSIlist:
        #Displacement Lagrange Multiplier
        R_FSI =  inner(m_D, D_F - D_S)('+')*dFSI
        R_FSI += inner(c_F, L_D)('+')*dFSI

        #Velocity Lagrange Multiplier
        R_FSI += inner(m_U,U_F - U_S)('+')*dFSI
        R_FSI += inner(v_F,L_U)('+')*dFSI

        #Stress Continuity
        if Exact_SigmaF is None or Exact_SigmaF == []:
            #Calculated fluid traction on structure
            Sigma_F = PiolaTransform(_Sigma_F(U_Fmid, P_Fmid, D_Fmid, mu_F), D_Fmid)
            R_FSI += -(inner(dot(Sigma_F('+'),N_F('-')),c_S('-')))*dFSI
        else:
            #Prescribed fluid traction on structure
            info("Using perscribed Fluid Stress on fsi boundary")
            R_FSI += (inner(Exact_SigmaF('+'),c_S('-')))*dFSI
            
        #Optional boundary traction term
        if G_S is not None and G_S != []:
            info("Using additional FSI boundary traction term")
            R_FSI += inner(G_S('-'),c_S('-'))*dFSI
        forms.append(R_FSI)
    return sum(forms)
