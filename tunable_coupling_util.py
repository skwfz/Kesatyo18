# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:53:36 2018

@author: rissans2
"""

from qutip import *
import numpy as np
from matplotlib import pyplot as pl
import scipy as sc
import util

class tunable_coupling_simulation:
    def __init__(self,levels,C,Cc,C12,omega):
        self.electron_charge = 1.60217662e-19
        self.hbar = 1.0545718e-34
        
        self.levels = levels
        self.bs = [tensor(destroy(levels),qeye(levels),qeye(levels)),#Qubit 1
                        tensor(qeye(levels),destroy(levels),qeye(levels)),#Qubit c
                        tensor(qeye(levels),qeye(levels),destroy(levels))]#Qubit 2
        self.C = C
        self.Cc = Cc
        self.C12 = C12
        self.eta = Cc[0]*Cc[1]/(C12*C[1])
        self.EC = self.electron_charge**2/(2*C)/self.hbar
        self.alpha = -self.EC
        self.omega = omega
        self.g = 0.5*(Cc/np.sqrt(np.array([C[0],C[2]])*C[1]))*np.sqrt(np.array([omega[0],omega[2]])*omega[1])
        self.g12 = 0.5*(1+self.eta)*C12/(np.sqrt(C[0]*C[2]))*np.sqrt(omega[0]*omega[2])
        #Additional possibly useful terms
        self.sigma = [omega[0]+omega[1],omega[2]+omega[1]]
        self.ops = (1/self.sigma[0]+1/self.sigma[1])/2
        self.delta1 = omega[0]-omega[1]
        self.delta2 = omega[2]-omega[1]
        self.opd = (1/self.delta1+1/self.delta2)/2
        
        self.symm_comp = (tensor(basis(levels,1),basis(levels,0),basis(levels,0))+tensor(basis(levels,0),basis(levels,0),basis(levels,1))).unit()
        self.antisymm_comp = (tensor(basis(levels,1),basis(levels,0),basis(levels,0))-tensor(basis(levels,0),basis(levels,0),basis(levels,1))).unit()
        self.state100_comp = (tensor(basis(levels,1),basis(levels,0),basis(levels,0))).unit()
        self.state001_comp = (tensor(basis(levels,0),basis(levels,0),basis(levels,1))).unit()
        self.state010_comp = (tensor(basis(levels,0),basis(levels,1),basis(levels,0))).unit()
        self.state111_comp = (tensor(basis(levels,1),basis(levels,1),basis(levels,1))).unit()
        self.state000_comp = (tensor(basis(levels,0),basis(levels,0),basis(levels,0))).unit()
        self.state101_comp = (tensor(basis(levels,1),basis(levels,0),basis(levels,1))).unit()
        self.state011_comp = (tensor(basis(levels,0),basis(levels,1),basis(levels,1))).unit()
        self.state110_comp = (tensor(basis(levels,1),basis(levels,1),basis(levels,0))).unit()
        self.state020_comp = (tensor(basis(levels,0),basis(levels,2),basis(levels,0))).unit()
        
        #Fixing the values of omega[1] and g to the decoupled state
        omega0 = sc.optimize.fsolve(self.gef_accurate_fsolve,5e9)[0]
        omega[1] = omega0
        self.g = 0.5*(self.Cc/np.sqrt(np.array([self.C[0],self.C[2]])*self.C[1]))*np.sqrt(np.array([self.omega[0],self.omega[2]])*self.omega[1])

    def gef(self,omegac):
        """The (approximate) effective coupling constant between qubits 1 and 2"""
        delta1 = np.array(self.omega[0]-omegac)
        delta2 = np.array(self.omega[2]-omegac)
        opd = (1/delta1+1/delta2)/2
        sigma = np.array([self.omega[0]+omegac,self.omega[2]+omegac])
        ops = (1/sigma[0]+1/sigma[1])/2
        return (0.5*(omegac*0.5*opd*self.eta-omegac*0.5*ops*self.eta+self.eta+1)*self.C12/np.sqrt((self.C[0]*self.C[2]))*np.sqrt(self.omega[0]*self.omega[2]))
    
    def geftSine(self,t,A,T,usedgef):
        """The expression which integral should equal to -pi/2 if the pulse performs the iSWAP gate."""
        omegac = (1-A*np.sin(np.pi*t/T))*self.omega[1]
        return usedgef(omegac)
    
    def geftSquare(self,t,A,T,usedgef):
        """The expression which integral should equal to -pi/2 if the pulse performs the iSWAP gate"""
        omegac = (1-A)*self.omega[1]
        return usedgef(omegac)
    
    def geff_integral(self,A,T,pulse,usedgef):
        """The integral expression that should equal to 0 if pulse performs the iSWAP gate"""
        return (sc.integrate.quadrature(pulse,0,T,args=(A,T,usedgef)))[0]+np.pi/2
    
    def sch_Hamiltonian(self,omegac):
        """Gives the Hamiltonian matrix in the Schrödinger picture. Can be used for finding eigenstates of the system."""
        omega_temp = [self.omega[0],omegac,self.omega[2]]
        g_temp = 0.5*(self.Cc/np.sqrt(np.array([self.C[0],self.C[2]])*self.C[1]))*np.sqrt(np.array([self.omega[0],self.omega[2]])*omegac)#Depends on coupling qubit frequency
        Hqubits = [o*b.dag()*b+a/2*b.dag()*b.dag()*b*b for (o,a,b) in zip(omega_temp,self.alpha,self.bs)]#Free qubit Hamiltonians
        #Coupling to coupler terms
        Hc = [g_temp*(b.dag()*self.bs[1]+b*self.bs[1].dag())-g_temp*b.dag()*self.bs[1].dag()-g_temp*b*self.bs[1] for (g_temp,b) in zip(g_temp,[self.bs[0],self.bs[2]])]
        #Direct coupling between qubits
        H12 = self.g12*(self.bs[0].dag()*self.bs[2]+self.bs[0]*self.bs[2].dag())
        H_Sch = sum(Hqubits)+sum(Hc)+H12
        return H_Sch
    
    def simulation_Hamiltonian(self,used_pulse):
        """Returns the the Hamiltonian that can be used in the simulation.
        Doesn't have the information on the amplitude and length of used_pulse."""
        logical_eigens = self.eigenstates_logical_states(self.omega[1])
        Eavg = (logical_eigens[0][0]+logical_eigens[1][0])/2#Average not really needed if in the decoupled state where the energy is degenerate
        
        """The Hamiltonian, time independent and dependent parts"""
        #Individual qubit Hamiltonians
        Hs =  [(o-Eavg)*b.dag()*b + a/2*b.dag()*b.dag()*b*b for (o,a,b) in zip(self.omega,self.alpha,self.bs)]
        Hs[1] = Hs[1] - self.omega[1]*self.bs[1].dag()*self.bs[1]#Taking out the time dependent part, see below
        Hs1t = self.omega[1]*self.bs[1].dag()*self.bs[1]#Time-dependent part of coupler harmonic term
        #Coupling to coupler terms
        HcJC = [g*(b.dag()*self.bs[1]+b*self.bs[1].dag()) for (g,b) in zip(self.g,[self.bs[0],self.bs[2]])]#(Jaynes-Cummings)Is modulated by change of omegac
        HcCRdag = -self.g[0]*self.bs[0].dag()*self.bs[1].dag()-self.g[1]*self.bs[2].dag()*self.bs[1].dag()#(Counter-Rotating) Is modulated by change of omegac and rotates in time
        HcCR = -self.g[0]*self.bs[0]*self.bs[1]-self.g[1]*self.bs[2]*self.bs[1]#Is modulated by change of omegac and rotates in time
        #Direct coupling between qubits
        H12 = self.g12*(self.bs[0].dag()*self.bs[2]+self.bs[0]*self.bs[2].dag())
        H0 = sum(Hs) + H12
        
        """The final simulation Hamiltonian"""
        H = [H0,[Hs1t,used_pulse],[HcCR,lambda t,args: np.sqrt(used_pulse(t,args))*np.exp(-2*Eavg*1.0j*t)]\
                ,[HcCRdag, lambda t,args: np.sqrt(used_pulse(t,args))*np.exp(2*Eavg*1.0j*t)],[sum(HcJC), lambda t,args: np.sqrt(used_pulse(t,args))]]
        return H

    def theoretical_pulse_amplitude(self,used_pulse,T,used_gef):
        A = 0
        if used_pulse == sine_pulse:
            A = sc.optimize.fsolve(self.geff_integral,0.189,args=(T,self.geftSine,used_gef))[0]
        elif used_pulse == square_pulse:
#            A = sc.optimize.brentq(self.geff_integral,a=0,b=(self.omega[1]-self.omega[0])/self.omega[1],args=(T,self.geftSquare,used_gef))[0]
            A = sc.optimize.fsolve(self.geff_integral,0.189,args=(T,self.geftSquare,used_gef))[0]
        return A
    
    def eigenstates_logical_states(self,omegac):
        """Finds the eigenstates and energies corresponding to the states |100> and |001> or their symmetric and antisymmetric combinations
        for a given coupling qubit frequency. In the transition period from these two possibilities, not sure."""
        H_Sch = self.sch_Hamiltonian(omegac)
        #Solving the eigenstates and finding the states corresponding to |100> and |001>
        eigstates = H_Sch.eigenstates()
        #Assume that the eigenstates are symmetric and antisymmetric combinations of |100> and |001>s
        (state1, state2) = util.kets_from_best_ket_match(eigstates[1],self.state001_comp,self.state100_comp)
        energy1 = eigstates[0][util.find_best_ket_match_index(eigstates[1],state1)]
        energy2 = eigstates[0][util.find_best_ket_match_index(eigstates[1],state2)]
        return ((energy1,state1), (energy2,state2))
    
    def gef_accurate(self,omegac):
        """The more accurate effective coupling constant between qubits 1 and 2. When used, it is assumed that the 
        eigenstates of the system don't change when changing omegac, just the eigenenergies. Otherwise should be accurate if
        the computational states used are the ones derived from the eigenstates."""
        energystates = self.eigenstates_logical_states(omegac)
        energies = [energystates[0][0], energystates[1][0]]
        return (energies[0]-energies[1])/2
    
    def gef_accurate_fsolve(self,omegac):
        """Helping function for using fsolve"""
        return self.gef_accurate(omegac[0])
    
    def gef_accurate_interpolation(self,filename,newinterpolation = False):
        """Returns an interpolated function made from calculating values of gef_accurate"""
        omegas = np.linspace(2*self.omega[0]-self.omega[1],self.omega[1]*1.5,500)
        if(newinterpolation == False):
            geffs = np.load(filename)
        else:
            geffs = np.array([self.gef_accurate(o) for o in omegas])
        geffint = sc.interpolate.interp1d(omegas,geffs,'cubic')
        np.save(filename,geffs)
        return (omegas,geffint)
    
    def find_swapping_eigenstates(self, omegac):
        """Returns the eigenenergies and eigenstates of the four eigenstates that essentially perform
        the swap, having coefficients mostly from the computational states |001>,|100>,|010>,(|111>).
        Also sorts the states by order."""
        realeigs = []
        eigs = self.sch_Hamiltonian(omegac).eigenstates()
        for e in zip(eigs[0],eigs[1]):
            if abs((self.state001_comp.dag()*e[1])[0][0])**2 + abs((self.state100_comp.dag()*e[1])[0][0])**2\
                    +abs((self.state010_comp.dag()*e[1])[0][0])**2 > 0.9:
                    #+abs((self.state111_comp.dag()*e[1])[0][0])**2
                realeigs.append(e)
        realeigs.sort(key = lambda e: e[0])
        return realeigs
    
    def find_cphase_eigenstates(self,omegac):
        """Returns the eigenenergies and eigenstates of the four eigenstates that essentially cause the 
        parasitic 'cphase' (or that's what they called it in the article) operation to happen. """
        realeigs = []
        eigs = self.sch_Hamiltonian(omegac).eigenstates()
        for e in zip(eigs[0],eigs[1]):
            if abs((self.state011_comp.dag()*e[1])[0][0])**2 + abs((self.state110_comp.dag()*e[1])[0][0])**2\
                    +abs((self.state101_comp.dag()*e[1])[0][0])**2+abs((self.state020_comp.dag()*e[1])[0][0])**2 > 0.9:
                        realeigs.append(e)
        realeigs.sort(key = lambda e: e[0])
        return realeigs
    
    def dynamic_phase_fixer(self,state,eig1,eig2,E1,E2,time):
        """Rotates the eigenstates eig1 and eig2 by E1*time and E2*time in the state "state" and returns
        the new state."""
        return state - (eig1.dag()*state)*eig1 + (eig1.dag()*state)*eig1*np.exp(1.0j*E1*time)\
                - (eig2.dag()*state)*eig2 + (eig2.dag()*state)*eig2*np.exp(1.0j*E2*time)
#        D = self.Dproto + basis(self.levels**3,index1)*basis(self.levels**3,index1).dag()*np.exp(E1*time*1.0j)\
#            +  basis(self.levels**3,index2)*basis(self.levels**3,index2).dag()*np.exp(E2*time*1.0j)
#        D.dims = [[self.levels,self.levels,self.levels],[self.levels,self.levels,self.levels]]
#        return self.Q*D*self.Q.dag()
    def dynamic_phase_fixer_2(self,state,starting_state,eig1,eig2):
        """Rotates 'state':s components along eigenvectors eig1 and eig2 
        to match with the state 'starting_state' phase"""
        eig1proj = (eig1.dag()*state)*eig1
        eig2proj = (eig2.dag()*state)*eig2
        starting_eig1proj = (eig1.dag()*starting_state)*eig1
        starting_eig2proj = (eig2.dag()*starting_state)*eig2
        return state - eig1proj + eig1proj*np.exp(1.0j*util.angle_between_states(eig1proj,starting_eig1proj))\
                - eig2proj + eig2proj*np.exp(1.0j*util.angle_between_states(eig2proj,starting_eig2proj))
    
    def dynamic_phase_fixer_3(self,state,target_state,eig1,eig2,eig3):
        """Finds the best rotation angles for qubit 1 and 2 to reach the target_state from state. Pretty slow,
        as it has to do 2d optimization."""
        eig1proj = (eig1.dag()*state)*eig1
        eig2proj = (eig2.dag()*state)*eig2
        eig3proj = (eig3.dag()*state)*eig3
        def rotate_eigstates(phi1,phi2):
            return state-eig1proj + eig1proj*np.exp(1.0j*phi1) - eig2proj + eig2proj*np.exp(1.0j*phi2)\
                    - eig3proj + eig3proj*np.exp(1.0j*(phi1+phi2))
        def negative_target_closeness(phis):
            return -np.abs((target_state.dag()*rotate_eigstates(phis[0],phis[1]))[0][0][0])
        res = sc.optimize.minimize(negative_target_closeness,np.array([0,0]))
        return (-res.fun,res.x,rotate_eigstates(res.x[0],res.x[1]))
    
    def dynamic_phase_fixer_4(self,state,starting_phase_difference,eig000,eig101):
        """Rotates projections of state to eig000 and eig101 until their phase difference is the 
        same as in the beginning. Doesn't work if eig000 or eig101 coefficient is zero."""
        eig000coef = (eig000.dag()*state)
        eig101coef = (eig101.dag()*state)
#        phase_difference = np.angle(eig101coef.unit()[0][0][0]*(eig000coef.unit()[0][0][0])**(-1))
        phase_difference = np.angle(eig101coef[0][0][0])-np.angle((eig000coef[0][0][0]))
        if phase_difference < 0:
            phase_difference = 2*np.pi+phase_difference
        phase_change = (phase_difference-starting_phase_difference)/2
        if phase_change < 0:
            phase_change = phase_change+np.pi
        eig000proj = eig000coef*eig000
        eig101proj = eig101coef*eig101
        return (state-eig000proj+eig000proj*np.exp(1.0j*phase_change)-eig101proj+eig101proj*np.exp(-1.0j*phase_change),phase_change)
    
    def dynamic_phase_fixer_5(self,state,target_state,eig000,eig101):
        eig000proj = (eig000.dag()*state)*eig000
        eig101proj = (eig101.dag()*state)*eig101
        def rotate_eigstates(phi):
            return state-eig000proj+eig000proj*np.exp(1.0j*(-phi))-eig101proj+eig101proj*np.exp(1.0j*phi)
        def negative_target_closeness(phi):
            return -np.abs((target_state.dag()*rotate_eigstates(phi))[0][0][0])
        res = sc.optimize.minimize_scalar(negative_target_closeness)#,bounds=[0,np.pi],method='bounded')
        return (-res.fun,res.x,rotate_eigstates(res.x))
    
    def rotate_qubit_states(self,state,eig000,eig101,phi):
        eig000proj = (eig000.dag()*state)*eig000
        eig101proj = (eig101.dag()*state)*eig101
        return state-eig000proj+eig000proj*np.exp(1.0j*(-phi))-eig101proj+eig101proj*np.exp(1.0j*phi)

"""Options for the time dependence of the Hs[1] term in the Hamiltonian"""
def sine_pulse(t,args = {'T':0.5e-6,'A':0.2}):
    A = args['A']
    T = args['T']
    if(t > T):
        return 1
    else:
        return 1-A*np.sin(t*np.pi/T)

def square_pulse(t,args = {'T':0.5e-6, 'A':0.2}):
    A = args['A']
    T = args['T']
    if(t > T):
        return 1
    else:
        return 1-A

def no_pulse(t,args = {'T':0.5e-6,'A':0.2}):
    return 1

"""Plotting helper"""

def plot_state_evolution(title,times,states,state):
    state_coefs = [(state.dag()*s)[0][0][0] for s in states]
    fig = pl.figure()
    pl.title(title)
    pl.plot(times,np.real(state_coefs))
    pl.plot(times,np.imag(state_coefs))
    pl.plot(times,np.abs(state_coefs))
    pl.legend(["Real","Imaginary","Norm"])
    pl.xlabel("Time")
    return fig