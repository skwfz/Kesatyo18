# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:09:53 2018

@author: rissans2
"""

from qutip import *
import numpy as np
from matplotlib import pyplot as pl
import scipy as sc
import util
import tunable_coupling_util as tcu
import time
import datetime
#%%
"""Simulation constants and the functions directly dependent on them"""
levels = 5 #Amount of energy levels for qubits

C = np.array([70e-15,200e-15,70e-15]) #Capasitance of the capacitor in the qubits 1,c,2 These values are different than in the article now!
Cc =  np.array([4e-15,4e-15]) #Capacitance between the coupler and gate 1 and 2
C12 =  np.array(0.1e-15) #Capacitance directly between qubits 1 and 2
omega = np.array([4e9,5e9,4e9])#Frequencies of qubits 1, c and 2

s = tcu.tunable_coupling_simulation(levels,C,Cc,C12,omega)
s.alpha[1] = 0#Corresponding to the center qubit being a harmonic oscillator
#%%
"""Finding out the eigenstates in the decoupled state corresponding to |001>, |100> and their symmetric and antisymmetric combinations."""
eigs = s.sch_Hamiltonian(s.omega[1]).eigenstates()
logical_eigens = s.eigenstates_logical_states(s.omega[1])
Eavg = (logical_eigens[0][0]+logical_eigens[1][0])/2
(symm_real,antisymm_real) = util.corresponding_kets_from_degenerate_subspace(logical_eigens[0][1],logical_eigens[1][1],s.symm_comp,s.antisymm_comp)
(state001_real,state100_real) = util.corresponding_kets_from_degenerate_subspace(logical_eigens[0][1],logical_eigens[1][1],s.state001_comp,s.state100_comp)
state000_real = util.find_best_ket_match(eigs[1],s.state000_comp)
state101_real = util.find_best_ket_match(eigs[1],s.state101_comp)

#%%

"""Just a normal simulation for plotting states below."""
S0 = state100_real
used_pulse = tcu.square_pulse
T = 5e-7
H = s.simulation_Hamiltonian(used_pulse=used_pulse)
times = np.linspace(0,T,1000)
A = 0.25#s.theoretical_pulse_amplitude(used_pulse,T,s.gef)
args = {"A":A, "T":T}
opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1000)
res_onesimulation = mesolve(H,S0,times,[],[],options=opts,args = args)


#%%
"""The simulation with 100 different square_pulse amplitudes and 1000 different times using a single starting state. """
#S0 = state100_real
#Scompare = state001_real*(1.0j)
#"""The simulation parameters given to the solver"""
#used_pulse = tcu.square_pulse
#T = 5e-7
#H = s.simulation_Hamiltonian(used_pulse=used_pulse)
#N = 100 #For how many different amplitudes the simulation is run
#fidelities = np.zeros((N,1000))
#phase_differences = np.zeros((N,1000))
#times = np.linspace(0,T,1000)
#As = np.linspace((s.omega[1]-s.omega[0])/s.omega[1],0,N)
#for k in range(0,N):
#    #A = s.theoretical_pulse_amplitude(used_pulse = used_pulse, T=T, used_gef = geffint)
#    A = As[k]
#    """Running the simulation"""
#    args = {'T':T,'A':A}
#    S0 = state100_real
#    opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1000)
#    res = mesolve(H,S0,times,[],[],options=opts,args = args)
#    fidelities[k] = np.array([abs((Scompare.dag()*state).data.toarray()[0][0]) for state in res.states])
#    phase_differences[k] = np.array([util.angle_between_states(Scompare,state) for state in res.states])
#    print(k)
#np.savez('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\Couplingalpha_0\squarepulse_100amplitudes_alpha0.npz',times=times,As=As,fidelities=fidelities,phase_differences=phase_differences)
#%%
"""200 different square pulse amplitudes, 16 different starting states for each."""
#ket0 = basis(s.levels,0)
#ket1 = basis(s.levels,1)
#ketx = (basis(s.levels,0)+basis(s.levels,1)).unit()
#kety = (basis(s.levels,0)+1.0j*basis(s.levels,1)).unit()
#kets = [ket0,ket1,ketx,kety]
#
#used_pulse = tcu.square_pulse
#
#T = 5e-7
#H = s.simulation_Hamiltonian(used_pulse=used_pulse)
#N = 200 #For how many different amplitudes the simulation is run
#avg_fidelities = np.zeros((N,1000))
#avg_phase_differences = np.zeros((N,1000))
#times = np.linspace(0,T,1000)
#As = np.linspace(2*(s.omega[1]-s.omega[0])/s.omega[1],0,N)
#starting_states_coefs = util.g_e_x_y_coefficients()
#
#start = time.time()#For measuring the time it took to calculate everything
#
#for k in range(0,N):
#
#    A = As[k]#to be replaced
#    opts = solver.Options(num_cpus=6, atol= 1e-11, rtol= 1e-9,store_states=True, nsteps=1000)
#    args = {'T':T,'A':A}
#    
#    #Fidelities and phase differences for this all the different states on this amplitude
#    amplitude_fidelities = []
#    amplitude_phase_differences = []
#    
#    for coefs in starting_states_coefs:
#        S0 = state000_real*coefs[0]+state001_real*coefs[1]+state100_real*coefs[2]+state101_real*coefs[3]
#        Scompare = state000_real*coefs[0]+1.0j*state001_real*coefs[2]\
#            +1.0j*state100_real*coefs[1]+state101_real*coefs[3]#iswapped state
#        
#        res = mesolve(H,S0,times,[],[],options=opts,args = args)
#        
#        amplitude_fidelities.append(np.array([abs((Scompare.dag()*state).data.toarray()[0][0])**2 for state in res.states]))
#        amplitude_phase_differences.append(np.array([util.angle_between_states(Scompare,state) for state in res.states]))
#        print("Coefficients ",coefs, " calculated for amplitude ", A)
#    
#    avg_fidelities[k] = np.array([np.mean(fs) for fs in zip(*amplitude_fidelities)])
#    avg_phase_differences[k] = np.array([np.mean(pds) for pds in zip(*amplitude_phase_differences)])
#    print("Fidelities calculated for ", k, ". amplitude: ", A)
#
#end = time.time()
#
#print("Amount of time elapsed: ",  str(datetime.timedelta(seconds=end-start)))   
#
#np.savez('Z:\Documents\\2018Kesätyö\Koodia\Tunable_qubits_data\Couplingalpha_0\squarepulse_200amplitudes_16states_avgfidelity_alpha0.npz',times=times,As=As,avg_fidelities=avg_fidelities,avg_phase_differences=avg_phase_differences)
#

#%%

"""Plotting"""
pl.close('all')

"""Plotting"""
#|001>-state
fig1 = tcu.plot_state_evolution("State "+r"|001$\rangle$"+" superposition coefficient",times,res_onesimulation.states,s.state001_comp)

#|100>-state
fig2 = tcu.plot_state_evolution("State "+r"|100$\rangle$"+" superposition coefficient",times,res_onesimulation.states,s.state100_comp)

#|010>-state. Qubits 1 and 2 should exchange energy here.
fig3 = tcu.plot_state_evolution("State "+r"|010$\rangle$"+" superposition coefficient",times,res_onesimulation.states,s.state010_comp)

#|111>-state. Qubits 1 and 2 should exchange energy here.
fig3 = tcu.plot_state_evolution("State "+r"|111$\rangle$"+" superposition coefficient",times,res_onesimulation.states,s.state111_comp)

#Calculated eigenstate in Schrödinger picture corresponding to |100>
fig4 = tcu.plot_state_evolution("The real state "+r"|100$\rangle$"+" superposition coefficient",times,res_onesimulation.states,state100_real)

#Calculated eigenstate in Schrödinger picture corresponding to |001>
fig4 = tcu.plot_state_evolution("The real state "+r"|001$\rangle$"+" superposition coefficient",times,res_onesimulation.states,state001_real)

#Calculated eigenstate in Schrödinger picture corresponding to |001>+|100>
fig4 = tcu.plot_state_evolution("The real state "+r"|001$\rangle$+|100$\rangle$"+" superposition coefficient",times,res_onesimulation.states,symm_real)

#Calculated eigenstate in Schrödinger picture corresponding to |001>-|100>
fig4 = tcu.plot_state_evolution("The real state "+r"|001$\rangle$-|100$\rangle$"+" superposition coefficient",times,res_onesimulation.states,antisymm_real)

#fig5 = tcu.plot_state_evolution("The operation state corresponding to "+r"|001$\rangle$"+" superposition coefficient",times,res.states,state001_operation)

#fig6 = tcu.plot_state_evolution("The operation state corresponding to "+r"|100$\rangle$"+" superposition coefficient",times,res.states,state100_operation)
